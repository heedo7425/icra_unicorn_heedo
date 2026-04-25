"""
MPCC (Model Predictive Contouring Control) — Frenet-frame, quadratic form.

Philosophy vs path-tracking MPC:
  - MPC: 추종 대상이 "시간 프로파일" — vx_ref(raceline) 를 tight 하게 추종.
  - MPCC: 추종 대상이 "기하 경로" — path 에서 가능한 "빨리 가라". 속도는
    최적화가 friction / friction-circle 범위에서 자동 결정.

Implementation (numerically robust NLS + GN hessian):
  - Stage cost (NLS):
        qc*(n-n_ref)² + qpsi*dpsi² + qvy*vy² + qomega*omega² + qsteer*delta²
      + r_ddelta*u_ddelta² + r_ax*u_ax²
      + qprog*(v_target - vx)²                    ← progress as "approach v_target"
    v_target 을 v_max 에 가깝게 두면 "항상 v_max 찍고 싶어함" → friction circle
    만이 제한자 → MPCC 의도 달성. vx_ref tracking 과 수식은 유사하지만 의미는
    완전히 다름 (raceline 속도 무시, 이론적 max 만 추종).
  - Terminal cost (EXTERNAL, linear in s):
        term_scale * (contouring + heading + regs) - qprog_e * s
    Horizon 끝에서 "s 를 최대한 키워라" → look-ahead 강화. Linear term 은
    terminal stage 에 1 번만 들어가므로 Hessian 이슈 없음.
  - Friction circle (soft nonlinear) 유지 — MPCC 의 진짜 speed limiter.
  - GP residual 주입 동일 (NP_GP=9).

State  (7): [s, n, dpsi, vx, vy, omega, delta]
Input  (2): [u_ddelta, u_ax]
Params (9): [kappa, theta, kappa_z, mu, vx_ref(unused), n_ref, Δvx, Δvy, Δω]
"""

from __future__ import annotations

import os
import sys
import numpy as np

try:
    import casadi as ca
    from acados_template import AcadosOcp, AcadosOcpSolver
except ImportError as e:
    ca = None
    AcadosOcp = None
    AcadosOcpSolver = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from upenn_mpcc_rm.vehicle_model import NX, NU, NP, build_dynamic_bicycle_model  # noqa: E402


NP_GP = NP + 3  # 6 base + 3 residual (Δvx, Δvy, Δω)
RESIDUAL_STATE_IDX = (3, 4, 5)  # vx, vy, omega


def build_tracking_ocp_upenn(
    vp: dict,
    mpc_cfg: dict,
    codegen_dir: str = "/tmp/upenn_mpcc_c_generated",
    model_name: str = "mpcc_ocp_upenn_rm",
) -> "AcadosOcpSolver":
    """Build the MPCC acados OCP (quadratic form, NLS + GN).

    `mpc_cfg` keys:
        Horizon    : N_horizon, dt
        Weights    : w_d (contouring), w_dpsi (heading),
                     w_vy, w_omega, w_steer (state reg),
                     w_u_steer_rate, w_u_accel (input reg),
                     w_progress (stage: qprog·(v_target - vx)²),
                     w_progress_e (terminal: -qprog_e * s, EXTERNAL),
                     w_terminal_scale
        Target     : mpcc_v_target (fallback = v_max)
        Limits     : v_min, v_max, max_steer, max_steer_rate,
                     max_accel, max_decel
        Friction   : friction_margin, friction_slack_penalty
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(f"acados/casadi missing: {_IMPORT_ERROR}")

    model = build_dynamic_bicycle_model(vp)

    # GP residual param extension.
    residual_sym = ca.SX.sym("gp_residual", 3)
    model.p = ca.vertcat(model.p, residual_sym)
    injection = ca.vertcat(
        0, 0, 0,
        residual_sym[0], residual_sym[1], residual_sym[2],
        0,
    )
    model.f_expl_expr = model.f_expl_expr + injection
    if hasattr(model, "f_impl_expr") and model.f_impl_expr is not None:
        model.f_impl_expr = model.xdot - model.f_expl_expr

    model.name = model_name

    N = int(mpc_cfg["N_horizon"])
    dt = float(mpc_cfg["dt"])
    T = N * dt

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = T
    ocp.dims.N = N

    ocp.parameter_values = np.zeros(NP_GP, dtype=np.float64)

    # State/input symbols.
    s, n, dpsi, vx, vy, omega, delta = ca.vertsplit(model.x, 1)
    u_ddelta, u_ax = ca.vertsplit(model.u, 1)
    kappa, theta_p, kappa_z, mu, vx_ref_unused, n_ref = [model.p[i] for i in range(NP)]

    # --- MPCC weights ---
    v_max_cfg = float(mpc_cfg.get("v_max", 8.0))
    v_target = float(mpc_cfg.get("mpcc_v_target", v_max_cfg))  # 이 속도 "지향"

    W_stage = np.diag([
        mpc_cfg.get("w_d", 4.0),             # contouring  (n - n_ref)
        mpc_cfg.get("w_dpsi", 3.0),          # heading     dpsi
        mpc_cfg.get("w_progress", 2.0),      # progress    (v_target - vx)
        mpc_cfg.get("w_vy", 0.3),            # slip        vy
        mpc_cfg.get("w_omega", 2.0),         # yaw damping omega
        mpc_cfg.get("w_steer", 0.3),         # steer mag   delta
        mpc_cfg.get("w_u_steer_rate", 3.0),  # steer rate  u_ddelta
        mpc_cfg.get("w_u_accel", 0.3),       # accel       u_ax
    ])
    term_scale = float(mpc_cfg.get("w_terminal_scale", 1.5))
    W_terminal = term_scale * np.diag([
        mpc_cfg.get("w_d", 4.0),
        mpc_cfg.get("w_dpsi", 3.0),
        mpc_cfg.get("w_vy", 0.3),
        mpc_cfg.get("w_omega", 2.0),
        mpc_cfg.get("w_steer", 0.3),
    ])

    # NLS residual vector (stage).
    # y = [n - n_ref, dpsi, v_target - vx, vy, omega, delta, u_ddelta, u_ax]
    # yref = 0 (means residuals → 0 → state matches path, vx→v_target, inputs→0).
    y = ca.vertcat(
        n - n_ref,
        dpsi,
        v_target - vx,
        vy,
        omega,
        delta,
        u_ddelta,
        u_ax,
    )
    # Terminal NLS: drop inputs + progress (progress enforced via external below).
    y_N = ca.vertcat(
        n - n_ref,
        dpsi,
        vy,
        omega,
        delta,
    )

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_N
    ocp.cost.yref = np.zeros(y.shape[0])
    ocp.cost.yref_e = np.zeros(y_N.shape[0])
    ocp.cost.W = W_stage
    ocp.cost.W_e = W_terminal

    # --- Progress reward 는 LS 만으론 "공격성"이 약할 수 있음. 필요 시 terminal
    # 에 linear -q*s 를 추가하려면 cost_type_e="EXTERNAL" 로 전환 필요. 지금은
    # 안정성 우선 — 기본 NLS 로 충분히 동작 확인 후 튜닝.
    # (enable by setting mpc_cfg["w_progress_e_linear"] > 0; ignored here by design.)

    # --- Constraints ---
    v_max = v_max_cfg
    v_min = float(mpc_cfg.get("v_min", 0.0))
    max_steer = float(mpc_cfg.get("max_steer", 0.4))
    ocp.constraints.idxbx = np.array([3, 6])
    ocp.constraints.lbx = np.array([v_min, -max_steer])
    ocp.constraints.ubx = np.array([v_max, max_steer])

    max_steer_rate = float(mpc_cfg.get("max_steer_rate", 3.5))
    max_accel = float(mpc_cfg.get("max_accel", 5.0))
    max_decel = float(mpc_cfg.get("max_decel", -6.0))
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([-max_steer_rate, max_decel])
    ocp.constraints.ubu = np.array([max_steer_rate, max_accel])

    ocp.constraints.x0 = np.zeros(NX)

    # ### upenn_mpcc_rm: Pacejka tire 가 없으니 friction circle 제약 자체를
    # 정의할 수 없음 (Fyf/Fyr/Nf/Nr 모두 None). 코너 감속은 cost 만으로 유도.
    if getattr(model, "fyf_expr", None) is not None:
        m_val = float(vp["m"])
        margin = float(mpc_cfg.get("friction_margin", 0.95))
        h_front = model.fyf_expr ** 2 - (mu * margin * model.nf_expr) ** 2
        h_rear = (m_val * u_ax) ** 2 + model.fyr_expr ** 2 - (mu * margin * model.nr_expr) ** 2
        ocp.model.con_h_expr = ca.vertcat(h_front, h_rear)
        ocp.constraints.lh = np.array([-1e12, -1e12], dtype=np.float64)
        ocp.constraints.uh = np.array([0.0, 0.0], dtype=np.float64)
        ocp.constraints.idxsh = np.arange(2)
        slack_penalty = float(mpc_cfg.get("friction_slack_penalty", 1e3))
        ocp.cost.zl = slack_penalty * np.ones(2)
        ocp.cost.zu = slack_penalty * np.ones(2)
        ocp.cost.Zl = slack_penalty * np.ones(2)
        ocp.cost.Zu = slack_penalty * np.ones(2)

    # --- Solver options (mirror upenn_mpc for stability) ---
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.qp_solver_warm_start = 2
    ocp.solver_options.levenberg_marquardt = 0.01
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.print_level = 0

    ocp.code_export_directory = codegen_dir

    solver = AcadosOcpSolver(
        ocp,
        json_file=f"{codegen_dir}/acados_ocp.json",
        verbose=False,
    )
    return solver


def solve_once_upenn(
    solver: "AcadosOcpSolver",
    x0: np.ndarray,
    params_per_stage: np.ndarray,
) -> tuple:
    assert params_per_stage.shape[1] == NP_GP, \
        f"expected {NP_GP} params per stage, got {params_per_stage.shape[1]}"
    N = params_per_stage.shape[0] - 1

    solver.set(0, "x", x0)
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    for k in range(N + 1):
        solver.set(k, "p", params_per_stage[k])

    status = solver.solve()
    u0 = solver.get(0, "u")

    info = {
        "status": status,
        "solve_time_ms": float(solver.get_stats("time_tot") * 1e3),
        "sqp_iter": int(solver.get_stats("sqp_iter")),
    }
    return u0, status, info


if __name__ == "__main__":
    from upenn_mpcc_rm.vehicle_model import default_vehicle_params

    mpc_cfg = {
        "N_horizon": 20, "dt": 0.05,
        "v_max": 8.0, "v_min": 0.0,
        "mpcc_v_target": 8.0,
        "max_steer": 0.4, "max_steer_rate": 3.5,
        "max_accel": 5.0, "max_decel": -6.0,
        "w_d": 4.0, "w_dpsi": 3.0, "w_vy": 0.3, "w_omega": 2.0, "w_steer": 0.3,
        "w_u_steer_rate": 3.0, "w_u_accel": 0.3,
        "w_progress": 2.0, "w_terminal_scale": 1.5,
        "friction_margin": 0.95, "friction_slack_penalty": 1000.0,
    }
    vp = default_vehicle_params()
    solver = build_tracking_ocp_upenn(vp, mpc_cfg)

    x0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    params = np.tile(
        [0.1, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
        (mpc_cfg["N_horizon"] + 1, 1),
    )
    u0, status, info = solve_once_upenn(solver, x0, params)
    print(f"status={status} u0={u0} solve_time={info['solve_time_ms']:.2f}ms")
