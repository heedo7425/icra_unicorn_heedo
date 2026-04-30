"""
acados OCP for dynamic-bicycle + Pacejka tracking MPC with GP residual injection.

Based on `mpc/mpcc_ocp.py`, this variant extends the stage parameter vector with
3 residual slots: Δvx, Δvy, Δω. These are added to the model RHS so that the
OCP propagates corrected dynamics:

    xdot_corrected = xdot_pacejka + [0, 0, 0, Δvx, Δvy, Δω, 0]

State order (from vehicle_model.py): [s, n, dpsi, vx, vy, omega, delta]
Residual injection targets indices 3,4,5 → vx, vy, omega channels.

Param vector per stage (9D): [kappa, theta, kappa_z, mu, vx_ref, n_ref, Δvx, Δvy, Δω]
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

from upenn_mpc.vehicle_model import NX, NU, NP, build_dynamic_bicycle_model  # noqa: E402


NP_GP = NP + 3  # 6 base + 3 residual (Δvx, Δvy, Δω)
RESIDUAL_STATE_IDX = (3, 4, 5)  # vx, vy, omega


def build_tracking_ocp_upenn(
    vp: dict,
    mpc_cfg: dict,
    codegen_dir: str = "/tmp/upenn_mpc_c_generated",
    model_name: str = "tracking_ocp_upenn",
) -> "AcadosOcpSolver":
    """acados OCP with 3 extra stage params for GP residual (Δvx, Δvy, Δω).

    When residual = 0 (GP off / not yet trained), behaves identically to base
    tracking OCP.
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(f"acados/casadi missing: {_IMPORT_ERROR}")

    model = build_dynamic_bicycle_model(vp)

    # Extend params with 3 residual slots.
    residual_sym = ca.SX.sym("gp_residual", 3)
    model.p = ca.vertcat(model.p, residual_sym)

    # Inject residual into dynamics RHS: xdot[3:6] += residual.
    injection = ca.vertcat(
        0, 0, 0,
        residual_sym[0], residual_sym[1], residual_sym[2],
        0,
    )
    model.f_expl_expr = model.f_expl_expr + injection
    # Implicit form (if acados uses f_impl_expr for ERK/IRK): xdot - f_expl = 0.
    if hasattr(model, "f_impl_expr") and model.f_impl_expr is not None:
        model.f_impl_expr = model.xdot - model.f_expl_expr

    # Rename to avoid codegen collision (base mpc / comparison solver).
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

    # Cost (same as base OCP).
    s, n, dpsi, vx, vy, omega, delta = ca.vertsplit(model.x, 1)
    u_ddelta, u_ax = ca.vertsplit(model.u, 1)
    # First 6 params are original (kappa, theta, kappa_z, mu, vx_ref, n_ref).
    # Last 3 are residuals (not referenced in cost — only in dynamics).
    kappa, theta, kappa_z, mu, vx_ref, n_ref = [model.p[i] for i in range(NP)]

    y = ca.vertcat(
        n - n_ref, dpsi, vx - vx_ref, vy, omega, delta, u_ddelta, u_ax,
    )
    y_N = ca.vertcat(
        n - n_ref, dpsi, vx - vx_ref, vy, omega, delta,
    )
    ny = int(y.shape[0])
    ny_e = int(y_N.shape[0])

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_N
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    W_stage = np.diag([
        mpc_cfg.get("w_d", 10.0),
        mpc_cfg.get("w_dpsi", 5.0),
        mpc_cfg.get("w_vx", 1.0),
        mpc_cfg.get("w_vy", 0.5),
        mpc_cfg.get("w_omega", 0.1),
        mpc_cfg.get("w_steer", 0.01),
        mpc_cfg.get("w_u_steer_rate", 0.5),
        mpc_cfg.get("w_u_accel", 0.05),
    ])
    W_terminal_scale = float(mpc_cfg.get("w_terminal_scale", 10.0))
    W_N = W_terminal_scale * np.diag([
        mpc_cfg.get("w_d", 10.0),
        mpc_cfg.get("w_dpsi", 5.0),
        mpc_cfg.get("w_vx", 1.0),
        mpc_cfg.get("w_vy", 0.5),
        mpc_cfg.get("w_omega", 0.1),
        mpc_cfg.get("w_steer", 0.01),
    ])
    ocp.cost.W = W_stage
    ocp.cost.W_e = W_N

    # Constraints.
    v_max = float(mpc_cfg.get("v_max", 8.0))
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

    # Friction circle (soft nonlinear) — yaml `friction_circle` 로 toggle.
    ### HJ : friction_circle 자체를 NaN 진단용으로 끌 수 있게 conditional.
    ###       yaml 의 `friction_circle: false` 면 constraint 미생성.
    fric_circle_enable = bool(mpc_cfg.get("friction_circle", True))
    if fric_circle_enable:
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

    # Solver options.
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ### HJ : ERK 적분 sub-step 수. default=1 (dt=0.05 한 번에 적분).
    ###       sim 은 dt=0.01 로 5 sub-step 적분 → MPC 와 정밀도 mismatch (가설:
    ###       이게 ratio 0.83 dynamic understeer gap 17% 의 원인).
    ###       5 로 늘려 dt_internal=0.01 매칭. solve time 약간 ↑.
    ocp.solver_options.sim_method_num_steps = 5
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ### HJ : 1 → 3 시도했으나 RESCUE 24→31 로 늘어남 → 1 로 원복.
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.qp_solver_warm_start = 2
    ### HJ : 0.1 시도했으나 QP_FAILED 32% 폭증 → 0.01 원복. LM 너무 강하면 SQP
    ###       step 왜곡되어 resulting QP 풀기 어려움.
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
    params_per_stage: np.ndarray,  # (N+1, NP_GP)
) -> tuple:
    """Single-shot solve with GP residual params. Returns (u0, status, info)."""
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
    from upenn_mpc.vehicle_model import default_vehicle_params

    mpc_cfg = {
        "N_horizon": 20,
        "dt": 0.05,
        "v_max": 8.0,
        "max_steer": 0.4,
        "max_steer_rate": 3.5,
        "max_accel": 5.0,
        "max_decel": -6.0,
    }
    vp = default_vehicle_params()
    solver = build_tracking_ocp_upenn(vp, mpc_cfg)

    x0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    params = np.tile(
        [0.1, 0.0, 0.0, 0.7, 3.0, 0.0, 0.0, 0.0, 0.0],  # residual=0
        (mpc_cfg["N_horizon"] + 1, 1),
    )
    u0, status, info = solve_once_upenn(solver, x0, params)
    print(f"status={status} u0={u0} solve_time={info['solve_time_ms']:.2f}ms")

    # With nonzero residual.
    params[:, NP:] = [0.5, -0.2, 0.3]
    u0, status, info = solve_once_upenn(solver, x0, params)
    print(f"[residual] status={status} u0={u0} solve_time={info['solve_time_ms']:.2f}ms")
