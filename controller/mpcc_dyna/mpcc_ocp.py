"""
Dynamic single-track MPCC OCP — **Liniger 정통 정렬**.

Reference: alexliniger/MPCC (C++/Cost/cost.cpp + Params/cost.json).
State (8): [X, Y, psi, vx, vy, omega, delta, theta]
Input (3): [u_ddelta, u_a, vs]

Liniger residual structure (per-stage NLS):
    eC                    qC      contour error
    eL                    qL      lag error
    psi - psi_ref         qMu     heading deviation (작게; psi_ref 는 build_preview
                                  에서 unwrap + psi_anchor 정규화 → wrap-safe)
    beta = vy/vx_reg      qBeta   slip-angle reg (small-angle approx of atan(vy/vx))
    omega                 qR      yaw-rate reg
    delta                 qDelta  steering reg (Liniger r_delta, 매우 작음)
    u_a                   qA      accel rate-input reg
    u_ddelta              qDDelta steering rate-input reg
    vs                    qVs     progress reward (yref=vs_max → push to bound)

Terminal: contour/lag/heading/yaw-rate, with qC_N_mult / qR_N_mult (Liniger).

Constraints (Liniger):
    box: vx ∈ [v_min, v_max], delta ∈ [-max_steer, max_steer]
    h-soft: track_inner, track_outer, |F_yf|, |F_yr|  (4 rows, all soft slack)

Wrap handling:
    Liniger 의 `theta_ref += 2π·round((phi-theta_ref)/(2π))` 는 host 단에서
    reference_builder 의 psi_anchor + np.unwrap 으로 처리 (ψ branch 정렬).

Solver: SQP_RTI + Gauss-Newton + Levenberg-Marquardt (Liniger 와 동일 family).
"""
from __future__ import annotations

import os

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

from mpcc_dyna.vehicle_model import build_dynamic_mpcc_model, NX, NU, NP


def build_mpcc_ocp(
    mpc_cfg: dict,
    vp: dict,
    codegen_dir: str = "/tmp/mpcc_dyna_codegen",
) -> "AcadosOcpSolver":
    if _IMPORT_ERROR is not None:
        raise ImportError(f"acados/casadi not available: {_IMPORT_ERROR}")

    os.makedirs(codegen_dir, exist_ok=True)
    model = build_dynamic_mpcc_model(vp)

    N = int(mpc_cfg.get("N_horizon", 30))
    dt = float(mpc_cfg.get("dt", 0.05))

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt
    ocp.dims.N = N
    ocp.parameter_values = np.zeros(NP, dtype=np.float64)

    # ---- Symbol shortcuts ----
    X_, Y_, psi, vx, vy, omega, delta, theta = ca.vertsplit(model.x, 1)
    u_ddelta, u_a, vs = ca.vertsplit(model.u, 1)
    psi_ref = model.p[2]

    # ---- Liniger weights (cost.json defaults) ----
    qC      = float(mpc_cfg.get("qC",      0.1))
    qL      = float(mpc_cfg.get("qL",    500.0))
    qMu     = float(mpc_cfg.get("qMu",     0.001))
    qBeta   = float(mpc_cfg.get("qBeta",   0.01))
    qR      = float(mpc_cfg.get("qR",      0.01))
    qDelta  = float(mpc_cfg.get("qDelta",  1e-6))
    qA      = float(mpc_cfg.get("qA",      1e-6))
    qDDelta = float(mpc_cfg.get("qDDelta", 5e-3))
    qVs     = float(mpc_cfg.get("qVs",     0.02))
    qC_Nmult = float(mpc_cfg.get("qC_term", 10.0))   # Liniger qCNmult=10
    qR_Nmult = float(mpc_cfg.get("qR_term", 10.0))   # Liniger qRNmult=10

    # ---- NLS residual (9 stage, 4 terminal) ----
    eC = model.eC_expr
    eL = model.eL_expr
    eps_v = 0.5
    vx_reg = ca.sqrt(vx * vx + eps_v * eps_v)
    beta = vy / vx_reg                             # small-angle slip approx (Liniger linearizes)

    y = ca.vertcat(
        eC,                       # 0  qC
        eL,                       # 1  qL
        psi - psi_ref,            # 2  qMu
        beta,                     # 3  qBeta
        omega,                    # 4  qR
        delta,                    # 5  qDelta
        u_a,                      # 6  qA
        u_ddelta,                 # 7  qDDelta
        vs,                       # 8  qVs   (yref=vs_max → push)
    )
    W_stage = np.diag([qC, qL, qMu, qBeta, qR, qDelta, qA, qDDelta, qVs])

    # Terminal: contour/lag/heading/yaw-rate (Liniger 도 terminal cost 가 stage 부분집합).
    y_term = ca.vertcat(eC, eL, psi - psi_ref, omega)
    W_term = np.diag([qC * qC_Nmult, qL, qMu, qR * qR_Nmult])

    # Initial yref (build-time defaults; node 가 reload 시 갱신)
    vs_max_init = float(mpc_cfg.get("vs_max", 4.0))
    yref_init = np.zeros(int(y.shape[0]))
    yref_init[8] = vs_max_init        # progress: (vs - vs_max)² ⇒ push vs → vs_max

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_term
    ocp.cost.yref = yref_init
    ocp.cost.yref_e = np.zeros(int(y_term.shape[0]))
    ocp.cost.W = W_stage
    ocp.cost.W_e = W_term

    # ---- Box constraints (vx idx=3, delta idx=6) ----
    v_min = float(mpc_cfg.get("v_min", 0.0))
    v_max = float(mpc_cfg.get("v_max", 3.0))
    max_steer = float(mpc_cfg.get("max_steer", 0.4))
    max_steer_rate = float(mpc_cfg.get("max_steer_rate", 3.0))
    max_accel = float(mpc_cfg.get("max_accel", 3.0))
    max_decel = float(mpc_cfg.get("max_decel", -3.0))
    vs_min = float(mpc_cfg.get("vs_min", 0.0))
    vs_max = float(mpc_cfg.get("vs_max", 4.0))

    ocp.constraints.idxbx = np.array([3, 6])
    ocp.constraints.lbx = np.array([v_min, -max_steer])
    ocp.constraints.ubx = np.array([v_max, max_steer])
    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.lbu = np.array([-max_steer_rate, max_decel, vs_min])
    ocp.constraints.ubu = np.array([max_steer_rate, max_accel, vs_max])
    ocp.constraints.x0 = np.zeros(NX)

    # ---- h-constraints: track (Liniger half-space) + tire ellipse (Liniger sc_*_tire) ----
    h_track_inner = -model.h_track_inner_expr
    h_track_outer = -model.h_track_outer_expr

    # tire ellipse — Liniger: |F_y| ≤ μ·D·F_z (peak Pacejka). margin 으로 안쪽 cap.
    fric_margin = float(mpc_cfg.get("friction_margin", 0.85))
    mu_v = float(vp.get("mu", 0.85))
    m_v = float(vp.get("m", 3.54))
    l_f_v = float(vp.get("l_f", 0.162))
    l_r_v = float(vp.get("l_r", 0.145))
    l_wb_v = float(vp.get("l_wb", l_f_v + l_r_v))
    Df_v = float(vp.get("Df", 0.65))
    Dr_v = float(vp.get("Dr", 0.62))
    g_v = 9.81

    Fzf = m_v * g_v * l_r_v / l_wb_v
    Fzr = m_v * g_v * l_f_v / l_wb_v
    Fy_max_f = fric_margin * mu_v * Df_v * Fzf
    Fy_max_r = fric_margin * mu_v * Dr_v * Fzr
    h_fric_f = model.fyf_expr * model.fyf_expr - (Fy_max_f ** 2)
    h_fric_r = model.fyr_expr * model.fyr_expr - (Fy_max_r ** 2)

    ocp.model.con_h_expr = ca.vertcat(h_track_inner, h_track_outer,
                                      h_fric_f, h_fric_r)
    ocp.constraints.lh = np.array([-1e12, -1e12, -1e12, -1e12])
    ocp.constraints.uh = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.constraints.idxsh = np.arange(4)

    # Liniger sc_*_track / sc_*_tire — quadratic + linear slack penalty.
    sc_track_q = float(mpc_cfg.get("sc_quad_track", 100.0))
    sc_track_l = float(mpc_cfg.get("sc_lin_track",   1.0))
    sc_tire_q  = float(mpc_cfg.get("sc_quad_tire",   1.0))
    sc_tire_l  = float(mpc_cfg.get("sc_lin_tire",    0.1))
    Z_quad = np.array([sc_track_q, sc_track_q, sc_tire_q, sc_tire_q], dtype=np.float64)
    z_lin  = np.array([sc_track_l, sc_track_l, sc_tire_l, sc_tire_l], dtype=np.float64)
    ocp.cost.Zl = Z_quad.copy()
    ocp.cost.Zu = Z_quad.copy()
    ocp.cost.zl = z_lin.copy()
    ocp.cost.zu = z_lin.copy()

    # ---- Solver options ----
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_steps = int(mpc_cfg.get("sim_method_num_steps", 5))
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = int(mpc_cfg.get("nlp_solver_max_iter", 2))
    ocp.solver_options.qp_solver_iter_max = int(mpc_cfg.get("qp_solver_iter_max", 200))
    ocp.solver_options.qp_solver_warm_start = 2
    ocp.solver_options.levenberg_marquardt = float(mpc_cfg.get("levenberg_marquardt", 1.0))
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.print_level = 0

    ocp.code_export_directory = codegen_dir
    solver = AcadosOcpSolver(
        ocp, json_file=f"{codegen_dir}/acados_ocp_mpcc.json", verbose=False,
    )
    return solver


def solve_once(
    solver: "AcadosOcpSolver",
    x0: np.ndarray,
    params_per_stage: np.ndarray,
) -> tuple:
    assert x0.shape[0] == NX
    assert params_per_stage.shape[1] == NP
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
