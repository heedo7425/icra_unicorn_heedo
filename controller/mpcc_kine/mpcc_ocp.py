"""
Kinematic MPCC OCP — minimal Liniger-style.

NLS cost; soft track slack; NO friction ellipse (kinematic, no tire forces).
SQP_RTI with Levenberg-Marquardt regularization for cold-start stability.
"""
from __future__ import annotations

import os
from typing import Optional

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

from mpcc_kine.vehicle_model import build_kinematic_mpcc_model, NX, NU, NP


def build_mpcc_ocp(
    mpc_cfg: dict,
    vp: dict,
    codegen_dir: str = "/tmp/mpcc_kine_codegen",
) -> "AcadosOcpSolver":
    """Kinematic MPCC OCP solver."""
    if _IMPORT_ERROR is not None:
        raise ImportError(f"acados/casadi not available: {_IMPORT_ERROR}")

    os.makedirs(codegen_dir, exist_ok=True)
    model = build_kinematic_mpcc_model(vp)

    N = int(mpc_cfg.get("N_horizon", 30))
    dt = float(mpc_cfg.get("dt", 0.05))

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt
    ocp.dims.N = N
    ocp.parameter_values = np.zeros(NP, dtype=np.float64)

    # ---- Symbol shortcuts ----
    X_, Y_, psi, v, delta, theta = ca.vertsplit(model.x, 1)
    u_ddelta, u_a, vs = ca.vertsplit(model.u, 1)
    psi_ref = model.p[2]

    # ---- Cost weights ----
    qC = float(mpc_cfg.get("qC", 1.0))
    qL = float(mpc_cfg.get("qL", 100.0))
    qMu = float(mpc_cfg.get("qMu", 0.1))
    qVs = float(mpc_cfg.get("qVs", 0.5))
    qV = float(mpc_cfg.get("qV", 0.5))    # forward-push toward v_max
    qVsv = float(mpc_cfg.get("qVsv", 1.0))   # ### HJ : vs ↔ v 동기화. 이게 없으면
                                              # vs 가 vs_target 으로 끌리고 v 와 어긋
                                              # 나서 θ drift 누적 → warm start cold reset.
    qDelta = float(mpc_cfg.get("qDelta", 0.0))
    qDDelta = float(mpc_cfg.get("qDDelta", 0.05))
    qA = float(mpc_cfg.get("qA", 0.05))
    qC_term = float(mpc_cfg.get("qC_term", 10.0)) * qC
    qL_term = float(mpc_cfg.get("qL_term", 1.0)) * qL

    vs_target = float(mpc_cfg.get("vs_target", mpc_cfg.get("vs_max", 8.0)))
    v_target = float(mpc_cfg.get("v_max", 3.0))

    # ---- NLS residual ----
    # y = [eC, eL, dpsi, delta, u_ddelta, u_a, vs_target-vs, v_target-v, vs-v]
    eC = model.eC_expr
    eL = model.eL_expr
    dpsi_err = psi - psi_ref
    y = ca.vertcat(eC, eL, dpsi_err, delta, u_ddelta, u_a,
                   vs_target - vs, v_target - v, vs - v)
    W_stage = np.diag([qC, qL, qMu, qDelta, qDDelta, qA, qVs, qV, qVsv])

    y_term = ca.vertcat(eC, eL, dpsi_err, delta)
    W_term = np.diag([qC_term, qL_term, qMu, qDelta])

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_term
    ocp.cost.yref = np.zeros(int(y.shape[0]))
    ocp.cost.yref_e = np.zeros(int(y_term.shape[0]))
    ocp.cost.W = W_stage
    ocp.cost.W_e = W_term

    # ---- Box constraints ----
    v_min = float(mpc_cfg.get("v_min", 0.0))
    v_max = float(mpc_cfg.get("v_max", 3.0))
    max_steer = float(mpc_cfg.get("max_steer", 0.4))
    max_steer_rate = float(mpc_cfg.get("max_steer_rate", 3.0))
    max_accel = float(mpc_cfg.get("max_accel", 3.0))
    max_decel = float(mpc_cfg.get("max_decel", -3.0))
    vs_min = float(mpc_cfg.get("vs_min", 0.0))
    vs_max = float(mpc_cfg.get("vs_max", 8.0))

    # state idx: 3=v, 4=delta
    ocp.constraints.idxbx = np.array([3, 4])
    ocp.constraints.lbx = np.array([v_min, -max_steer])
    ocp.constraints.ubx = np.array([v_max, max_steer])
    # input idx: 0=u_ddelta, 1=u_a, 2=vs
    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.lbu = np.array([-max_steer_rate, max_decel, vs_min])
    ocp.constraints.ubu = np.array([max_steer_rate, max_accel, vs_max])
    ocp.constraints.x0 = np.zeros(NX)

    # ---- Track half-space (soft slack only) ----
    # Want: car_proj - inner_proj ≥ 0 and outer_proj - car_proj ≥ 0
    # As h ≤ 0: -(car_proj - inner_proj) ≤ 0 and -(outer_proj - car_proj) ≤ 0
    h_track_inner = -model.h_track_inner_expr
    h_track_outer = -model.h_track_outer_expr
    ocp.model.con_h_expr = ca.vertcat(h_track_inner, h_track_outer)
    ocp.constraints.lh = np.array([-1e12, -1e12])
    ocp.constraints.uh = np.array([0.0, 0.0])
    ocp.constraints.idxsh = np.arange(2)

    track_pen = float(mpc_cfg.get("track_slack_penalty", 1000.0))
    z_vec = np.array([track_pen, track_pen], dtype=np.float64)
    ocp.cost.zl = z_vec.copy()
    ocp.cost.zu = z_vec.copy()
    ocp.cost.Zl = z_vec.copy()
    ocp.cost.Zu = z_vec.copy()

    # ---- Solver options ----
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_steps = 5
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.qp_solver_iter_max = 200
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
    """x0: (NX,)  params_per_stage: (N+1, NP)"""
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
