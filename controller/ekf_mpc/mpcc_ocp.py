"""
acados OCP for dynamic-bicycle + Pacejka tracking MPC in Frenet frame.

Cost (NONLINEAR_LS):
    y  = [n - n_ref, dpsi, vx - vx_ref, vy, omega, delta, u_ddelta, u_ax]
    l  = y^T W y
    y_N = [n - n_ref, dpsi, vx - vx_ref, vy, omega, delta]
    l_N = y_N^T W_N y_N

Inputs are regularized. "Tracking MPC" starts here; a progress term for MPCC
lands later (cost += -w_progress * s, or a via-point contouring term).
"""

from __future__ import annotations

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

import os
import sys
# Allow both `from mpc.mpcc_ocp import ...` (via sys.path hack in mpc_node.py)
# and direct `python3 controller/mpc/mpcc_ocp.py` invocation.
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from ekf_mpc.vehicle_model import (  # noqa: E402
    NX,
    NU,
    NP,
    build_dynamic_bicycle_model,
)


def build_tracking_ocp(
    vp: dict,
    mpc_cfg: dict,
    codegen_dir: str = "/tmp/mpc_c_generated",
) -> "AcadosOcpSolver":
    """Build acados OCP solver for tracking MPC.

    Parameters
    ----------
    vp : vehicle params (dict)
    mpc_cfg : MPC tuning (dict, subset of mpc_srx1.yaml 'mpc' section)
    codegen_dir : where acados writes generated C code
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(f"acados/casadi missing: {_IMPORT_ERROR}")

    model = build_dynamic_bicycle_model(vp)

    N = int(mpc_cfg["N_horizon"])
    dt = float(mpc_cfg["dt"])
    T = N * dt

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = T

    ocp.dims.N = N

    # ---- Parameters per stage ----
    ocp.parameter_values = np.zeros(NP, dtype=np.float64)

    # ---- Cost: NONLINEAR_LS ----
    s, n, dpsi, vx, vy, omega, delta = ca.vertsplit(model.x, 1)
    u_ddelta, u_ax = ca.vertsplit(model.u, 1)
    kappa, theta, kappa_z, mu, vx_ref, n_ref = ca.vertsplit(model.p, 1)

    # Residual vector (stage)
    y = ca.vertcat(
        n - n_ref,
        dpsi,
        vx - vx_ref,
        vy,
        omega,
        delta,
        u_ddelta,
        u_ax,
    )
    y_N = ca.vertcat(
        n - n_ref,
        dpsi,
        vx - vx_ref,
        vy,
        omega,
        delta,
    )
    ny = int(y.shape[0])
    ny_e = int(y_N.shape[0])

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_N
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # Weights (diag). Order MUST match y / y_N above.
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

    # ---- Constraints ----
    # State bounds: [s, n, dpsi, vx, vy, omega, delta]
    v_max = float(mpc_cfg.get("v_max", 8.0))
    v_min = float(mpc_cfg.get("v_min", 0.0))
    max_steer = float(mpc_cfg.get("max_steer", 0.4))

    # Only constrain vx and delta (s,n,dpsi,vy,omega soft/tracked via cost).
    ocp.constraints.idxbx = np.array([3, 6])  # indices: vx, delta
    ocp.constraints.lbx = np.array([v_min, -max_steer])
    ocp.constraints.ubx = np.array([v_max, max_steer])

    # Input bounds
    max_steer_rate = float(mpc_cfg.get("max_steer_rate", 3.5))
    max_accel = float(mpc_cfg.get("max_accel", 5.0))
    max_decel = float(mpc_cfg.get("max_decel", -6.0))
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([-max_steer_rate, max_decel])
    ocp.constraints.ubu = np.array([max_steer_rate, max_accel])

    # Initial state placeholder
    ocp.constraints.x0 = np.zeros(NX)

    # ---- Friction circle (soft nonlinear) ----
    # For each axle:  F_x² + F_y² ≤ (μ·margin·N)²
    #  Front: no drive → F_x_f ≈ 0, so constraint is |F_yf| ≤ μ·N_f
    #  Rear (RWD): F_x_r = m·u_ax (all longitudinal from rear).
    # Using squared form to keep expressions smooth (no sqrt Jacobian at 0).
    # Soft via slack ensures OCP stays feasible when transient exceeds limit.
    m_val = float(vp["m"])
    margin = float(mpc_cfg.get("friction_margin", 0.95))  # 95% of limit as safety
    h_front = model.fyf_expr ** 2 - (mu * margin * model.nf_expr) ** 2
    h_rear  = (m_val * u_ax) ** 2 + model.fyr_expr ** 2 - (mu * margin * model.nr_expr) ** 2
    ocp.model.con_h_expr = ca.vertcat(h_front, h_rear)
    ocp.constraints.lh = np.array([-1e12, -1e12], dtype=np.float64)
    ocp.constraints.uh = np.array([0.0, 0.0], dtype=np.float64)
    # Soft via slack — very high penalty so constraint is respected but never
    # makes OCP infeasible at numerical transients.
    ocp.constraints.idxsh = np.arange(2)
    slack_penalty = float(mpc_cfg.get("friction_slack_penalty", 1e3))
    ocp.cost.zl = slack_penalty * np.ones(2)
    ocp.cost.zu = slack_penalty * np.ones(2)
    ocp.cost.Zl = slack_penalty * np.ones(2)
    ocp.cost.Zu = slack_penalty * np.ones(2)

    # ---- Solver options ----
    # HPIPM with strong Levenberg-Marquardt regularization: keeps Hessian
    # well-conditioned even near the Pacejka low-speed singularity.
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.qp_solver_warm_start = 2
    # LM damping moderate: 1.0 was overwhelming the Hessian (solver returned
    # warm-start unchanged). With correct Pacejka signs and proper x0, a
    # small regularizer is enough to keep HPIPM well-conditioned.
    ocp.solver_options.levenberg_marquardt = 0.01
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.print_level = 0

    # Codegen output dir.
    ocp.code_export_directory = codegen_dir

    solver = AcadosOcpSolver(
        ocp,
        json_file=f"{codegen_dir}/acados_ocp.json",
        verbose=False,
    )
    return solver


def solve_once(
    solver: "AcadosOcpSolver",
    x0: np.ndarray,
    params_per_stage: np.ndarray,  # (N+1, NP)
) -> tuple[np.ndarray, int, dict]:
    """Single-shot solve. Returns (u0, status, info)."""
    N = params_per_stage.shape[0] - 1

    # Set initial condition. Use all three (set x, set lbx, set ubx) so every
    # acados version pins stage 0 to x0. Do NOT overwrite stage 1..N; acados'
    # internal warm-start from the previous solve gives a better starting
    # trajectory than the flat-x0 seed.
    solver.set(0, "x", x0)
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    # Set per-stage params
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
    # Smoke test: build + solve dummy.
    from ekf_mpc.vehicle_model import default_vehicle_params

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
    solver = build_tracking_ocp(vp, mpc_cfg)

    x0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])  # going 2 m/s
    params = np.tile(
        [0.1, 0.0, 0.0, 0.7, 3.0, 0.0],  # kappa=0.1, mu=0.7, vx_ref=3, n_ref=0
        (mpc_cfg["N_horizon"] + 1, 1),
    )
    u0, status, info = solve_once(solver, x0, params)
    print(f"status={status} u0={u0} solve_time={info['solve_time_ms']:.2f}ms")
