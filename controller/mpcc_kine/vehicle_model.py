"""
Kinematic single-track MPCC vehicle model.

Reference:
    - Liniger MPCC (https://github.com/alexliniger/MPCC) — Cartesian formulation,
      we strip dynamic + tire and keep kinematic only.
    - Niraj nonlinear MPCC (https://github.com/nirajbasnet/Nonlinear_MPCC_for_autonomous_racing)
      — same kinematic core, simpler input set.

State (6): [X, Y, psi, v, delta, theta]
    X, Y    : cartesian pose (map frame)
    psi     : heading
    v       : forward velocity
    delta   : steering angle (state, integrated from u_ddelta)
    theta   : virtual progress along reference path

Input (3): [u_ddelta, u_a, vs]
    u_ddelta : steering rate
    u_a      : longitudinal acceleration
    vs       : virtual progress speed (theta_dot = vs)

Params (8): [x_ref, y_ref, psi_ref, kappa_ref,
             inner_x, inner_y, outer_x, outer_y]
    spline-evaluated reference + track bounds at theta_pred[k].

Dynamics (kinematic single-track):
    dX     = v · cos(psi)
    dY     = v · sin(psi)
    dpsi   = v · tan(delta) / l_wb
    dv     = u_a
    ddelta = u_ddelta
    dtheta = vs

Cost (NLS):
    eC = -sin(psi_ref)·(X - x_ref) + cos(psi_ref)·(Y - y_ref)   contour
    eL =  cos(psi_ref)·(X - x_ref) + sin(psi_ref)·(Y - y_ref)   lag
    L  = qC·eC² + qL·eL² + qMu·(psi-psi_ref)²
       + qVs·(vs_target - vs)² + qV·(v_target - v)²
       + qDDelta·u_ddelta² + qA·u_a² + qDelta·delta²

Track constraint (soft slack):
    inner_proj ≤ car_proj ≤ outer_proj
    where car_proj = -sin(psi_ref)·X + cos(psi_ref)·Y
"""
from __future__ import annotations

import numpy as np

try:
    import casadi as ca
    from acados_template import AcadosModel
except ImportError as e:
    ca = None
    AcadosModel = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


STATE_NAMES = ("X", "Y", "psi", "v", "delta", "theta")
INPUT_NAMES = ("u_ddelta", "u_a", "vs")
PARAM_NAMES = (
    "x_ref", "y_ref", "psi_ref", "kappa_ref",
    "inner_x", "inner_y", "outer_x", "outer_y",
)
NX = len(STATE_NAMES)
NU = len(INPUT_NAMES)
NP = len(PARAM_NAMES)


def build_kinematic_mpcc_model(vp: dict) -> "AcadosModel":
    """Build kinematic single-track AcadosModel.

    vp keys: l_wb (or l_f + l_r). Other vehicle params unused (kinematic).
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(f"acados/casadi not available: {_IMPORT_ERROR}")

    # ---- Symbolics ----
    X = ca.SX.sym("X")
    Y = ca.SX.sym("Y")
    psi = ca.SX.sym("psi")
    v = ca.SX.sym("v")
    delta = ca.SX.sym("delta")
    theta = ca.SX.sym("theta")
    x = ca.vertcat(X, Y, psi, v, delta, theta)

    u_ddelta = ca.SX.sym("u_ddelta")
    u_a = ca.SX.sym("u_a")
    vs = ca.SX.sym("vs")
    u = ca.vertcat(u_ddelta, u_a, vs)

    x_ref = ca.SX.sym("x_ref")
    y_ref = ca.SX.sym("y_ref")
    psi_ref = ca.SX.sym("psi_ref")
    kappa_ref = ca.SX.sym("kappa_ref")
    inner_x = ca.SX.sym("inner_x")
    inner_y = ca.SX.sym("inner_y")
    outer_x = ca.SX.sym("outer_x")
    outer_y = ca.SX.sym("outer_y")
    p = ca.vertcat(x_ref, y_ref, psi_ref, kappa_ref,
                   inner_x, inner_y, outer_x, outer_y)

    # ---- Vehicle params (kinematic uses only wheelbase) ----
    l_wb = float(vp.get("l_wb", 0.307))

    # ---- Dynamics (kinematic single-track) ----
    dX = v * ca.cos(psi)
    dY = v * ca.sin(psi)
    dpsi = v * ca.tan(delta) / l_wb
    dv = u_a
    ddelta = u_ddelta
    dtheta = vs
    xdot = ca.vertcat(dX, dY, dpsi, dv, ddelta, dtheta)

    # ---- AcadosModel ----
    model = AcadosModel()
    model.name = "mpcc_kinematic"
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = xdot
    model.xdot = ca.SX.sym("xdot", NX)
    model.f_impl_expr = model.xdot - xdot

    # Contour / lag errors
    dx_e = X - x_ref
    dy_e = Y - y_ref
    model.eC_expr = -ca.sin(psi_ref) * dx_e + ca.cos(psi_ref) * dy_e
    model.eL_expr = ca.cos(psi_ref) * dx_e + ca.sin(psi_ref) * dy_e

    # Track half-space (Liniger projection)
    car_proj = -ca.sin(psi_ref) * X + ca.cos(psi_ref) * Y
    inner_proj = -ca.sin(psi_ref) * inner_x + ca.cos(psi_ref) * inner_y
    outer_proj = -ca.sin(psi_ref) * outer_x + ca.cos(psi_ref) * outer_y
    model.h_track_inner_expr = car_proj - inner_proj   # ≥ 0 desired
    model.h_track_outer_expr = outer_proj - car_proj   # ≥ 0 desired

    return model
