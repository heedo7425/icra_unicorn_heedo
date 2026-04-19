"""
Frenet-frame dynamic bicycle + Pacejka tire model for acados OCP.

State  (7):  [s, n, dpsi, vx, vy, omega, delta]
Input  (2):  [u_ddelta, u_ax]
Params (6):  [kappa, theta, kappa_z, mu, vx_ref, n_ref]

Conventions:
- Frenet: s = arc length along ref, n = lateral offset, dpsi = heading err.
- Body frame: vx forward, vy left, omega yaw rate.
- kappa: lateral curvature of ref (1/m). theta: grade (rad). kappa_z: vertical
  curvature (1/m) for normal-force term (0 in 2D).

Pacejka lateral force (front/rear):
    F_y = mu * N * D * sin(C * atan(B*a - E*(B*a - atan(B*a))))
with slip angle
    a_f = atan2((vy + l_f*omega), vx_reg) - delta
    a_r = atan2((vy - l_r*omega), vx_reg)
and normal force
    N_{f,r} = m*g*cos(theta)*l_{r,f}/l_wb - m*vx^2*kappa_z*...  (lumped constant for now)

Smoothing: vx_reg = sqrt(vx^2 + eps) to avoid singularity at stand-still.
"""

from __future__ import annotations

import numpy as np

try:
    import casadi as ca
    from acados_template import AcadosModel
except ImportError as e:  # acados not installed — import-time guard
    ca = None
    AcadosModel = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# State / input / param indices — single source of truth.
STATE_NAMES = ("s", "n", "dpsi", "vx", "vy", "omega", "delta")
INPUT_NAMES = ("u_ddelta", "u_ax")
PARAM_NAMES = ("kappa", "theta", "kappa_z", "mu", "vx_ref", "n_ref")

NX = len(STATE_NAMES)
NU = len(INPUT_NAMES)
NP = len(PARAM_NAMES)


def build_dynamic_bicycle_model(vp: dict) -> "AcadosModel":
    """Return acados AcadosModel for dynamic bicycle + Pacejka in Frenet frame.

    Parameters
    ----------
    vp : dict
        Vehicle params (from /shared_vehicle/vehicle_srx1.yaml):
            m, l_f, l_r, l_wb, I_z, h_cg, tau_steer
        Tire front/rear:
            Bf, Cf, Df, Ef, Br, Cr, Dr, Er
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(
            f"acados/casadi not available: {_IMPORT_ERROR}. "
            "Install acados or switch framework."
        )

    # ---- Symbolics ----
    s = ca.SX.sym("s")
    n = ca.SX.sym("n")
    dpsi = ca.SX.sym("dpsi")
    vx = ca.SX.sym("vx")
    vy = ca.SX.sym("vy")
    omega = ca.SX.sym("omega")
    delta = ca.SX.sym("delta")
    x = ca.vertcat(s, n, dpsi, vx, vy, omega, delta)

    u_ddelta = ca.SX.sym("u_ddelta")
    u_ax = ca.SX.sym("u_ax")
    u = ca.vertcat(u_ddelta, u_ax)

    kappa = ca.SX.sym("kappa")
    theta = ca.SX.sym("theta")
    kappa_z = ca.SX.sym("kappa_z")
    mu = ca.SX.sym("mu")
    vx_ref = ca.SX.sym("vx_ref")
    n_ref = ca.SX.sym("n_ref")
    p = ca.vertcat(kappa, theta, kappa_z, mu, vx_ref, n_ref)

    # ---- Params ----
    m = float(vp["m"])
    l_f = float(vp["l_f"])
    l_r = float(vp["l_r"])
    l_wb = float(vp["l_wb"])
    I_z = float(vp["I_z"])
    tau_steer = float(vp.get("tau_steer", 0.0))
    Bf, Cf, Df, Ef = float(vp["Bf"]), float(vp["Cf"]), float(vp["Df"]), float(vp["Ef"])
    Br, Cr, Dr, Er = float(vp["Br"]), float(vp["Cr"]), float(vp["Dr"]), float(vp["Er"])
    g = 9.81

    # ---- Normal force (grade + vertical curvature) ----
    # Static weight distribution: N_f_static = m*g*cos(theta) * l_r / l_wb.
    # Vertical curvature reduces N at crests (kappa_z > 0) and increases at sags.
    Nz_static = m * g * ca.cos(theta)
    Nz_vert = m * vx * vx * kappa_z
    Nz = Nz_static - Nz_vert
    Nf = Nz * l_r / l_wb
    Nr = Nz * l_f / l_wb

    # ---- Slip angles (with low-speed regularization) ----
    # Pacejka slip at vx≈0 is numerically pathological. vx_reg floors the denominator
    # so atan2 remains well-conditioned; HPIPM solver needs this for convergence.
    eps_vx = 1.5  # [m/s] — effective "pseudo-speed" minimum inside tire model
    vx_reg = ca.sqrt(vx * vx + eps_vx * eps_vx)
    # Standard single-track slip-angle convention (Liniger 2015, Rajamani):
    #   α_f = δ − atan((v_y + l_f·ω)/v_x)
    #   α_r =   − atan((v_y − l_r·ω)/v_x)
    # Positive α produces positive lateral force via Pacejka magic-formula,
    # which with dvy/dt = (F_yf·cosδ + F_yr)/m − v_x·ω makes positive δ (left
    # steer) accelerate the body in +y (left) — the physically correct sign.
    alpha_f = delta - ca.atan2(vy + l_f * omega, vx_reg)
    alpha_r = -ca.atan2(vy - l_r * omega, vx_reg)

    # ---- Pacejka magic formula (lateral force) ----
    def _pacejka(alpha, N, B, C, D, E):
        x_p = B * alpha
        phi = x_p - E * (x_p - ca.atan(x_p))
        return mu * N * D * ca.sin(C * ca.atan(phi))

    Fyf = _pacejka(alpha_f, Nf, Bf, Cf, Df, Ef)
    Fyr = _pacejka(alpha_r, Nr, Br, Cr, Dr, Er)

    # ---- Dynamics ----
    # Frenet kinematics.
    # s_dot_scale = 1 - n·kappa becomes zero/negative when |n| > 1/|kappa|.
    # Clamp to small positive value to avoid NaN during horizon integration if
    # the QP lets predicted n drift too far.
    s_dot_scale_raw = 1.0 - n * kappa
    s_dot_scale = 0.5 * (s_dot_scale_raw + ca.sqrt(s_dot_scale_raw**2 + 0.01))  # ≥ ~0.05
    ds_dot = (vx * ca.cos(dpsi) - vy * ca.sin(dpsi)) / s_dot_scale
    dn_dot = vx * ca.sin(dpsi) + vy * ca.cos(dpsi)
    ddpsi_dot = omega - kappa * ds_dot

    # Body-frame dynamics (with grade)
    dvx_dot = u_ax - g * ca.sin(theta) - Fyf * ca.sin(delta) / m + vy * omega
    dvy_dot = (Fyf * ca.cos(delta) + Fyr) / m - vx * omega
    domega_dot = (l_f * Fyf * ca.cos(delta) - l_r * Fyr) / I_z

    # Steering first-order lag: dδ/dt = u_ddelta (direct rate command).
    # tau_steer handled by rate bounds (|u_ddelta| <= max_steer_rate).
    ddelta_dot = u_ddelta

    xdot = ca.vertcat(ds_dot, dn_dot, ddpsi_dot, dvx_dot, dvy_dot, domega_dot, ddelta_dot)

    # ---- AcadosModel ----
    model = AcadosModel()
    model.name = "icra2026_dyn_bicycle_pacejka"
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = xdot
    model.xdot = ca.SX.sym("xdot", NX)
    model.f_impl_expr = model.xdot - xdot

    # Expose slip/forces as expressions for cost/constraint hooks.
    model.con_expr_slack = ca.vertcat(alpha_f, alpha_r, Fyf, Fyr)
    # Expose individual tire forces + normal loads for friction-circle constraint
    # built in mpcc_ocp.py (see build_tracking_ocp).
    model.fyf_expr = Fyf
    model.fyr_expr = Fyr
    model.nf_expr = Nf
    model.nr_expr = Nr

    return model


def default_vehicle_params() -> dict:
    """Fallback vehicle_srx1.yaml params (used when rosparam not loaded).

    These must mirror /shared_vehicle/vehicle_srx1.yaml — sync on change.
    """
    return {
        "m": 3.54,
        "l_f": 0.162,
        "l_r": 0.145,
        "l_wb": 0.307,
        "I_z": 0.05797,
        "h_cg": 0.014,
        "tau_steer": 0.158,
        "Bf": 4.80,  "Cf": 2.16, "Df": 0.65, "Ef": 0.37,
        "Br": 20.0,  "Cr": 1.50, "Dr": 0.62, "Er": 0.0,
    }


def load_vehicle_params_from_ros(rospy) -> dict:
    """Read /vehicle/* and /tire_{front,rear}/* from rosparam into flat dict."""
    gp = rospy.get_param
    return {
        "m": gp("/vehicle/m", 3.54),
        "l_f": gp("/vehicle/l_f", 0.162),
        "l_r": gp("/vehicle/l_r", 0.145),
        "l_wb": gp("/vehicle/l_wb", 0.307),
        "I_z": gp("/vehicle/I_z", 0.05797),
        "h_cg": gp("/vehicle/h_cg", 0.014),
        "tau_steer": gp("/vehicle/tau_steer", 0.158),
        "Bf": gp("/tire_front/B", 4.80),
        "Cf": gp("/tire_front/C", 2.16),
        "Df": gp("/tire_front/D", 0.65),
        "Ef": gp("/tire_front/E", 0.37),
        "Br": gp("/tire_rear/B", 20.0),
        "Cr": gp("/tire_rear/C", 1.50),
        "Dr": gp("/tire_rear/D", 0.62),
        "Er": gp("/tire_rear/E", 0.0),
    }


if __name__ == "__main__":
    vp = default_vehicle_params()
    model = build_dynamic_bicycle_model(vp)
    print(f"Model name: {model.name}")
    print(f"nx={model.x.shape[0]}, nu={model.u.shape[0]}, np={model.p.shape[0]}")
