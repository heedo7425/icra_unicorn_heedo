"""
Dynamic single-track MPCC vehicle model (Pacejka tire forces).

Reference:
    - Liniger MPCC (Cartesian formulation, dynamic + Pacejka).
    - f1tenth_simulator/src/std_kinematics.cpp:14-132 — sim Pacejka 정확 매칭
      (slip α 부호, 저속 cutoff, smooth tanh blend).

State (8): [X, Y, psi, vx, vy, omega, delta, theta]
    X, Y    : cartesian pose (map frame)
    psi     : heading
    vx, vy  : body-frame velocities
    omega   : yaw rate
    delta   : steering angle (state, integrated from u_ddelta)
    theta   : virtual progress along reference path

Input (3): [u_ddelta, u_a, vs]
    u_ddelta : steering rate
    u_a      : longitudinal acceleration (= 차의 longitudinal force / m)
    vs       : virtual progress speed (theta_dot = vs)

Params (8): [x_ref, y_ref, psi_ref, kappa_ref, inner_x, inner_y, outer_x, outer_y]
    spline ref + track bounds at theta_pred[k] (NP unchanged from kine).

Pacejka tire (sim-matched):
    α_f = δ - atan2(vy + l_f·ω, vx)        ← sim sign convention
    α_r =   - atan2(vy - l_r·ω, vx)
    F_zf = m·g·l_r/l_wb,  F_zr = m·g·l_f/l_wb     (static; load transfer 옵션)
    F_yf_dyn = μ·D_f·F_zf·sin(C_f·atan(B_f·α_f - E_f·(B_f·α_f - atan(B_f·α_f))))
    F_yr_dyn = (rear 동일 계수)

Sim-matched 저속 gate:
    v_b=3, v_s=1 (sim 의 blend 영역).
    w_std = 0.5·(1 + tanh((vx - v_b)/v_s))           ← 부드러운 dynamic 활성화
    floor = 0.5·(1 + tanh((vx - v_min)/0.2))         ← v_min=1 hard cutoff 근사
    gate  = w_std · floor
    F_yf  = gate · F_yf_dyn  (저속에선 0 → kinematic 분지로)
    F_yr  = gate · F_yr_dyn

저속 (vx < v_b) 영역에선 vy/omega 가 kinematic 값으로 끌림:
    omega_kin = vx · tan(δ) / l_wb
    vy_kin    = l_r · omega_kin
    dvy    = w_std · dvy_dyn + (1-w_std) · (vy_kin - vy) / τ_blend
    domega = w_std · domega_dyn + (1-w_std) · (omega_kin - omega) / τ_blend

Dynamics:
    dX     = vx·cos(psi) - vy·sin(psi)
    dY     = vx·sin(psi) + vy·cos(psi)
    dpsi   = omega
    dvx    = u_a - F_yf·sin(δ)/m + vy·omega
    dvy / domega : 위 blended 식
    ddelta = u_ddelta
    dtheta = vs

Track half-space (Liniger projection) — kine 와 동일.
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


STATE_NAMES = ("X", "Y", "psi", "vx", "vy", "omega", "delta", "theta")
INPUT_NAMES = ("u_ddelta", "u_a", "vs")
PARAM_NAMES = (
    "x_ref", "y_ref", "psi_ref", "kappa_ref",
    "inner_x", "inner_y", "outer_x", "outer_y",
)
NX = len(STATE_NAMES)
NU = len(INPUT_NAMES)
NP = len(PARAM_NAMES)


def build_dynamic_mpcc_model(vp: dict) -> "AcadosModel":
    """Dynamic single-track + Pacejka AcadosModel (sim-matched).

    vp keys (defaults SRX1):
      m, l_f, l_r, l_wb, I_z, mu,
      Bf, Cf, Df, Ef, Br, Cr, Dr, Er
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(f"acados/casadi not available: {_IMPORT_ERROR}")

    # ---- Symbolics ----
    X = ca.SX.sym("X");        Y = ca.SX.sym("Y")
    psi = ca.SX.sym("psi")
    vx = ca.SX.sym("vx");      vy = ca.SX.sym("vy")
    omega = ca.SX.sym("omega")
    delta = ca.SX.sym("delta")
    theta = ca.SX.sym("theta")
    x = ca.vertcat(X, Y, psi, vx, vy, omega, delta, theta)

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

    # ---- Vehicle params (defaults = SRX1) ----
    m = float(vp.get("m", 3.54))
    l_f = float(vp.get("l_f", 0.162))
    l_r = float(vp.get("l_r", 0.145))
    l_wb = float(vp.get("l_wb", l_f + l_r))
    I_z = float(vp.get("I_z", 0.05797))
    mu = float(vp.get("mu", 0.85))
    Bf = float(vp.get("Bf", 4.798)); Cf = float(vp.get("Cf", 2.164))
    Df = float(vp.get("Df", 0.650)); Ef = float(vp.get("Ef", 0.373))
    Br = float(vp.get("Br", 20.0));  Cr = float(vp.get("Cr", 1.500))
    Dr = float(vp.get("Dr", 0.618)); Er = float(vp.get("Er", 0.0))
    g = 9.81

    # Sim blend params (mirror std_kinematics.cpp v_b/v_s)
    v_b = float(vp.get("v_b", 3.0))
    v_s = float(vp.get("v_s", 1.0))
    v_min_floor = float(vp.get("v_pacejka_floor", 1.0))   # sim hard cutoff
    floor_band = float(vp.get("v_pacejka_floor_band", 0.2))
    tau_blend = float(vp.get("tau_blend", 0.05))   # 저속 kinematic 수렴 시간

    # ---- Slip angles (sim sign convention) ----
    # ### HJ : vx=0 일 때 atan2(0,0) NaN 가능성 — eps 정규화. 작게 두면 (0.05) gate
    # 가 어차피 ~0 으로 force 를 막아서 영향 미미.
    eps_vx = 0.05
    vx_reg = ca.sqrt(vx * vx + eps_vx * eps_vx)
    alpha_f = delta - ca.atan2(vy + l_f * omega, vx_reg)
    alpha_r = -ca.atan2(vy - l_r * omega, vx_reg)

    # ---- Static normal force (load transfer 생략 — 안정성 우선) ----
    Nz = m * g
    Nf = Nz * l_r / l_wb
    Nr = Nz * l_f / l_wb

    # ---- Pacejka full formula ----
    def _pacejka(alpha, N, B, C, D, E):
        x_p = B * alpha
        phi = x_p - E * (x_p - ca.atan(x_p))
        return mu * N * D * ca.sin(C * ca.atan(phi))

    Fyf_dyn = _pacejka(alpha_f, Nf, Bf, Cf, Df, Ef)
    Fyr_dyn = _pacejka(alpha_r, Nr, Br, Cr, Dr, Er)

    # ---- Sim-matched 저속 gate (이중 tanh) ----
    # w_std : sim 의 dynamic 활성화 weight, smooth.
    w_std = 0.5 * (1.0 + ca.tanh((vx - v_b) / v_s))
    # floor : sim 의 hard cutoff (vx < 1 → α=0) 근사. transition 폭 0.2.
    floor = 0.5 * (1.0 + ca.tanh((vx - v_min_floor) / floor_band))
    gate = w_std * floor

    Fyf = gate * Fyf_dyn
    Fyr = gate * Fyr_dyn

    # ---- Kinematic blend at low vx ----
    omega_kin = vx * ca.tan(delta) / l_wb
    vy_kin = l_r * omega_kin

    # ---- Dynamic ODEs ----
    dvx_dyn = u_a - Fyf * ca.sin(delta) / m + vy * omega
    dvy_dyn = (Fyf * ca.cos(delta) + Fyr) / m - vx * omega
    domega_dyn = (l_f * Fyf * ca.cos(delta) - l_r * Fyr) / I_z

    # ---- Kinematic-anchored ODEs (저속 fallback) ----
    dvx_kin = u_a
    dvy_kin = (vy_kin - vy) / tau_blend
    domega_kin = (omega_kin - omega) / tau_blend

    # ---- Blended dynamics ----
    dvx = w_std * dvx_dyn + (1.0 - w_std) * dvx_kin
    dvy = w_std * dvy_dyn + (1.0 - w_std) * dvy_kin
    domega = w_std * domega_dyn + (1.0 - w_std) * domega_kin

    dX = vx * ca.cos(psi) - vy * ca.sin(psi)
    dY = vx * ca.sin(psi) + vy * ca.cos(psi)
    dpsi = omega
    ddelta = u_ddelta
    dtheta = vs

    xdot = ca.vertcat(dX, dY, dpsi, dvx, dvy, domega, ddelta, dtheta)

    # ---- AcadosModel ----
    model = AcadosModel()
    model.name = "mpcc_dynamic"
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = xdot
    model.xdot = ca.SX.sym("xdot", NX)
    model.f_impl_expr = model.xdot - xdot

    # Expose for OCP cost / friction-ellipse / metrics
    model.fyf_expr = Fyf
    model.fyr_expr = Fyr
    model.nf_expr = Nf
    model.nr_expr = Nr
    model.alphaf_expr = alpha_f
    model.alphar_expr = alpha_r

    # Contour / lag errors (kine 와 동일)
    dx_e = X - x_ref
    dy_e = Y - y_ref
    model.eC_expr = -ca.sin(psi_ref) * dx_e + ca.cos(psi_ref) * dy_e
    model.eL_expr = ca.cos(psi_ref) * dx_e + ca.sin(psi_ref) * dy_e

    # Track half-space (Liniger projection — kine 와 동일)
    car_proj = -ca.sin(psi_ref) * X + ca.cos(psi_ref) * Y
    inner_proj = -ca.sin(psi_ref) * inner_x + ca.cos(psi_ref) * inner_y
    outer_proj = -ca.sin(psi_ref) * outer_x + ca.cos(psi_ref) * outer_y
    model.h_track_inner_expr = car_proj - inner_proj
    model.h_track_outer_expr = outer_proj - car_proj

    return model


# Backward-compat alias (mpcc_ocp.py 에서 같은 이름으로 import)
build_kinematic_mpcc_model = build_dynamic_mpcc_model
