"""
Frenet-frame **KINEMATIC** bicycle model for acados OCP (upenn_mpcc_rm variant).

upenn_mpcc 원본의 dynamic bicycle + Pacejka 가 sim (linear tire) 와 mismatch 가
커서 friction_circle 이 제대로 작동 안 하던 문제 (sim 한바퀴 못 돔). 이 변형은:
  - tire model 제거 (Pacejka X)
  - friction circle 제거 (slip force 정의 X)
  - vy/omega 는 state 슬롯 유지하지만 dynamics 0 → kinematic omega 만 사용
  - 노드/OCP 인터페이스 호환을 위해 state shape 7D 유지

State  (7):  [s, n, dpsi, vx, vy, omega, delta]
             vy 와 omega 는 더 이상 dynamics state 가 아님 (그냥 0 유지).
Input  (2):  [u_ddelta, u_ax]
Params (6):  [kappa, theta, kappa_z (unused), mu (unused), vx_ref, n_ref]

Kinematic dynamics:
    omega_kin = vx · tan(δ) / l_wb
    ds/dt    = vx · cos(dpsi) / (1 - n·kappa)
    dn/dt    = vx · sin(dpsi)
    ddpsi/dt = omega_kin - kappa · ds/dt
    dvx/dt   = u_ax
    dvy/dt   = 0
    domega/dt = 0
    ddelta/dt = u_ddelta

노드는 매 tick 의 initial state 에서 vy=0, omega=0 으로 강제해 horizon 동안
kinematic 가정 유지. tire saturation 이 없으므로 v_max, max_steer 는 hard
bound 로만 작동 — friction_circle 제약 없이 코너 감속은 cost 의
qprog·(v_target - vx)² 와 path tracking 의 균형이 유도해야 함.
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


STATE_NAMES = ("s", "n", "dpsi", "vx", "vy", "omega", "delta")
INPUT_NAMES = ("u_ddelta", "u_ax")
PARAM_NAMES = ("kappa", "theta", "kappa_z", "mu", "vx_ref", "n_ref")

NX = len(STATE_NAMES)
NU = len(INPUT_NAMES)
NP = len(PARAM_NAMES)


def build_dynamic_bicycle_model(vp: dict) -> "AcadosModel":
    """Build kinematic bicycle acados model (name kept for upenn_mpcc API compat).

    Parameters
    ----------
    vp : dict
        Vehicle params. Only `l_wb` (wheelbase) is used; tire/inertia keys are
        accepted but ignored (kinematic model has no Pacejka or moment of inertia).
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(
            f"acados/casadi not available: {_IMPORT_ERROR}. "
            "Install acados or switch framework."
        )

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

    l_wb = float(vp["l_wb"])
    g = 9.81

    # Kinematic yaw rate from steering geometry.
    omega_kin = vx * ca.tan(delta) / l_wb

    # Frenet kinematics (dpsi 는 차 헤딩 - 트랙 탄젠트 각).
    s_dot_scale_raw = 1.0 - n * kappa
    s_dot_scale = 0.5 * (s_dot_scale_raw + ca.sqrt(s_dot_scale_raw ** 2 + 0.01))
    ds_dot = vx * ca.cos(dpsi) / s_dot_scale
    dn_dot = vx * ca.sin(dpsi)
    ddpsi_dot = omega_kin - kappa * ds_dot

    # Longitudinal: 단순 적분기. grade θ 는 중력 성분만.
    dvx_dot = u_ax - g * ca.sin(theta)

    # vy, omega 는 dynamics 에서 제거 — 0 유지.
    dvy_dot = ca.SX(0.0)
    domega_dot = ca.SX(0.0)

    ddelta_dot = u_ddelta

    xdot = ca.vertcat(
        ds_dot, dn_dot, ddpsi_dot, dvx_dot, dvy_dot, domega_dot, ddelta_dot
    )

    model = AcadosModel()
    model.name = "icra2026_kinematic_bicycle"
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = xdot
    model.xdot = ca.SX.sym("xdot", NX)
    model.f_impl_expr = model.xdot - xdot

    # Kinematic 모델은 tire force / friction circle 정의 없음 — OCP 측에서
    # 이 attr 의 부재를 감지해 친구 friction 제약을 skip 해야 함.
    model.fyf_expr = None
    model.fyr_expr = None
    model.nf_expr = None
    model.nr_expr = None

    return model


def default_vehicle_params() -> dict:
    """Fallback vehicle params. Kinematic 만 쓰지만 dynamic 호환 키 유지."""
    return {
        "m": 3.54,
        "l_f": 0.162,
        "l_r": 0.145,
        "l_wb": 0.307,
        "I_z": 0.05797,
        "h_cg": 0.014,
        "tau_steer": 0.158,
        "Bf": 4.80, "Cf": 2.16, "Df": 0.65, "Ef": 0.37,
        "Br": 20.0, "Cr": 1.50, "Dr": 0.62, "Er": 0.0,
    }


def load_vehicle_params_from_ros(rospy) -> dict:
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
