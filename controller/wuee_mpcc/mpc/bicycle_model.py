#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser

import types

import numpy as np
# 최신 acados_template 은 casadi 심볼 re-export 안 함 → casadi 에서 직접 import.
# casadi 의 `mod` 는 `fmod` 로 이름이 바뀌었으므로 alias 매핑.
from casadi import MX, Function, cos, fmax, interpolant, sin, vertcat
from casadi import fmod as mod  # noqa: F401  (원본의 `mod(s, pathlength)` 유지용)
from casadi import *  # noqa: F401,F403  (atan2, atan, if_else, fabs 등 사용)


def bicycle_model(s0: list, kapparef: list, d_left: list, d_right: list, cfg_dict: dict):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end

    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    s0 = np.append([s0[: length - 1] - s0[length - 1]], s0)
    kapparef = np.append(kapparef, kapparef[1:length])
    kapparef = np.append([kapparef[: length - 1] - kapparef[length - 1]], kapparef)

    d_left = np.append(d_left, d_left[1:length])
    d_left = np.append([d_left[: length - 1] - d_left[length - 1]], d_left)
    d_right = np.append(d_right, d_right[1:length])
    d_right = np.append([d_right[: length - 1] - d_right[length - 1]], d_right)

    N = cfg_dict["N"]

    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    outer_bound_s = interpolant("outer_bound_s", "bspline", [s0], d_left)
    inner_bound_s = interpolant("inner_bound_s", "bspline", [s0], d_right)

    ## CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    alpha = MX.sym("alpha")
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    omega = MX.sym("omega")
    D = MX.sym("D")
    delta = MX.sym("delta")
    theta = MX.sym("theta")

    x = vertcat(s, n, alpha, vx, vy, omega, D, delta, theta)

    # controls
    derD = MX.sym("derD")
    derDelta = MX.sym("derDelta")
    derTheta = MX.sym("derTheta")
    u = vertcat(derD, derDelta, derTheta)

    next_D = D + derD / N
    next_delta = delta + derDelta / N

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    alphadot = MX.sym("alphadot")
    vxdot = MX.sym("vxdot")
    vydot = MX.sym("vydot")
    omegadot = MX.sym("omegadot")
    Ddot = MX.sym("Ddot")
    deltadot = MX.sym("deltadot")
    thetadot = MX.sym("thetadot")
    xdot = vertcat(sdot, ndot, alphadot, vxdot, vydot, omegadot, Ddot, deltadot, thetadot)

    m = MX.sym("m")
    C1 = MX.sym("C1")
    C2 = MX.sym("C2")
    CSf = MX.sym("CSf")
    CSr = MX.sym("CSr")
    Cr0 = MX.sym("Cr0")
    Cr2 = MX.sym("Cr2")
    Cr3 = MX.sym("Cr3")
    Iz = MX.sym("Iz")
    lr = MX.sym("lr")
    lf = MX.sym("lf")
    Df = MX.sym("Df")
    Cf = MX.sym("Cf")
    Bf = MX.sym("Bf")
    Dr = MX.sym("Dr")
    Cr = MX.sym("Cr")
    Br = MX.sym("Br")
    Imax_c = MX.sym("Imax_c")
    Caccel = MX.sym("Caccel")
    Cdecel = MX.sym("Deccel")
    qc = MX.sym("qc")
    ql = MX.sym("ql")
    gamma = MX.sym("gamma")
    r1 = MX.sym("r1")
    r2 = MX.sym("r2")
    r3 = MX.sym("r3")
    # ### HJ : damping weights (yaw-rate / steer / slip) 추가 — tailing 억제.
    q_omega = MX.sym("q_omega")
    q_delta = MX.sym("q_delta")
    q_vy = MX.sym("q_vy")
    # ### HJ : race_stack 정합 Pacejka 계수 — E 계수 + mu 스케일링 도입.
    # (Bf, Cf, Df, Br, Cr, Dr 은 기존 그대로. Ef/Er/mu_tire 만 추가.)
    Ef = MX.sym("Ef")
    Er = MX.sym("Er")
    mu_tire = MX.sym("mu_tire")

    # algebraic variables
    z = vertcat([])

    # parameters
    p = vertcat(
        m,
        C1,
        C2,
        CSf,
        CSr,
        Cr0,
        Cr2,
        Cr3,
        Iz,
        lr,
        lf,
        Bf,
        Cf,
        Df,
        Br,
        Cr,
        Dr,
        Imax_c,
        Caccel,
        Cdecel,
        qc,
        ql,
        gamma,
        r1,
        r2,
        r3,
        q_omega,
        q_delta,
        q_vy,
        Ef,
        Er,
        mu_tire,
    )

    s_mod = mod(s, pathlength)

    # constraint on forces
    a_lat = next_D * sin(C1 * next_delta)
    a_long = next_D

    n_outer_bound = outer_bound_s(s_mod) + n
    n_inner_bound = inner_bound_s(s_mod) - n

    # Model bounds
    model.n_min = -1e3
    model.n_max = 1e3

    constraint.n_min = cfg_dict["track_savety_margin"]  # width of the track [m]
    constraint.n_max = 1e3  # width of the track [m]
    # state bounds
    model.throttle_min = -5.0
    model.throttle_max = 5.0

    model.delta_min = -0.40  # minimum steering angle [rad]
    model.delta_max = 0.40  # maximum steering angle [rad]

    # input bounds
    # ### HJ : steer-rate 한계 축소 (tailing 방지). cfg override 가능.
    _ddelta_max_abs = float(cfg_dict.get("ddelta_max", 0.5))
    model.ddelta_min = -_ddelta_max_abs
    model.ddelta_max = _ddelta_max_abs
    model.dthrottle_min = -10  # -10.0  # minimum throttle change rate
    model.dthrottle_max = 10  # 10.0  # maximum throttle change rate
    model.dtheta_min = -3.2
    model.dtheta_max = 5

    # nonlinear constraint
    constraint.alat_min = -100  # maximum lateral force [m/s^2]
    constraint.alat_max = 100  # maximum lateral force [m/s^1]

    constraint.along_min = -4  # maximum lateral force [m/s^2]
    constraint.along_max = 4  # maximum lateral force [m/s^2]

    constraint.vx_min = 0
    constraint.vx_max = 30

    # ### HJ : vy 하드 바운드 완화 (±1 → ±3).
    # Pacejka 스케일 상향 + 코너 과도 슬립 가능 → horizon 중 예측 vy 가 ±1 을
    # 넘어 QP infeasible 되는 일 방지. 실제 물리적 vy 는 이보다 작게 나옴.
    constraint.vy_min = -3
    constraint.vy_max = 3

    # NOTE: 원본은 CasADi Function 으로 accel/decel 을 wrap 했지만, 최신 casadi 는
    # Function 의 free-variable 을 거부함 (Imax_c, Cr0, Caccel, Cdecel 이 파라미터
    # MX 라 "free"). Function 없이 인라인 MX 식으로 대체 → 수학적으로 동일.
    accel_expr = (Imax_c - Cr0 * vx) * D / (model.throttle_max * Caccel)
    decel_expr = (-Imax_c - Cr0 * vx) * fabs(D) / (model.throttle_max * Cdecel)

    # dynamics
    # ### HJ : Frenet denominator `1 - n*kappa` 가 |n|>1/|kappa| 일 때 0/음수 →
    # horizon 중 어느 stage 에서든 터지면 solver 터짐. upenn_mpc 처럼 smooth max
    # 로 하한 바닥 ~0.1 을 보장 (0.5·(raw + sqrt(raw²+0.01)) ≥ ≈0.05).
    _sdots_raw = 1.0 - kapparef_s(s) * n
    _sdots_guard = 0.5 * (_sdots_raw + sqrt(_sdots_raw * _sdots_raw + 0.01))
    sdota = (vx * cos(alpha) - vy * sin(alpha)) / _sdots_guard

    Fx = if_else(D >= 0, m * accel_expr, m * decel_expr)
    # Fx = m * next_D

    # ### HJ : 원본은 fmax(vx, 0.1) 로 저속 가드를 걸었으나 fmax 는 미분 불연속 →
    # 새 Pacejka 스케일(~9 N, 이전 1.3 N 대비 ~7 배) 에서 QP Hessian 이 깨져
    # ACADOS_MINSTEP 이 발생. upenn_mpc 와 동일한 smooth 정규화로 교체.
    _eps_vx = 1.5
    vx = sqrt(vx * vx + _eps_vx * _eps_vx)

    # Carron

    if cfg_dict["slip_angle_approximation"]:
        beta = atan2(vy, vx)
        ar = -beta + lr * omega / vx
        af = delta - beta - lf * omega / vx

    else:

        af = -atan2(vy + lf * omega, vx) + next_delta
        ar = -atan2(vy - lr * omega, vx)

    Fr = CSr * ar
    Ff = CSf * af

    if cfg_dict["use_pacejka_tiremodel"]:
        # ### HJ : race_stack convention Pacejka.
        # F_y = mu * N * D * sin(C * atan(B*α - E*(B*α - atan(B*α))))
        # N: static normal load per axle (m·g·l_rearaxle / l_wb).
        # E: curvature 계수 (wueestry 원본은 생략하고 있었음).
        # 이렇게 하면 /home/.../SRX1_pacejka.yaml (B, C, D, E) 값을 그대로 쓸 수 있음.
        g = 9.81
        Nf = m * g * lr / (lr + lf)   # front axle static load
        Nr = m * g * lf / (lr + lf)   # rear axle static load

        def _pacejka_xy(alpha, N, B, C, D, E):
            xp = B * alpha
            phi = xp - E * (xp - atan(xp))
            return mu_tire * N * D * sin(C * atan(phi))

        Fr = _pacejka_xy(ar, Nr, Br, Cr, Dr, Er)
        Ff = _pacejka_xy(af, Nf, Bf, Cf, Df, Ef)

    f_expl = vertcat(
        sdota,
        vx * sin(alpha) + vy * cos(alpha),
        omega,
        1 / m * (Fx - Ff * sin(next_delta) + m * vy * omega),
        1 / m * (Fr + Ff * cos(next_delta) - m * vx * omega),
        1 / Iz * (Ff * lf * cos(next_delta) - Fr * lr),
        derD,
        derDelta,
        derTheta,
    )

    # constraint on forces
    a_lat = next_D * sin(C1 * next_delta)
    a_long = next_D

    n_outer_bound = outer_bound_s(s_mod) + n
    n_inner_bound = inner_bound_s(s_mod) - n

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 0, 0, 0, 0, 0, 0])

    # ### HJ : original wueestry cost 는 lag/contour/progress/input-reg 만 가지고
    # yaw-rate, steer, slip 에 대한 damping 이 없어 좌우 tailing 이 심함.
    # q_omega·ω² + q_delta·δ² + q_vy·vy² 3 항 추가.
    model.cost_expr_ext_cost = (
        ql * (s - theta) ** 2
        + qc * n**2
        - gamma * derTheta
        + r1 * derD**2
        + r2 * derDelta**2
        + r3 * derTheta**2
        + q_omega * omega**2
        + q_delta * delta**2
        + q_vy * vy**2
    )
    # Terminal: damping 만 유지 (progress reward 는 stage 에서 축적되므로 불필요).
    model.cost_expr_ext_cost_e = (
        ql * (s - theta) ** 2
        + qc * n**2
        + q_omega * omega**2
        + q_delta * delta**2
        + q_vy * vy**2
    )

    # define constraints struct
    # a_lat 은 C1 (free param MX) 을 포함 → allow_free=True 로 허용.
    # 이 Function 은 현재 코드베이스에서 사용처 없음 (미래 확장용).
    constraint.alat = Function("a_lat", [x, u], [a_lat], {"allow_free": True})
    constraint.pathlength = pathlength

    # ### HJ : Friction circle constraint — 진짜 MPC way 로 코너 감속 강제.
    # 휴리스틱 v_max(k) 대신 매 stage 에서 tire 총 힘이 μ·N 원 안에 있도록 제약.
    # front: 횡력만 (no drive) / rear: 횡+종 (rear-driven).
    # mu_tire, Nf, Nr 은 Pacejka 블록에서 이미 정의. 수치 안정 위해 safety margin 0.95.
    _fc_margin = float(cfg_dict.get("friction_margin", 0.95))
    if cfg_dict["use_pacejka_tiremodel"]:
        h_fc_front = Ff * Ff - (mu_tire * _fc_margin * Nf) ** 2
        h_fc_rear = Fx * Fx + Fr * Fr - (mu_tire * _fc_margin * Nr) ** 2
    else:
        # Linear tire: friction circle 물리의미가 약해져 제약 비활성 (dummy).
        h_fc_front = MX(0)
        h_fc_rear = MX(0)
    constraint.expr = vertcat(
        a_long, a_lat, n_inner_bound, n_outer_bound, h_fc_front, h_fc_rear
    )

    # f_expl_func = Function(
    #     "f_expl_func", [s, n, alpha, vx, vy, D, omega, delta, theta, derD, derDelta, derTheta, Fx, p], [f_expl]
    # )

    # Define model struct
    params = types.SimpleNamespace()
    params.C1 = C1
    params.C2 = C2
    params.CSf = CSf
    params.CSr = CSr
    params.Cr0 = Cr0
    params.Cr2 = Cr2
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    model.kappa = kapparef_s
    # model.f_expl_func = f_expl_func
    model.outer_bound_s = outer_bound_s
    model.inner_bound_s = inner_bound_s
    return model, constraint, params
