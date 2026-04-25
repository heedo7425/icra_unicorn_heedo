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

import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from mpc.bicycle_model import bicycle_model
from utils.indecies import Input, State


def acados_settings(Ts, N, s0, kapparef, d_left, d_right, cfg_dict: dict):
    # create render arguments
    ocp = AcadosOcp()

    # export model
    model, constraint, params = bicycle_model(s0, kapparef, d_left, d_right, cfg_dict)

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    p = get_parameters(cfg_dict)
    params.p = p
    ocp.parameter_values = p

    # define constraint
    model_ac.con_h_expr = constraint.expr

    # dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    nsbx = 1
    nh = constraint.expr.shape[0]
    nsh = nh
    ns = nsh + nsbx

    # discretization
    ocp.dims.N = N

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.zu = 100 * np.ones((ns,))
    ocp.cost.Zl = 1 * np.ones((ns,))
    ocp.cost.Zu = 1 * np.ones((ns,))

    ocp.constraints.lbu = np.array([model.dthrottle_min, model.ddelta_min, model.dtheta_min])
    ocp.constraints.ubu = np.array([model.dthrottle_max, model.ddelta_max, model.dtheta_max])
    ocp.constraints.idxbu = np.array(
        [Input.D_DUTY_CYCLE, Input.D_STEERING_ANGLE, Input.D_PROGRESS]
    )

    ocp.constraints.lsbx = np.zeros([nsbx])
    ocp.constraints.usbx = np.zeros([nsbx])
    ocp.constraints.idxsbx = np.array(range(nsbx))

    ocp.constraints.lbx = np.array(
        [
            model.n_min,
            constraint.vx_min,
            constraint.vy_min,
            model.throttle_min,
            model.delta_min,
        ]
    )
    ocp.constraints.ubx = np.array(
        [
            model.n_max,
            constraint.vx_max,
            constraint.vy_max,
            model.throttle_max,
            model.delta_max,
        ]
    )
    ocp.constraints.idxbx = np.array(
        [
            State.MIN_DIST_TO_CENTER_LINE_N,
            State.VELOCITY_VX,
            State.VELOCITY_VY,
            State.DUTY_CYCLE_D,
            State.STEERING_ANGLE_DELTA,
        ]
    )

    # ### HJ : constraint.expr 에 friction-circle 2개 (h_fc_front, h_fc_rear)
    # 추가됨 → lh/uh 도 동일 차원으로 확장. 형식: h ≤ 0 (tire force sq ≤ limit sq).
    ocp.constraints.lh = np.array(
        [
            constraint.along_min,
            constraint.alat_min,
            constraint.n_min,
            constraint.n_min,
            -1e12,   # h_fc_front: 하한 없음
            -1e12,   # h_fc_rear:  하한 없음
        ]
    )
    ocp.constraints.uh = np.array(
        [
            constraint.along_max,
            constraint.alat_max,
            constraint.n_max,
            constraint.n_max,
            0.0,     # h_fc_front: F_yf² ≤ (μ·margin·N_f)²
            0.0,     # h_fc_rear:  F_x² + F_yr² ≤ (μ·margin·N_r)²
        ]
    )

    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array(range(nsh))

    # set intial condition
    ocp.constraints.x0 = model.x0

    # ### HJ : 실시간성 + robustness 개선.
    # (1) SQP → SQP_RTI (한 tick 당 1 iter) — upenn_mpc 와 동일. MINSTEP 시
    #     의 100회 재시도 루프 제거. warm-start 로 다음 tick 에서 개선.
    # (2) LM regularization 을 0.01 → 1e-3 에서 1e-2 사이로 튜닝해서 Hessian
    #     정칙성 확보.
    # (3) qp_solver_iter_max 를 명시적으로 크게 (200).
    ocp.solver_options.tf = Ts * N
    ocp.solver_options.Tsim = Ts
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.qp_solver_warm_start = 2
    ocp.solver_options.levenberg_marquardt = 0.01
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.tol = 1e-2
    ocp.solver_options.print_level = 0
    # ocp.solver_options.nlp_solver_tol_comp = 1e-1

    # Route codegen to /tmp (CWD-independent under roslaunch).
    import os as _os
    _codegen_dir = cfg_dict.get("codegen_dir", "/tmp/wuee_mpcc_c_generated")
    _os.makedirs(_codegen_dir, exist_ok=True)
    ocp.code_export_directory = _codegen_dir

    acados_solver = AcadosOcpSolver(
        ocp, json_file=_os.path.join(_codegen_dir, "acados_ocp.json")
    )

    return constraint, model, acados_solver, params


def get_parameters(cfg: dict) -> np.ndarray:
    params = np.array(
        [
            cfg["m"],
            cfg["C1"],
            cfg["C2"],
            cfg["CSf"],
            cfg["CSr"],
            cfg["Cr0"],
            cfg["Cr2"],
            cfg["Cr3"],
            cfg["Iz"],
            cfg["lr"],
            cfg["lf"],
            cfg["Bf"],
            cfg["Cf"],
            cfg["Df"],
            cfg["Br"],
            cfg["Cr"],
            cfg["Dr"],
            cfg["Imax_c"],
            cfg["Caccel"],
            cfg["Cdecel"],
            cfg["qc"],
            cfg["ql"],
            cfg["gamma"],
            cfg["r1"],
            cfg["r2"],
            cfg["r3"],
            # ### HJ : damping weights (Parameter.q_omega/q_delta/q_vy).
            cfg.get("q_omega", 3.0),
            cfg.get("q_delta", 0.3),
            cfg.get("q_vy", 0.3),
            # ### HJ : race_stack 정합 Pacejka (Ef, Er, mu_tire).
            cfg.get("Ef", 0.37),
            cfg.get("Er", 0.0),
            cfg.get("mu_tire", 0.85),
        ]
    ).astype(float)

    return params
