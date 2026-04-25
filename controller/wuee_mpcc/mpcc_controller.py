#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Port of wueestry/f110-mpcc/src/mpcc_controller.py into race_stack.

Topics/contract kept identical to upstream. Small adaptations:
  - /tf_odom 은 race_stack 에 없음 → /car_state/odom 의 twist.linear.y 사용.
  - acados codegen/ocp.json 을 /tmp/wuee_mpcc_c_generated 로 이동 (CWD 의존 제거).
  - sys.path 에 스크립트 디렉토리 추가 → `from mpc.X`, `from utils.X` import OK.
"""

import os
import sys
import time
from datetime import datetime
from typing import Tuple

# Make `mpc` and `utils` subpackages importable when launched by roslaunch
# (which puts CWD somewhere else). This must happen BEFORE any relative import.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import rospy
import yaml
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import ObstacleArray, WpntArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from scipy.integrate import solve_ivp
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

from mpc.acados_settings import acados_settings
from plotting_fnc import plot_res
from utils.frenet_cartesian_converter import convert_frenet_to_cartesian
from utils.indecies import Input, Parameter, State
from utils.splinify import SplineTrack


class MPC:
    def __init__(self, conf_file: str) -> None:
        rospy.loginfo(f"[wuee_mpcc solver] loading config: {conf_file}")

        self.conf_file = conf_file
        self.vel_x = 0
        self.vel_y = 0
        self.omega = 0

        # steering state — cmd 값이 아니라 실제 actuator 상태 (sim 은 GT 토픽 제공).
        # 이전 코드는 _input_cb 에서 cmd 를 읽어 "actuator 가 cmd 대로 됐다" 고 open-loop
        # 가정 → saturate 시 MPC 내부 δ 와 현실이 수백 mrad 발산 → plan 이 물리 불가능
        # steer 요구 → tailing. 여기서는 /car_state/steering_angle_gt (std_msgs/Float32)
        # 를 구독해 실측 δ 를 solver 의 stage-0 state 로 주입.
        self.steering_angle = 0.0       # = 실측 δ (gt), 최초 수신 전엔 0.
        self._steer_meas_ok = False      # 한 번이라도 GT 메시지 받았는지.

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._odom_frenet_cb)
        rospy.Subscriber("/car_state/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, self._input_cb
        )
        from std_msgs.msg import Float32 as _F32
        rospy.Subscriber("/car_state/steering_angle_gt", _F32, self._steer_gt_cb)
        rospy.Subscriber("/vesc/sensors/imu/raw", Imu, self._imu_cb)
        rospy.Subscriber("/obstacles", ObstacleArray, self._obstacle_cb)
        # race_stack 엔 /tf_odom 이 없으므로 vy 를 /car_state/odom 에서 직접 사용.
        # 아래 _odom_cb 에서 self.vel_y 까지 같이 세팅하므로 별도 subscriber 불필요.

        self.pred_pos_pub = rospy.Publisher(
            "/mpc_controller/predicted_position", MarkerArray, queue_size=10
        )
        self.target_pos_pub = rospy.Publisher(
            "/mpc_controller/target_position", MarkerArray, queue_size=10
        )
        self.drive_next_pub = rospy.Publisher(
            "/mpc_controller/next_input", AckermannDriveStamped, queue_size=10
        )
        self.d_left_pub = rospy.Publisher("/mpc_controller/d_left", MarkerArray, queue_size=10)
        self.d_mid_pub = rospy.Publisher("/mpc_controller/d_mid", MarkerArray, queue_size=10)
        self.d_right_pub = rospy.Publisher("/mpc_controller/d_right", MarkerArray, queue_size=10)
        self.d_left_adj_pub = rospy.Publisher(
            "/mpc_controller/d_left_adj", MarkerArray, queue_size=10
        )
        self.d_right_adj_pub = rospy.Publisher(
            "/mpc_controller/d_right_adj", MarkerArray, queue_size=10
        )
        self.pos_pub = rospy.Publisher("/mpc_controller/current_position", Marker, queue_size=5)
        self.pos_n_pub = rospy.Publisher("/mpc_controller/current_pos_n", Odometry, queue_size=10)

    def initialize(self) -> None:
        try:
            shortest_path = rospy.wait_for_message("/global_waypoints", WpntArray, 20.0)
        except:
            raise TimeoutError("No waypoints received in the appropriate amount of time.")

        # ### HJ : pose/odom 첫 샘플이 들어오기 전에 control_loop 가 돌면
        # _project_pose_to_spline() 이 self.pos_x AttributeError 로 터짐.
        # wait_for_message 는 임시 구독이라 persistent callback 실행을 보장하지
        # 않으므로, 반환된 메시지를 직접 콜백에 주입해 self.* 를 확실히 채운다.
        try:
            pose_msg = rospy.wait_for_message("/car_state/pose", PoseStamped, 10.0)
            odom_msg = rospy.wait_for_message("/car_state/odom", Odometry, 10.0)
        except rospy.ROSException:
            raise TimeoutError("No /car_state/pose or /car_state/odom received.")
        self._pose_cb(pose_msg)
        self._odom_cb(odom_msg)

        # Used for wrapping
        s_coords = [x.s_m for x in shortest_path.wpnts]

        d_left, coords_path, d_right = self._transform_waypoints_to_coords(
            shortest_path.wpnts
        )  # on f track trajectory is 81.803 m long.

        self.spline = SplineTrack(coords_direct=coords_path)

        with open(self.conf_file, "r") as file:
            cfg = yaml.safe_load(file)
            for key in cfg.keys():
                if type(cfg[key]) is list:
                    cfg[key] = [float(i) for i in cfg[key]]

        self.Tf = cfg["Tf"]
        self.N = cfg["N"]
        self.T = cfg["T"]
        self.sref_N = cfg["sref_N"]
        self.s_offset = cfg["s_offset"]
        self.track_savety_margin = cfg["track_savety_margin"]
        self.slip_angle_approx = cfg["slip_angle_approximation"]
        self.use_pacejka = cfg["use_pacejka_tiremodel"]
        t_delay = cfg["t_delay"]
        t_MPC = 1 / cfg["MPC_freq"]

        # time delay propagation
        self.t_delay = t_delay + t_MPC
        self.Ts = t_MPC

        self.nr_laps = 0

        kapparef = [x.kappa_radpm for x in shortest_path.wpnts]
        s0 = self.spline.params

        self.constraint, self.model, self.acados_solver, self.model_params = acados_settings(
            self.Ts, self.N, s0, kapparef, d_left, d_right, cfg
        )

        self.obstacles = None

        self.kappa = self.model.kappa

    def _odom_cb(self, data: Odometry) -> None:
        self.vel_x = data.twist.twist.linear.x
        self.vel_y = data.twist.twist.linear.y  # /tf_odom 대체 (race_stack 호환).
        self.omega = data.twist.twist.angular.z

    def _odom_frenet_cb(self, data: Odometry) -> None:
        # NOTE: /car_state/odom_frenet 의 (s, n) 은 frenet_odom_republisher 의 spline
        # (= /global_waypoints 의 raceline 기준) 이다. mpcc 는 자기 spline
        # (centerline or raceline, launch arg 에 따라 다름) 를 쓰므로 직접 사용 시
        # 프레임 불일치로 1m+ offset 발생. 실제 (s, n) 은 _project_pose_to_spline()
        # 에서 mpcc 자기 spline 으로 재계산. 이 콜백은 호환 유지용으로만 남겨둠.
        pass

    def _project_pose_to_spline(self) -> None:
        """Project actual car pose onto mpcc 자기 spline → self.pos_s, self.pos_n 세팅."""
        coord = np.array([self.pos_x, self.pos_y])
        theta_est = getattr(self, "_prev_s_proj", None)
        # ### HJ : find_theta 는 warm-start 가 랩 wrap 경계에서 멀어지면 numeric
        # issue 를 낼 수 있어서 slow fallback 이 필요. 단 KeyboardInterrupt /
        # rospy 종료 같은 컨트롤 흐름까지 삼키지 않도록 수치 예외만 잡는다.
        try:
            if theta_est is None:
                s = float(self.spline.find_theta_slow(coord))
            else:
                s = float(self.spline.find_theta(coord, theta_est))
        except (
            ValueError,
            ArithmeticError,
            RuntimeError,
            IndexError,
            TypeError,
            np.linalg.LinAlgError,
        ) as e:
            rospy.logwarn_throttle(
                2.0, f"[wuee_mpcc] find_theta fallback to slow: {type(e).__name__}: {e}"
            )
            s = float(self.spline.find_theta_slow(coord))
        self._prev_s_proj = s

        # n = signed lateral distance. convert_frenet_to_cartesian 의 실제 연산
        # `deriv @ R(+π/2)` 은 `(dx,dy) @ [[0,-1],[1,0]] = (dy,-dx)` 로 row-vector
        # multiplication 결과가 **RIGHT** normal 임 (주석은 LEFT 라고 써 있지만
        # 연산 결과는 반대). 따라서 +n = RIGHT 가 되어야 cur_pos/pred 마커가
        # 올바른 쪽에 렌더됨. 코너 정렬 오차의 원인이었음.
        base = np.asarray(self.spline.get_coordinate(s)).reshape(2)
        deriv = np.asarray(self.spline.get_derivative(s)).reshape(2)
        t = deriv / (np.linalg.norm(deriv) + 1e-9)
        normal_right = np.array([t[1], -t[0]])
        self.pos_s = s
        self.pos_n = float(np.dot(coord - base, normal_right))

    def _pose_cb(self, data: PoseStamped) -> None:
        self.pos_x = data.pose.position.x
        self.pos_y = data.pose.position.y

        self.theta = euler_from_quaternion(
            [
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
                data.pose.orientation.w,
            ]
        )[2]

    def _input_cb(self, data: AckermannDriveStamped) -> None:
        # cmd 를 steering_angle 에 덮지 않는다. GT 토픽이 있으면 그걸 신뢰하고,
        # 없을 때만 fallback 으로 cmd 를 사용.
        if not self._steer_meas_ok:
            self.steering_angle = data.drive.steering_angle

    def _steer_gt_cb(self, data) -> None:
        self.steering_angle = float(data.data)
        self._steer_meas_ok = True

    def _imu_cb(self, data: Imu) -> None:
        self.acceleration = data.linear_acceleration.x  # Checked it with plotting. Should be fine.

    def _obstacle_cb(self, data: ObstacleArray) -> None:
        self.obstacles = data.obstacles

    def _transform_waypoints_to_coords(
        self, data: WpntArray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        waypoints = np.zeros((len(data), 2))
        d_left = np.zeros(len(data))
        d_right = np.zeros(len(data))
        boundaries = np.zeros((len(data), 2))
        for idx, wpnt in enumerate(data):
            waypoints[idx] = [wpnt.x_m, wpnt.y_m]
            d_left[idx] = wpnt.d_right  # Fix for boundaries
            d_right[idx] = wpnt.d_left  # Fix for boundaries
        res_coords = np.array([boundaries[:-1], waypoints[:-1], boundaries[:-1]])
        return d_left, res_coords, d_right

    def _dynamics_of_car(self, t, x0) -> list:
        """
        Used for forward propagation. This function takes the dynamics from the acados model.
        """
        s = x0[State.POS_ON_CENTER_LINE_S]
        n = x0[State.MIN_DIST_TO_CENTER_LINE_N]
        alpha = x0[State.ORIENTATION_ALPHA]
        vx = max(0.1, x0[State.VELOCITY_VX])
        vy = x0[State.VELOCITY_VY]
        omega = x0[State.YAW_RATE_OMEGA]
        D = x0[State.DUTY_CYCLE_D]
        delta = x0[State.STEERING_ANGLE_DELTA]
        theta = x0[State.PROGRESS_THETA]

        derD = x0[len(State) + Input.D_DUTY_CYCLE]
        derDelta = x0[len(State) + Input.D_STEERING_ANGLE]
        derTheta = x0[len(State) + Input.D_PROGRESS]

        m = self.model_params.p[Parameter.m]
        Imax_c = self.model_params.p[Parameter.Imax_c]
        Cr0 = self.model_params.p[Parameter.Cr0]
        Caccel = self.model_params.p[Parameter.Caccel]
        Cdecel = self.model_params.p[Parameter.Cdecel]
        lr = self.model_params.p[Parameter.lr]
        lf = self.model_params.p[Parameter.lf]
        CSr = self.model_params.p[Parameter.CSr]
        CSf = self.model_params.p[Parameter.CSf]
        Dr = self.model_params.p[Parameter.Dr]
        Df = self.model_params.p[Parameter.Df]
        Cr = self.model_params.p[Parameter.Cr]
        Cf = self.model_params.p[Parameter.Cf]
        Br = self.model_params.p[Parameter.Br]
        Bf = self.model_params.p[Parameter.Bf]
        Iz = self.model_params.p[Parameter.Iz]

        def accel(vx: float, D: float) -> float:
            return m * (Imax_c - Cr0 * vx) * D / (self.model.throttle_max * Caccel)

        def decel(vx: float, D: float) -> float:
            return m * (-Imax_c - Cr0 * vx) * abs(D) / (self.model.throttle_max * Cdecel)

        Fx = accel(vx, D) if D >= 0 else decel(vx, D)

        if self.slip_angle_approx:
            beta = np.arctan2(vy, vx)
            ar = -beta + lr * omega / vx
            af = delta - beta - lf * omega / vx
        else:
            af = -np.arctan2(vy + lf * omega, vx) + delta
            ar = -np.arctan2(vy - lr * omega, vx)

        Fr = CSr * ar
        Ff = CSf * af

        if self.use_pacejka:
            Fr = Dr * np.sin(Cr * np.arctan(Br * ar))
            Ff = Df * np.sin(Cf * np.arctan(Bf * af))

        xdot = [
            (vx * np.cos(alpha) - vy * np.sin(alpha)) / (1 - float(self.model.kappa(s)) * n),
            vx * np.sin(alpha) + vy * np.cos(alpha),
            omega,
            1 / m * (Fx - Ff * np.sin(delta) + m * vy * omega),
            1 / m * (Fr + Ff * np.cos(delta) - m * vx * omega),
            1 / Iz * (Ff * lf * np.cos(delta) - Fr * lr),
            derD,
            derDelta,
            derTheta,
            derD,
            derDelta,
            derTheta,
        ]

        return xdot

    def propagate_time_delay(self, states: np.array, inputs: np.array) -> np.array:

        # Initial condition on the ODE
        x0 = np.concatenate((states, inputs), axis=0)

        solution = solve_ivp(
            self._dynamics_of_car,
            t_span=[0, self.t_delay],
            y0=x0,
            method="RK45",
            atol=1e-8,
            rtol=1e-8,
        )

        solution = [x[-1] for x in solution.y]

        # Constraint on max. steering angle
        if abs(solution[State.STEERING_ANGLE_DELTA]) > self.model.delta_max:
            solution[State.STEERING_ANGLE_DELTA] = (
                np.sign(solution[State.STEERING_ANGLE_DELTA]) * self.model.delta_max
            )

        # Constraint on max. thrust
        if abs(solution[State.DUTY_CYCLE_D]) > self.model.throttle_max:
            solution[State.DUTY_CYCLE_D] = (
                np.sign(solution[State.DUTY_CYCLE_D]) * self.model.throttle_max
            )

        # ### HJ : stage-0 box constraints 내로 clip. propagated_x 는 그대로
        # lbx_0=ubx_0 (equality) 로 acados 에 주입되므로 벗어나면 QP infeasible.
        # 특히 저속에서 RK45 가 vx 를 음수로 튀게 만들거나, GT δ 측정치가 δ_max
        # 를 살짝 넘는 경우가 실제로 발생함.
        solution[State.VELOCITY_VX] = float(
            np.clip(solution[State.VELOCITY_VX], self.constraint.vx_min, self.constraint.vx_max)
        )
        solution[State.VELOCITY_VY] = float(
            np.clip(solution[State.VELOCITY_VY], self.constraint.vy_min, self.constraint.vy_max)
        )
        solution[State.MIN_DIST_TO_CENTER_LINE_N] = float(
            np.clip(
                solution[State.MIN_DIST_TO_CENTER_LINE_N], self.model.n_min, self.model.n_max
            )
        )

        # Only get the state as solution of where the car will be in t_delay seconds
        return np.array(solution)[: -len(Input)]

    def control_loop(self) -> None:
        self.initialize()
        rate = rospy.Rate(1 / self.Ts)

        x0 = self.get_initial_position(init=True)
        propagated_x = self.propagate_time_delay(x0, np.zeros(len(Input)))

        self.pred_traj = np.array([self.s_offset + x0[State.POS_ON_CENTER_LINE_S] + self.sref_N * j / self.N for j in range(self.N)])

        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)

        self.lap_times = [time.perf_counter()]
        self.nr_of_failures = 0
        self.qp_iterations = []

        # ### HJ : plot 용 기록 버퍼. iter_loop==PLOT_ITER 에서 한 번 plot 저장한
        # 뒤로는 기록을 **중단** (원본은 계속 기록해서 40Hz × 1e5 ≈ 42분 후 IndexError).
        PLOT_ITER = 5000
        BUF_LEN = PLOT_ITER + 1
        simX = np.zeros((BUF_LEN, self.model.x.size()[0]))
        simU = np.zeros((BUF_LEN, self.model.u.size()[0]))
        realX = np.zeros((BUF_LEN, self.model.x.size()[0]))
        propX = np.zeros((BUF_LEN, self.model.x.size()[0]))
        iter_loop = 1
        simX[0, :] = x0
        realX[0, :] = x0
        propX[0, :] = propagated_x

        tcomp_sum = 0
        tcomp_max = 0

        while not rospy.is_shutdown():

            start = time.perf_counter()
            self.lb_list = np.ones(self.N - 1) * (-1e3)
            self.ub_list = np.ones(self.N - 1) * (1e3)
            self.traj_list = self.pred_traj[1:]

            # ### HJ : stage 1..N-1 의 lbx/ubx 를 **매 tick 먼저 트랙 경계로 reset**.
            # 원본은 `if self.obstacles is not None:` 블록 안에서만 solver.set 을
            # 호출했기 때문에, 한 번 장애물 때문에 n-bound 가 수축되면 obstacles 가
            # 사라진 뒤에도 acados 내부에 stale bound 가 남아 infeasible / tailing
            # 의 원인이 됨. reset → (필요 시) 추가 수축 순서로 변경.
            has_obstacles = self.obstacles is not None and len(self.obstacles) > 0
            for j in range(1, self.N):
                s_traj_mod = self.pred_traj[j] % self.spline.track_length
                traj_ub = self.model.inner_bound_s(s_traj_mod) - self.track_savety_margin
                traj_lb = -self.model.outer_bound_s(s_traj_mod) + self.track_savety_margin

                if has_obstacles:
                    for obstacle in self.obstacles:
                        obs_right_shifted = obstacle.d_right - traj_lb
                        obs_left_shifted = obstacle.d_left - traj_lb
                        if (
                            s_traj_mod >= obstacle.s_start - 0.5
                            and s_traj_mod <= obstacle.s_end + 0.1
                        ):
                            gap_left = traj_ub - obs_left_shifted - traj_lb
                            gap_right = obs_right_shifted
                            if gap_right >= gap_left:
                                traj_ub = (
                                    obstacle.d_right - 0.25
                                    if obstacle.d_right < traj_ub
                                    else traj_ub
                                )
                            else:
                                traj_lb = (
                                    obstacle.d_left + 0.25
                                    if obstacle.d_left > traj_lb
                                    else traj_lb
                                )

                self.lb_list[j - 1] = traj_lb
                self.ub_list[j - 1] = traj_ub

                # ### HJ : stage box idxbx 는 (n, vx, vy, D, δ) 5D. 원본은 4D
                # 에 self.model.v_min (존재 안 함) 을 써서, obstacles 경로가
                # 실행되는 순간 AttributeError 로 crash 하는 잠재 버그였음.
                # constraint 네임스페이스의 실제 bound 로 교정.
                lbx = np.array(
                    [
                        traj_lb,
                        self.constraint.vx_min,
                        self.constraint.vy_min,
                        self.model.throttle_min,
                        self.model.delta_min,
                    ]
                )
                ubx = np.array(
                    [
                        traj_ub,
                        self.constraint.vx_max,
                        self.constraint.vy_max,
                        self.model.throttle_max,
                        self.model.delta_max,
                    ]
                )

                self.acados_solver.set(j, "lbx", lbx)
                self.acados_solver.set(j, "ubx", ubx)

            status = self.acados_solver.solve()

            first_try_failed = status != 0
            if first_try_failed:
                rospy.logerr(f"acados returned status {status} in closed loop iteration.")

            # ### HJ : Cold-start rescue — status!=0 가 N_rescue 틱 연속이면 solver
            # 내부 상태(warm-start) 를 깨고 x0 에서 재출발. upenn_mpc 와 동일 패턴.
            # 그렇지 않으면 한 번 실패 시 자기 warm-start 를 계속 재활용해 "No new
            # solution" 로그가 무한 반복됨.
            if first_try_failed:
                self._rescue_streak = getattr(self, "_rescue_streak", 0) + 1
            else:
                self._rescue_streak = 0
            if self._rescue_streak >= 5:
                rospy.logwarn(
                    f"[wuee_mpcc] RESCUE: status!=0 for {self._rescue_streak} ticks → cold-start"
                )
                u_zero = np.zeros(self.model.u.size()[0], dtype=np.float64)
                for k in range(self.N + 1):
                    self.acados_solver.set(k, "x", propagated_x)
                for k in range(self.N):
                    self.acados_solver.set(k, "u", u_zero)
                self._rescue_streak = 0
                status = self.acados_solver.solve()

            # ### HJ : failure counting — rescue 가 성공하면 통계상 fail 로 세지 않음.
            # rescue 후에도 실패면 최종 fail 로 +1. 원본은 first-try 시점에 미리
            # +1 해버려서 rescue 가 성공해도 nr_of_failures 가 부풀려짐.
            if status != 0:
                self.nr_of_failures += 1

            # get solution
            x0 = self.acados_solver.get(0, "x")
            self.u0 = self.acados_solver.get(0, "u")
            self.pred_x = self.acados_solver.get(1, "x")

            if status == 0:
                # ### HJ : upenn-style publish (2026-04-24). stage-1 state 를
                # 그대로 쏘면 solver 의 dynamic bicycle+Pacejka+drivetrain 모델
                # 오차 (실측 1.48× 과대평가) 가 cmd 에 직접 반영되어 actuator 가
                # 못 따라감 + chatter. 대신 실측 상태 + u0·Ts 1-tick 적분으로
                # "actuator 가 실현 가능한 setpoint" 를 쏜다.
                self.publish_ackermann_msg(self.pred_x, self.u0, x0)

            if iter_loop <= PLOT_ITER:
                simX[iter_loop, :] = self.pred_x
                simU[iter_loop, :] = self.u0

            rospy.logdebug(f"[wuee_mpcc] iter {iter_loop}")

            # Creating waypoint array with predicted positions
            mpc_sd = np.array([self.acados_solver.get(j, "x")[:2] for j in range(self.N)])

            pred_waypoints = convert_frenet_to_cartesian(self.spline, mpc_sd)

            self.pred_traj = np.array(
                [self.acados_solver.get(j, "x")[State.PROGRESS_THETA] for j in range(self.N)]
            )

            progress_waypoints = convert_frenet_to_cartesian(self.spline, np.array([self.pred_traj,np.zeros(len(self.pred_traj))]).T)

            self.publish_waypoint_markers(pred_waypoints, type="pred")
            self.publish_waypoint_markers(progress_waypoints, type="target")

            # self.publish_current_pos_n(x0)

            D0 = x0[State.DUTY_CYCLE_D]
            delta0 = x0[State.STEERING_ANGLE_DELTA]

            x0 = self.get_initial_position(
                init=False, prev_x0=x0
            )

            # D (duty cycle) 는 sim 에 대응 토픽이 없어 계속 solver 내부 state 로 유지.
            x0[State.DUTY_CYCLE_D] = D0
            # δ 는 GT 측정이 있으면 이미 get_initial_position 에서 실측 δ 로 들어가 있음
            # (self.steering_angle = /car_state/steering_angle_gt). 그걸 solver-plan δ 로
            # 덮으면 다시 open-loop 가 되어버림 → GT 미수신 시에만 fallback.
            if not self._steer_meas_ok:
                x0[State.STEERING_ANGLE_DELTA] = delta0

            self.qp_iterations.append(sum(self.acados_solver.get_stats("qp_iter")))

            propagated_x = self.propagate_time_delay(x0, self.u0)
            prop_x_plot = convert_frenet_to_cartesian(self.spline, propagated_x[:2])
            self.publish_current_pos(prop_x_plot)

            # (f"s: {x0[0]}, n: {x0[1]}, alpha: {x0[2]}, v: {x0[3]}, D: {x0[4]}, delta: {x0[5]}")

            if iter_loop <= PLOT_ITER:
                realX[iter_loop, :] = x0
                propX[iter_loop, :] = propagated_x

            # Get current position of the car which is taken as x0 for the next interation
            # self.publish_current_pos(self.spline.get_coordinate(x0[0])+x0[1]*self.spline.get_derivative(x0[0]) @ R)

            self.acados_solver.set(0, "lbx", propagated_x)
            self.acados_solver.set(0, "ubx", propagated_x)

            # Print trajectory bounds
            boundaries = np.zeros((self.acados_solver.N + 1, 3, 2))
            boundaries_adj = np.zeros((self.N - 1, 2, 2))

            for stage in range(self.acados_solver.N + 1):
                x_ = self.acados_solver.get(stage, "x")
                s_ = x_[State.POS_ON_CENTER_LINE_S]
                s_mod = s_ % self.spline.track_length
                n_ = x_[State.MIN_DIST_TO_CENTER_LINE_N]
                if stage == 1:
                    if (
                        self.model.outer_bound_s(s_mod) + n_ <= 0.4
                        or self.model.inner_bound_s(s_mod) - n_ <= 0.4
                    ):
                        rospy.logwarn(
                            f"Outer: {self.model.outer_bound_s(s_mod) + n_}, Inner: {self.model.inner_bound_s(s_mod) - n_}"
                        )

                boundaries[stage, 0, :] = convert_frenet_to_cartesian(
                    self.spline,
                    np.array([s_mod, -self.model.outer_bound_s(s_mod) + self.track_savety_margin]),
                )
                boundaries[stage, 2, :] = convert_frenet_to_cartesian(
                    self.spline,
                    np.array([s_mod, self.model.inner_bound_s(s_mod) - self.track_savety_margin]),
                )
                boundaries[stage, 1, :] = convert_frenet_to_cartesian(
                    self.spline, np.array([s_mod, 0])
                )

                if stage < self.N - 1:
                    x_traj = self.traj_list[stage]
                    boundaries_adj[stage, 0, :] = convert_frenet_to_cartesian(
                        self.spline, np.array([x_traj, self.lb_list[stage]])
                    )
                    boundaries_adj[stage, 1, :] = convert_frenet_to_cartesian(
                        self.spline, np.array([x_traj, self.ub_list[stage]])
                    )

            self.publish_waypoint_markers(boundaries[:, 0, :], "d_left")
            self.publish_waypoint_markers(boundaries[:, 1, :], "d_mid")
            self.publish_waypoint_markers(boundaries[:, 2, :], "d_right")
            self.publish_waypoint_markers(boundaries_adj[:, 0, :], "d_left_adj")
            self.publish_waypoint_markers(boundaries_adj[:, 1, :], "d_right_adj")

            iter_loop += 1

            elapsed = time.perf_counter() - start
            # manage timings
            tcomp_sum += elapsed
            if elapsed > tcomp_max:
                tcomp_max = elapsed

            # print(f"MPC took {elapsed:01.5f} s.")

            if iter_loop == PLOT_ITER:
                save_plot = True
                simX_plot = simX[:iter_loop, :]
                simU_plot = simU[:iter_loop, :]
                realX_plot = realX[:iter_loop, :]
                print(f"Max. computation time: {tcomp_max}")
                print(f"Average computation time: {tcomp_sum/iter_loop}")
                print("Average speed:{}m/s".format(np.average(simX_plot[:, 3])))
                print(f"Lap times: {str(self.lap_times)[:-1]}")

                if save_plot:
                    filename = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
                    print(f"Saving files with name: {filename}")
                    with open(f"{filename}_info.txt", "w") as f:
                        f.write(f"Max. computation time: {tcomp_max} \n")
                        f.write(f"Average computation time: {tcomp_sum/iter_loop} \n")
                        f.write("Average speed:{}m/s \n".format(np.average(simX_plot[:, 3])))
                        f.write(f"Lap times: {str(self.lap_times)[:-1]} \n")
                        f.write(f"Nr of failures: {str(self.nr_of_failures)} \n")
                        f.write(
                            f"Average nr of iterations required: {str(np.mean(self.qp_iterations))} \n"
                        )

                plot_res(self.spline, simX_plot, simU_plot, realX_plot, save_plot, filename)

            rate.sleep()

    def get_initial_position(self, init: bool = False, prev_x0=None) -> np.ndarray:
        if init:
            self.pred_x = np.zeros(6)
            self.steering_angle = 0
            self.acceleration = 0
            prev_x0 = self.pred_x

        # 매 tick 실제 pose 를 mpcc 자기 spline 에 projection 해서 (s, n) 재계산.
        # /car_state/odom_frenet 은 frenet_odom_republisher 의 spline (raceline 기준) 이라
        # mpcc 가 centerline 모드로 launch 되면 1m+ offset 발생 → 직접 projection 으로 해결.
        self._project_pose_to_spline()

        deriv = self.spline.get_derivative(self.pos_s)
        alpha = self.theta - np.arctan2(deriv[1], deriv[0])

        alpha = alpha % (2 * np.pi)
        if alpha > np.pi:
            alpha = alpha - 2 * np.pi

        # print(f"alpha: {alpha}")

        # ### HJ : 원본은 IMU x-accel (m/s²) 을 D (duty cycle, PWM-like [-5,5])
        # 에 그대로 대입 — 단위/의미 불일치. MPCC 의 drivetrain 모델
        # Fx = m·(Imax_c - Cr0·vx)·D/(5·Caccel) 가 엉뚱한 값을 예측해서
        # horizon 전반의 vx 가 왜곡됨.
        # f1tenth_simulator 는 D 개념이 없고 speed setpoint 만 소비하므로
        # MPCC 의 D 는 "OCP 내부 변수" 로만 의미 있음. 매 tick 0 으로 초기화.
        duty_cicle = 0.0
        delta = (
            self.steering_angle
        )  # ToDo: check if this is correct -> minus needed since steering_angle has been inverted

        track_length = self.spline.track_length - 0.1  # Needed as values are not exact

        # ### HJ : `//` 는 float 반환이라 self.nr_laps 가 float 로 오염됨.
        # 이후 current_pos_s = self.pos_s + self.nr_laps * track_length 계산에
        # float 곱이 들어가고, `!= self.nr_laps` 비교도 float 경계에서 흔들림.
        # int 로 고정.
        prev_lap = int(prev_x0[State.POS_ON_CENTER_LINE_S] // track_length)
        if self.pos_s < 0.2 and prev_lap != self.nr_laps:
            rospy.logdebug(
                f"---------------------------------LAP {int(self.nr_laps)} FINISHED------------------------------------"
            )
            self.lap_times[-1] = time.perf_counter() - self.lap_times[-1]
            self.nr_laps = prev_lap
            self.lap_times.append(time.perf_counter())

        current_pos_s = self.pos_s + self.nr_laps * self.spline.track_length

        # ### HJ : 원본은 state[5]=duty_cicle, state[6]=self.omega 로 넣어서
        # bicycle_model.py 의 state vector 순서 (s,n,α,vx,vy,ω,D,δ,θ) 와 맞지
        # 않았음 (ω/D swap). → MPCC 가 초기 상태에서 ω=D 자리, D=ω 자리로
        # 해석, 코너 진입 시 제어 발산 원인. 원 모델 순서대로 교정.
        return np.array(
            [
                current_pos_s,  # 0: s
                self.pos_n,     # 1: n
                alpha,          # 2: α
                self.vel_x,     # 3: vx
                self.vel_y,     # 4: vy
                self.omega,     # 5: ω   (was duty_cicle — SWAP fix)
                duty_cicle,     # 6: D   (was self.omega — SWAP fix)
                delta,          # 7: δ
                current_pos_s,  # 8: θ
            ]
        )

    def publish_current_pos(self, coord: np.array) -> None:
        waypoint_marker = Marker()
        waypoint_marker.header.frame_id = "map"
        waypoint_marker.header.stamp = rospy.Time.now()
        waypoint_marker.type = 2
        waypoint_marker.scale.x = 0.2
        waypoint_marker.scale.y = 0.2
        waypoint_marker.scale.z = 0.2
        waypoint_marker.color.r = 0.0
        waypoint_marker.color.g = 1.0
        waypoint_marker.color.b = 0.0
        waypoint_marker.color.a = 1.0
        waypoint_marker.pose.position.x = coord[0]
        waypoint_marker.pose.position.y = coord[1]
        waypoint_marker.pose.position.z = 0
        waypoint_marker.pose.orientation.x = 0
        waypoint_marker.pose.orientation.y = 0
        waypoint_marker.pose.orientation.z = 0
        waypoint_marker.pose.orientation.w = 1
        waypoint_marker.id = 1
        self.pos_pub.publish(waypoint_marker)

    def publish_waypoint_markers(self, waypoints: np.ndarray, type: str) -> None:
        # rospy.logdebug("Publish waypoints")
        waypoint_markers = MarkerArray()
        wpnt_id = 0

        for waypoint in waypoints:
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = "map"
            waypoint_marker.header.stamp = rospy.Time.now()
            waypoint_marker.type = 2
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            if type == "pred":
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 0.0
                waypoint_marker.color.b = 1.0
                waypoint_marker.color.a = 1.0
            elif type == "target":
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 1.0
                waypoint_marker.color.b = 0.0
                waypoint_marker.color.a = 1.0
            else:
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 0.0
                waypoint_marker.color.b = 0.0
                waypoint_marker.color.a = 1.0

            waypoint_marker.pose.position.x = waypoint[0]
            waypoint_marker.pose.position.y = waypoint[1]
            waypoint_marker.pose.position.z = 0
            waypoint_marker.pose.orientation.x = 0
            waypoint_marker.pose.orientation.y = 0
            waypoint_marker.pose.orientation.z = 0
            waypoint_marker.pose.orientation.w = 1
            waypoint_marker.id = wpnt_id + 1
            wpnt_id += 1
            waypoint_markers.markers.append(waypoint_marker)

        if type == "pred":
            self.pred_pos_pub.publish(waypoint_markers)
        elif type == "d_left":
            self.d_left_pub.publish(waypoint_markers)
        elif type == "d_mid":
            self.d_mid_pub.publish(waypoint_markers)
        elif type == "d_right":
            self.d_right_pub.publish(waypoint_markers)
        elif type == "d_left_adj":
            self.d_left_adj_pub.publish(waypoint_markers)
        elif type == "d_right_adj":
            self.d_right_adj_pub.publish(waypoint_markers)
        else:
            self.target_pos_pub.publish(waypoint_markers)

    def publish_current_pos_n(self, state: np.array) -> None:
        position = Odometry()
        position.header.stamp = rospy.Time.now()
        position.header.frame_id = "base_link"
        position.pose.pose.position.y = state[State.MIN_DIST_TO_CENTER_LINE_N]
        self.pos_n_pub.publish(position)

    def publish_ackermann_msg(
        self,
        pred_x: np.ndarray,
        u0: np.ndarray,
        x0_stage0: np.ndarray,
    ) -> None:
        """하이브리드 publish 전략 (2026-04-24 HJ).

        - **δ_cmd = 실측 δ + u0[D_STEERING_ANGLE] · Ts**  (upenn-style)
          solver 의 stage-1 δ 를 그대로 쏘던 원본은 solver 내부 Pacejka/drivetrain
          수치 오차가 cmd 로 새어나와 mean|Δδ|=0.23 rad/tick 의 심각한 chatter
          유발. 대신 `last_delta + dδ·Ts` 로 actuator rate 범위 증분만 쏘면
          sim/실차 servo 가 깨끗이 추종.

        - **vx_cmd = stage-1 vx**  (wuee 원본 유지)
          1-tick 적분 (self.vel_x + accel·Ts) 은 초기 0 근처에서 setpoint 가
          0.05 미만으로 너무 작아 sim 이 정지 (iter7 관측). stage-1 vx 는
          solver dynamics 오차 있지만 적어도 "MPCC 가 원하는 다음 속도"를
          명시적으로 담고 있어 sim PID 가 추종 가능.

        실차 이식 시 vx 도 upenn-style 로 바꾸려면 N-step lookahead
        (`self.vel_x + accel · Ts · K`) 방식 고려 — sim-to-real 차이.
        """
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.Time.now()
        ack_msg.header.frame_id = "base_link"

        # δ: upenn-style 1-tick 적분 (chatter 차단)
        u_ddelta = float(u0[Input.D_STEERING_ANGLE])
        new_delta = float(
            np.clip(
                self.steering_angle + u_ddelta * self.Ts,
                self.model.delta_min,
                self.model.delta_max,
            )
        )

        # vx: stage-1 predicted (원본 유지)
        vx_cmd = float(pred_x[State.VELOCITY_VX])

        ack_msg.drive.steering_angle = new_delta
        ack_msg.drive.speed = vx_cmd

        self.drive_next_pub.publish(ack_msg)


if __name__ == "__main__":
    # init_node 를 먼저 호출해야 `~config_file` private param 이 정상 resolve 됨.
    rospy.init_node("mpc_node", anonymous=True, log_level=rospy.DEBUG)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.join(dir_path, "mpc", "param_config.yaml")
    conf_file = rospy.get_param("~config_file", default_cfg)
    # ### HJ : roslaunch output="screen" 이라 stderr traceback 이 rosnode log 에
    # 기록되지 않음. crash 원인 추적용으로 예외를 rospy.logfatal + 파일로 덤프.
    import traceback as _tb
    try:
        controller = MPC(conf_file=conf_file)
        controller.control_loop()
    except Exception as _e:
        _tb_str = _tb.format_exc()
        rospy.logfatal(f"[wuee_mpcc] fatal: {type(_e).__name__}: {_e}\n{_tb_str}")
        try:
            with open("/tmp/wuee_mpcc_crash.log", "a") as _f:
                _f.write(f"\n===== {datetime.now().isoformat()} =====\n{_tb_str}\n")
        except Exception:
            pass
        raise
