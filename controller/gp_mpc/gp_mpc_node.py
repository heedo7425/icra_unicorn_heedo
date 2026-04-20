#!/usr/bin/env python3
"""
gp_mpc controller node — global raceline tracking with runtime μ.

Fork of controller/mpc_only/mpc_only_node.py. Adds a runtime μ input: instead
of hard-coding mu_default, the node subscribes to /gp_mpc/mu_estimate (Float32)
and injects the received value into every stage of the OCP preview.

μ source selection is done OUTSIDE this node via launch arg mu_source:
  static        — no subscriber, use mu_default from yaml
  ground_truth  — mu_patch_publisher publishes /mu_ground_truth (patch lookup)
                  remapped to /gp_mpc/mu_estimate at launch
  rls           — mu_estimator_rls publishes /gp_mpc/mu_estimate
  gp            — mu_estimator_gp  publishes /gp_mpc/mu_estimate

RViz topics:
  /gp_mpc/prediction   MarkerArray   predicted xy over horizon (green)
  /gp_mpc/reference    MarkerArray   raceline window (orange)
  /gp_mpc/solve_ms     Float32       solve time per tick
  /gp_mpc/mu_used      Float32       μ value actually fed into OCP this tick
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rospy
import tf.transformations as tft
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Float32MultiArray, Bool
from visualization_msgs.msg import Marker, MarkerArray

# Import shared mpc modules (kept in controller/mpc/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)  # -> controller/
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from mpc.reference_builder import build_preview, extract_frenet_state  # noqa: E402
from mpc.vehicle_model import NU, NX, NP, load_vehicle_params_from_ros  # noqa: E402
from gp_mpc.mpcc_ocp_gp import NP_GP  # noqa: E402


# Column indices for our "wpnts" ndarray (matches mpc_node.py expectations).
# [x, y, z, vx_mps, safety_ratio(unused=0), s_m, kappa_radpm, psi_rad, ax_mps2, d_m]
C_X, C_Y, C_Z, C_VX, _, C_S, C_KAPPA, C_PSI, C_AX, C_D = range(10)


class EkfMpcNode:
    def __init__(self) -> None:
        rospy.init_node("gp_mpc_controller", anonymous=False)
        self.name = "gp_mpc_controller"
        self.lock = threading.Lock()

        # ---- Params (under /gp_mpc/* namespace) ----
        NS = "gp_mpc"
        self.loop_rate = float(rospy.get_param(f"{NS}/loop_rate_hz", 50))
        self.test_mode = bool(rospy.get_param(f"{NS}/test_mode", False))
        self.N = int(rospy.get_param(f"{NS}/N_horizon", 20))
        self.dt = float(rospy.get_param(f"{NS}/dt", 0.05))
        self.v_max = float(rospy.get_param(f"{NS}/v_max", 12.0))
        self.max_steer = float(rospy.get_param(f"{NS}/max_steer", 0.4))
        self.mu_default = float(rospy.get_param(f"{NS}/mu_default", 0.85))
        # raceline vx_ref 배수 (1.0 = 원본 scaled raceline). 부하 테스트용.
        self.speed_boost = float(rospy.get_param(f"{NS}/speed_boost", 1.0))
        # μ-aware vx scaling: effective = speed_boost * (μ_gt/μ_default)^mu_scale_exp
        # - 0.5 = sqrt (friction-circle v∝√μ 물리적 가정)
        # - 0.0 = 끔 (μ 무시)
        # - 1.0 = 선형 (공격적)
        self.mu_scale_exp = float(rospy.get_param(f"{NS}/mu_scale_exp", 0.5))
        # μ-aware 용 latest ground-truth μ
        self._mu_gt_latest = self.mu_default
        rospy.Subscriber("/mu_ground_truth", Float32,
                         lambda m: setattr(self, "_mu_gt_latest", float(m.data)),
                         queue_size=1)
        self.window_size = int(rospy.get_param(f"{NS}/window_size", 200))

        self.mpc_cfg = {
            "N_horizon": self.N,
            "dt": self.dt,
            "v_max": self.v_max,
            "v_min": float(rospy.get_param(f"{NS}/v_min", 0.0)),
            "max_steer": self.max_steer,
            "max_steer_rate": float(rospy.get_param(f"{NS}/max_steer_rate", 2.0)),
            "max_accel": float(rospy.get_param(f"{NS}/max_accel", 3.0)),
            "max_decel": float(rospy.get_param(f"{NS}/max_decel", -3.0)),
            "w_d": float(rospy.get_param(f"{NS}/w_d", 8.0)),
            "w_dpsi": float(rospy.get_param(f"{NS}/w_dpsi", 6.0)),
            "w_vx": float(rospy.get_param(f"{NS}/w_vx", 3.0)),
            "w_vy": float(rospy.get_param(f"{NS}/w_vy", 1.0)),
            "w_omega": float(rospy.get_param(f"{NS}/w_omega", 3.0)),
            "w_steer": float(rospy.get_param(f"{NS}/w_steer", 0.5)),
            "w_u_steer_rate": float(rospy.get_param(f"{NS}/w_u_steer_rate", 3.0)),
            "w_u_accel": float(rospy.get_param(f"{NS}/w_u_accel", 0.1)),
            "w_terminal_scale": float(rospy.get_param(f"{NS}/w_terminal_scale", 1.0)),
            "friction_margin": float(rospy.get_param(f"{NS}/friction_margin", 0.95)),
            "friction_slack_penalty": float(rospy.get_param(f"{NS}/friction_slack_penalty", 1e3)),
        }
        self.vp = load_vehicle_params_from_ros(rospy)

        rospy.loginfo(
            f"[{self.name}] init N={self.N} dt={self.dt} v_max={self.v_max} "
            f"mu={self.mu_default} window={self.window_size} test_mode={self.test_mode}"
        )

        # ---- Build solver ----
        # 두 번째 solver (base) 를 같은 OCP 구조로 하나 더 생성해서 residual=0
        # 로 병렬 solve → GP 활성 시 cmd vs base 비교용.
        self.solver = None
        self.solver_base = None
        self.publish_base_cmp = bool(rospy.get_param(f"{NS}/publish_base_cmp", True))
        if not self.test_mode:
            try:
                from gp_mpc.mpcc_ocp_gp import build_tracking_ocp_gp  # noqa: E402
                t0 = time.perf_counter()
                codegen_dir = rospy.get_param(
                    f"{NS}/codegen_dir", "/tmp/gp_mpc_c_generated"
                )
                self.solver = build_tracking_ocp_gp(
                    self.vp, self.mpc_cfg, codegen_dir=codegen_dir
                )
                rospy.loginfo(
                    f"[{self.name}] acados OCP built in "
                    f"{(time.perf_counter() - t0):.1f}s → {codegen_dir}"
                )
                if self.publish_base_cmp:
                    t0 = time.perf_counter()
                    # Same builder, but we'll always pass residual=0 → base baseline.
                    # Separate solver 필요 — warm-start state 분리.
                    self.solver_base = build_tracking_ocp_gp(
                        self.vp, self.mpc_cfg,
                        codegen_dir=codegen_dir + "_base",
                        model_name="tracking_ocp_gp_base",
                    )
                    rospy.loginfo(
                        f"[{self.name}] base-comparison solver built in "
                        f"{(time.perf_counter() - t0):.1f}s"
                    )
            except Exception as e:
                rospy.logerr(f"[{self.name}] OCP build failed: {e}. Falling back to test_mode.")
                self.test_mode = True
                self.solver = None

        # ---- State ----
        self.odom: Optional[Odometry] = None
        self.pose: Optional[PoseStamped] = None
        self.frenet_odom: Optional[Odometry] = None
        self.imu: Optional[Imu] = None
        self.global_wpnts_np: Optional[np.ndarray] = None  # (M, 10) cached raceline
        self.track_length = float(rospy.get_param("/global_republisher/track_length", 0.0))

        self.last_delta = 0.0

        # Teleport / reset detection (same pattern as mpc_node).
        self._prev_car_xy: Optional[Tuple[float, float]] = None
        self._reset_jump_thres_m = float(rospy.get_param(f"{NS}/reset_jump_thres_m", 1.0))

        # Warm-start rescue counter.
        self._stuck_count = 0
        self._stuck_thres = int(rospy.get_param(f"{NS}/stuck_status_thres", 5))

        # ---- Pub / Sub ----
        self.drive_topic = rospy.get_param(
            "~drive_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_1"
        )
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=10)
        self.initialpose_pub = rospy.Publisher(
            "/initialpose", PoseWithCovarianceStamped, queue_size=1, latch=True
        )
        self.pred_pub = rospy.Publisher("/gp_mpc/prediction", MarkerArray, queue_size=1)
        self.ref_pub = rospy.Publisher("/gp_mpc/reference", MarkerArray, queue_size=1)
        self.solve_time_pub = rospy.Publisher("/gp_mpc/solve_ms", Float32, queue_size=1)
        self.mu_used_pub = rospy.Publisher("/gp_mpc/mu_used", Float32, queue_size=1)
        # Base comparison (residual=0 parallel solve): 속도/조향 비교용
        self.cmd_base_speed_pub = rospy.Publisher("/gp_mpc/cmd_base_speed", Float32, queue_size=1)
        self.cmd_base_steer_pub = rospy.Publisher("/gp_mpc/cmd_base_steer", Float32, queue_size=1)

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb)
        rospy.Subscriber("/car_state/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._frenet_cb)
        rospy.Subscriber("/imu/data", Imu, self._imu_cb)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self._gbscaled_cb, queue_size=1)
        rospy.Subscriber("/global_waypoints", WpntArray, self._gbraw_cb, queue_size=1)

        # --- Runtime μ source (injected into OCP preview each tick) ---
        self.mu_source = str(rospy.get_param(f"{NS}/mu_source", "static"))
        self.mu_runtime: float = self.mu_default
        self.mu_adapt_enable: bool = True   # 토글 가능한 adaptation on/off
        if self.mu_source != "static":
            mu_topic = str(rospy.get_param(f"{NS}/mu_estimate_topic", "/gp_mpc/mu_estimate"))
            rospy.Subscriber(mu_topic, Float32, self._mu_cb, queue_size=1)
            rospy.loginfo(f"[{self.name}] μ source = {self.mu_source} (subscribing {mu_topic})")
        else:
            rospy.loginfo(f"[{self.name}] μ source = static (mu={self.mu_default})")
        # Runtime toggle for μ adaptation (independent of mu_source).
        rospy.Subscriber("/gp_mpc/mu_adapt_enable", Bool, self._mu_enable_cb, queue_size=1)

        # --- GP residual injection ---
        # /gp_mpc/residual (Float32MultiArray, data=[Δvx, Δvy, Δω]) broadcast
        # across N+1 stages when gp_ready=True and residual_enable config is on.
        self.gp_residual = np.zeros(3, dtype=np.float64)
        self.gp_ready: bool = False
        self.residual_enable = bool(rospy.get_param(f"{NS}/gp/residual_enable", True))
        rospy.Subscriber("/gp_mpc/residual", Float32MultiArray,
                         self._gp_residual_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/gp_ready", Bool,
                         self._gp_ready_cb, queue_size=1)
        from std_msgs.msg import Empty
        rospy.Subscriber("/gp_mpc/gp_reset", Empty,
                         self._gp_reset_cb, queue_size=1)

        self.startup_delay_s = float(rospy.get_param(f"{NS}/startup_delay_s", 3.0))
        self._start_time = rospy.Time.now().to_sec()

        self.warmup_vx_min = float(rospy.get_param(f"{NS}/warmup_vx_min", 0.2))
        self.warmup_speed_cmd = float(rospy.get_param(f"{NS}/warmup_speed_cmd", 2.0))
        # warmup 은 최초 출발 시에만 활성. vx 가 한 번이라도 이 값 초과하면 영구 비활성.
        # (저마찰 코너에서 vx 가 떨어져도 steer=0 으로 하이재킹되어 벽에 박히는 것 방지)
        self.warmup_exit_vx = float(rospy.get_param(f"{NS}/warmup_exit_vx", 0.8))
        self._warmup_armed = True
        # 충돌 후 stuck 감지: vx<warmup_vx_min 가 이 시간 이상 지속되면 warmup 재무장
        self.crash_stuck_sec = float(rospy.get_param(f"{NS}/crash_stuck_sec", 1.5))
        self._stuck_ticks = 0

        rospy.loginfo(f"[{self.name}] subscribed. publishing {self.drive_topic}")

    # ---- Callbacks ----
    def _odom_cb(self, msg: Odometry) -> None:
        with self.lock:
            self.odom = msg

    def _pose_cb(self, msg: PoseStamped) -> None:
        with self.lock:
            self.pose = msg

    def _frenet_cb(self, msg: Odometry) -> None:
        with self.lock:
            self.frenet_odom = msg

    def _imu_cb(self, msg: Imu) -> None:
        with self.lock:
            self.imu = msg

    def _mu_cb(self, msg: Float32) -> None:
        # Clamp to a sane range to protect OCP (physical μ ∈ (0, 1.5]).
        mu = float(msg.data)
        if not (0.05 < mu < 1.5):
            return
        with self.lock:
            self.mu_runtime = mu

    def _mu_enable_cb(self, msg) -> None:
        new_state = bool(msg.data)
        with self.lock:
            changed = new_state != self.mu_adapt_enable
            self.mu_adapt_enable = new_state
        rospy.loginfo(f"[{self.name}] μ adaptation {'ENABLED' if new_state else 'DISABLED'}")
        if changed:
            self._respawn_and_reset_solvers()

    def _gp_residual_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 3:
            return
        with self.lock:
            self.gp_residual = np.array(msg.data[:3], dtype=np.float64)

    def _gp_ready_cb(self, msg: Bool) -> None:
        with self.lock:
            self.gp_ready = bool(msg.data)

    def _gp_reset_cb(self, _msg) -> None:
        rospy.loginfo(f"[{self.name}] GP reset received — respawn + cold-start")
        self._respawn_and_reset_solvers()

    def _gbscaled_cb(self, msg: WpntArray) -> None:
        self._cache_wpnts(msg, source="scaled")

    def _gbraw_cb(self, msg: WpntArray) -> None:
        # Only accept raw if scaled never arrived.
        if self.global_wpnts_np is None:
            self._cache_wpnts(msg, source="raw")

    def _cache_wpnts(self, msg: WpntArray, source: str) -> None:
        arr = np.array([
            [w.x_m, w.y_m, w.z_m, w.vx_mps, 0.0, w.s_m, w.kappa_radpm, w.psi_rad, w.ax_mps2, w.d_m]
            for w in msg.wpnts
        ], dtype=np.float64)
        if arr.shape[0] < 2:
            return
        with self.lock:
            self.global_wpnts_np = arr
            if self.track_length <= 0:
                # Infer from last wpnt's s if rosparam missing.
                self.track_length = float(arr[-1, C_S])
        rospy.loginfo_throttle(
            10.0,
            f"[{self.name}] global raceline cached from {source}: "
            f"{arr.shape[0]} wpnts, s∈[{arr[0,C_S]:.2f},{arr[-1,C_S]:.2f}], "
            f"vx∈[{arr[:,C_VX].min():.2f},{arr[:,C_VX].max():.2f}]",
        )

    def _ready(self) -> bool:
        return (
            self.odom is not None
            and self.pose is not None
            and self.frenet_odom is not None
            and self.global_wpnts_np is not None
            and self.global_wpnts_np.shape[0] > self.N + 2
        )

    # ---- Window slicing ----
    def _slice_window(self, car_x: float, car_y: float) -> np.ndarray:
        """Return (window_size, 10) slice of global raceline starting at the
        wpnt nearest to car_xy, wrapping around track_length."""
        gw = self.global_wpnts_np
        M = gw.shape[0]
        # nearest-waypoint search on xy (full raceline is ~800 wpnts, cheap).
        dxy2 = (gw[:, C_X] - car_x) ** 2 + (gw[:, C_Y] - car_y) ** 2
        idx_near = int(np.argmin(dxy2))
        idx = (np.arange(self.window_size) + idx_near) % M
        return gw[idx]

    # ---- State extraction ----
    def _current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        with self.lock:
            od = self.odom
            po = self.pose
            fr = self.frenet_odom
            im = self.imu

        car_x = float(po.pose.position.x)
        car_y = float(po.pose.position.y)

        wpnts = self._slice_window(car_x, car_y)

        # wpnt index 0 is nearest-to-car by construction.
        wpx, wpy, psi_ref = wpnts[0, C_X], wpnts[0, C_Y], wpnts[0, C_PSI]
        dx, dy = car_x - wpx, car_y - wpy
        n_local = -math.sin(psi_ref) * dx + math.cos(psi_ref) * dy

        q = po.pose.orientation
        yaw_world = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        dpsi = self._wrap_pi(yaw_world - psi_ref)

        vx = float(od.twist.twist.linear.x)
        vy = float(od.twist.twist.linear.y)
        omega = float(im.angular_velocity.z) if im is not None else 0.0

        self._ego_yaw = yaw_world
        self._n_local_raw = n_local

        n_clip = float(np.clip(n_local, -1.0, 1.0))
        x0 = extract_frenet_state(0.0, n_clip, dpsi, vx, vy, omega, self.last_delta)
        return x0, wpnts

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _cold_start_solver(self, x0: np.ndarray, include_base: bool = False) -> None:
        if self.solver is None:
            return
        u_zero = np.zeros(NU, dtype=np.float64)
        for k in range(self.N + 1):
            self.solver.set(k, "x", x0)
        for k in range(self.N):
            self.solver.set(k, "u", u_zero)
        if include_base and self.solver_base is not None:
            for k in range(self.N + 1):
                self.solver_base.set(k, "x", x0)
            for k in range(self.N):
                self.solver_base.set(k, "u", u_zero)
        self.last_delta = 0.0

    def _respawn_and_reset_solvers(self) -> None:
        """차를 wpnt[0] 으로 respawn + 두 solver cold-start.
        toggle / gp_reset 시 깨끗한 baseline 확보용.
        """
        gw = self.global_wpnts_np
        if gw is None or len(gw) < 5:
            rospy.logwarn(f"[{self.name}] respawn skipped — no global waypoints")
            return
        # wpnt[0] + tangent from wpnt[5] (spawn_on_waypoint 과 동일)
        x0 = float(gw[0, C_X]); y0 = float(gw[0, C_Y])
        x1 = float(gw[5 % len(gw), C_X]); y1 = float(gw[5 % len(gw), C_Y])
        yaw = math.atan2(y1 - y0, x1 - x0)
        from geometry_msgs.msg import PoseWithCovarianceStamped
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = x0
        msg.pose.pose.position.y = y0
        qz = math.sin(yaw / 2.0); qw = math.cos(yaw / 2.0)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance = [0.0] * 36
        # Latched publish × a few for reliability.
        for _ in range(5):
            self.initialpose_pub.publish(msg)
            rospy.sleep(0.05)
        # Zero-state cold start on both solvers.
        x0_state = np.zeros(NX, dtype=np.float64)
        self._cold_start_solver(x0_state, include_base=True)
        self._warmup_armed = True
        self._stuck_ticks = 0
        self._stuck_count = 0
        rospy.logwarn(f"[{self.name}] RESPAWN (wpnt[0]) + solvers cold-started")

    # ---- Solve ----
    def _solve(self, x0: np.ndarray, wpnts: np.ndarray) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
        from gp_mpc.mpcc_ocp_gp import solve_once_gp

        # Use runtime μ (patch/rls/gp) only if adaptation enabled; else fallback.
        if self.mu_source != "static" and self.mu_adapt_enable:
            mu_for_stage = self.mu_runtime
        else:
            mu_for_stage = self.mu_default
        params_base = build_preview(
            wpnts,
            s_ego=0.0,
            N=self.N,
            dt=self.dt,
            track_length=self.track_length,
            mu_default=mu_for_stage,
        )
        # Speed boost + μ-aware scaling.
        # 물리: friction-circle 내에서 v ∝ √(μ · g / κ). 그립 많을수록 더 빠르게.
        # μ_gt 는 patch 관측치 (현재 위치만). 전체 horizon 에 broadcast — 다음
        # patch 로 넘어가는 구간엔 조금 과·감속 가능 (safety 는 friction_circle 제약).
        if self.speed_boost != 1.0 or self.mu_scale_exp != 0.0:
            mu_ratio = max(self._mu_gt_latest, 0.1) / max(self.mu_default, 0.1)
            mu_factor = mu_ratio ** self.mu_scale_exp
            scale = self.speed_boost * mu_factor
            params_base[:, 4] = np.minimum(params_base[:, 4] * scale, self.v_max)
        # Extend params with 3 residual columns (broadcast across N+1 stages).
        # gp_ready False OR residual_enable False OR mu_adapt_enable False → zero.
        with self.lock:
            use_residual = (
                self.residual_enable and self.gp_ready and self.mu_adapt_enable
            )
            res_vec = self.gp_residual.copy() if use_residual else np.zeros(3)
        residual_block = np.tile(res_vec, (params_base.shape[0], 1))
        params = np.concatenate([params_base, residual_block], axis=1)
        assert params.shape[1] == NP_GP, \
            f"params shape {params.shape} != expected (_, {NP_GP})"
        self.mu_used_pub.publish(Float32(data=float(mu_for_stage)))

        if not np.all(np.isfinite(x0)):
            rospy.logerr_throttle(1.0, f"[{self.name}] x0 non-finite: {x0}")
            return 0.0, 0.0, 0.0, 0.0, None
        if not np.all(np.isfinite(params)):
            bad = np.argwhere(~np.isfinite(params))
            rospy.logerr_throttle(1.0, f"[{self.name}] params non-finite at {bad[:5].tolist()}")
            return 0.0, 0.0, 0.0, 0.0, None

        # Active solver 선택:
        #   adapt ON  → solver (residual 주입) 이 주행
        #   adapt OFF → solver_base (residual=0) 이 주행. 순수 base 동작.
        # 비교용으로는 항상 다른 쪽 solver 의 u0 도 publish.
        if self.mu_adapt_enable:
            u0, status, info = solve_once_gp(self.solver, x0, params)
            # 비교: base solver w/ residual=0
            if self.solver_base is not None and self.publish_base_cmp:
                try:
                    params_zero = params.copy()
                    params_zero[:, NP:] = 0.0
                    u0_base, _, _ = solve_once_gp(self.solver_base, x0, params_zero)
                    self.cmd_base_speed_pub.publish(Float32(data=float(np.clip(
                        x0[3] + float(u0_base[1]) * self.dt, 0.0, self.v_max))))
                    self.cmd_base_steer_pub.publish(Float32(data=float(np.clip(
                        self.last_delta + float(u0_base[0]) * self.dt,
                        -self.max_steer, self.max_steer))))
                except Exception as e:
                    rospy.logwarn_throttle(5.0, f"[{self.name}] base cmp solve fail: {e}")
        else:
            # adapt OFF: base solver 로 주행. params 의 residual 은 이미 0.
            if self.solver_base is not None:
                u0, status, info = solve_once_gp(self.solver_base, x0, params)
            else:
                u0, status, info = solve_once_gp(self.solver, x0, params)
            # 비교: GP solver w/ 실제 residual (있으면) — publish_base_cmp 는 이제
            # "활성 solver 의 반대쪽 publish" 의미.
            if self.solver is not None and self.publish_base_cmp:
                try:
                    # GP residual 주입한 버전
                    params_gp = params.copy()
                    with self.lock:
                        if self.gp_ready:
                            params_gp[:, NP:] = self.gp_residual
                    u0_gp, _, _ = solve_once_gp(self.solver, x0, params_gp)
                    # cmd_base_speed/steer 토픽은 "비활성 solver (= GP)" 가 내놓는 값
                    self.cmd_base_speed_pub.publish(Float32(data=float(np.clip(
                        x0[3] + float(u0_gp[1]) * self.dt, 0.0, self.v_max))))
                    self.cmd_base_steer_pub.publish(Float32(data=float(np.clip(
                        self.last_delta + float(u0_gp[0]) * self.dt,
                        -self.max_steer, self.max_steer))))
                except Exception as e:
                    rospy.logwarn_throttle(5.0, f"[{self.name}] gp cmp solve fail: {e}")

        if status == 0:
            self._stuck_count = 0
        else:
            self._stuck_count += 1
            if self._stuck_count >= self._stuck_thres:
                rospy.logwarn(
                    f"[{self.name}] RESCUE: status!=0 for {self._stuck_count} ticks → cold-start"
                )
                active = self.solver if self.mu_adapt_enable else (self.solver_base or self.solver)
                # Cold-start active solver 만.
                u_zero = np.zeros(NU, dtype=np.float64)
                for k in range(self.N + 1):
                    active.set(k, "x", x0)
                for k in range(self.N):
                    active.set(k, "u", u_zero)
                self.last_delta = 0.0
                self._stuck_count = 0
                u0, status, info = solve_once_gp(active, x0, params)

        active_solver = self.solver if self.mu_adapt_enable else (self.solver_base or self.solver)
        x0_solver = active_solver.get(0, "x")
        x_end = active_solver.get(self.N, "x")
        rospy.loginfo_throttle(
            1.0,
            f"[{self.name}] x0={x0[:4].round(3)} x0_s={x0_solver[:4].round(3)} "
            f"x_end={x_end[:4].round(3)}",
        )
        self.solve_time_pub.publish(Float32(data=info["solve_time_ms"]))

        if status not in (0, 2, 4):
            rospy.logwarn_throttle(
                1.0,
                f"[{self.name}] solve status={status} time={info['solve_time_ms']:.1f}ms — safing",
            )
            return 0.0, 0.0, 0.0, 0.0, None

        u_ddelta = float(u0[0])
        u_ax = float(u0[1])

        new_delta = float(np.clip(
            self.last_delta + u_ddelta * self.dt,
            -self.max_steer, self.max_steer,
        ))
        self.last_delta = new_delta

        vx_cmd = float(np.clip(x0[3] + u_ax * self.dt, 0.0, self.v_max))

        # Horizon xy in world frame.
        M = wpnts.shape[0]
        traj = np.empty((self.N + 1, 2))
        for k in range(self.N + 1):
            xk = self.solver.get(k, "x")
            n_k = float(xk[1])
            idx = min(k, M - 1)
            wx, wy, wpsi = wpnts[idx, C_X], wpnts[idx, C_Y], wpnts[idx, C_PSI]
            traj[k, 0] = wx - n_k * math.sin(wpsi)
            traj[k, 1] = wy + n_k * math.cos(wpsi)

        return vx_cmd, u_ax, 0.0, new_delta, traj

    # ---- Viz ----
    def _viz_trajectory(self, traj: np.ndarray, pub: rospy.Publisher, color: Tuple[float, float, float]) -> None:
        ma = MarkerArray()
        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = rospy.Time.now()
        line.ns = pub.name
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.05
        line.color.r, line.color.g, line.color.b = color
        line.color.a = 0.9
        line.pose.orientation.w = 1.0
        for xy in traj:
            p = Point()
            p.x, p.y, p.z = float(xy[0]), float(xy[1]), 0.0
            line.points.append(p)
        ma.markers.append(line)
        pub.publish(ma)

    def _publish_drive(self, speed: float, accel: float, jerk: float, steer: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(np.clip(speed, 0.0, self.v_max))
        msg.drive.acceleration = accel
        msg.drive.jerk = jerk
        msg.drive.steering_angle = float(np.clip(steer, -self.max_steer, self.max_steer))
        self.drive_pub.publish(msg)

    def run(self) -> None:
        rate = rospy.Rate(self.loop_rate)
        warned_waiting = False
        while not rospy.is_shutdown():
            if self.test_mode:
                self._publish_drive(0.0, 0.0, 0.0, 0.0)
                rate.sleep()
                continue

            if not self._ready():
                if not warned_waiting:
                    rospy.loginfo_throttle(2.0, f"[{self.name}] waiting for inputs (odom/pose/frenet/global_wpnts)…")
                self._publish_drive(0.0, 0.0, 0.0, 0.0)
                rate.sleep()
                continue
            warned_waiting = True

            elapsed = rospy.Time.now().to_sec() - self._start_time
            if elapsed < self.startup_delay_s:
                self._publish_drive(0.0, 0.0, 0.0, 0.0)
                rospy.loginfo_throttle(
                    1.0,
                    f"[{self.name}] startup hold ({elapsed:.1f}/{self.startup_delay_s:.1f}s)",
                )
                rate.sleep()
                continue

            try:
                x0, wpnts = self._current_state()

                # Teleport detection.
                car_xy = (self.pose.pose.position.x, self.pose.pose.position.y)
                if self._prev_car_xy is not None:
                    ddx = car_xy[0] - self._prev_car_xy[0]
                    ddy = car_xy[1] - self._prev_car_xy[1]
                    jump = math.hypot(ddx, ddy)
                    if jump > self._reset_jump_thres_m:
                        rospy.logwarn(
                            f"[{self.name}] RESET detected (Δxy={jump:.2f}m) → cold-start solver"
                        )
                        self._cold_start_solver(x0)
                        self._prev_car_xy = car_xy
                        self._publish_drive(0.0, 0.0, 0.0, 0.0)
                        rate.sleep()
                        continue
                self._prev_car_xy = car_xy

                # OFF-TRACK halt.
                if abs(self._n_local_raw) > 1.5:
                    rospy.logwarn_throttle(
                        1.0,
                        f"[{self.name}] OFF-TRACK (n={self._n_local_raw:+.2f}m); halted",
                    )
                    self._publish_drive(0.0, 0.0, 0.0, 0.0)
                    rate.sleep()
                    continue

                speed, accel, jerk, steer, traj = self._solve(x0, wpnts)

                vx_now = float(x0[3])
                if self._warmup_armed and vx_now >= self.warmup_exit_vx:
                    self._warmup_armed = False
                # Stuck 복구: vx < warmup_vx_min 이 crash_stuck_sec 연속이면
                # (= 충돌 후 정지) warmup 재무장 → kick 명령으로 빠져나옴.
                if vx_now < self.warmup_vx_min:
                    self._stuck_ticks = getattr(self, "_stuck_ticks", 0) + 1
                else:
                    self._stuck_ticks = 0
                crash_ticks = int(self.crash_stuck_sec * self.loop_rate)
                if self._stuck_ticks >= crash_ticks and not self._warmup_armed:
                    self._warmup_armed = True
                    rospy.logwarn(
                        f"[{self.name}] CRASH RECOVERY — warmup re-armed after "
                        f"{self._stuck_ticks / self.loop_rate:.1f}s stuck"
                    )
                if self._warmup_armed and vx_now < self.warmup_vx_min:
                    speed = self.warmup_speed_cmd
                    accel = 1.0
                    steer = 0.0
                    self.last_delta = 0.0
                    rospy.loginfo_throttle(
                        1.0,
                        f"[{self.name}] WARMUP (vx={vx_now:.2f} < {self.warmup_vx_min}) → speed={speed} steer=0",
                    )
                self._publish_drive(speed, accel, jerk, steer)

                # Viz decimation: 50Hz control → 10Hz prediction, 2Hz reference.
                # Software-rendered RViz (libGL nouveau fallback) 에서 per-tick
                # publish 가 프레임 박살 내므로 drop.
                if traj is not None:
                    self._viz_tick = getattr(self, "_viz_tick", 0) + 1
                    if self._viz_tick % 5 == 0:
                        self._viz_trajectory(traj, self.pred_pub, (0.1, 0.9, 0.2))
                    if self._viz_tick % 25 == 0:
                        self._viz_trajectory(wpnts[:, :2], self.ref_pub, (0.9, 0.6, 0.1))
            except Exception as e:
                rospy.logerr_throttle(1.0, f"[{self.name}] solve exception: {e}")
                self._publish_drive(0.0, 0.0, 0.0, 0.0)

            rate.sleep()


if __name__ == "__main__":
    EkfMpcNode().run()
