#!/usr/bin/env python3
"""
multisurface_mpc controller node — global raceline tracking with runtime μ.

Fork of controller/mpc_only/mpc_only_node.py. Adds a runtime μ input: instead
of hard-coding mu_default, the node subscribes to /mpc_ms/mu_estimate (Float32)
and injects the received value into every stage of the OCP preview.

μ source selection is done OUTSIDE this node via launch arg mu_source:
  static        — no subscriber, use mu_default from yaml
  ground_truth  — mu_patch_publisher publishes /mu_ground_truth (patch lookup)
                  remapped to /mpc_ms/mu_estimate at launch
  rls           — mu_estimator_rls publishes /mpc_ms/mu_estimate
  gp            — mu_estimator_gp  publishes /mpc_ms/mu_estimate

RViz topics:
  /mpc_ms/prediction   MarkerArray   predicted xy over horizon (green)
  /mpc_ms/reference    MarkerArray   raceline window (orange)
  /mpc_ms/solve_ms     Float32       solve time per tick
  /mpc_ms/mu_used      Float32       μ value actually fed into OCP this tick
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
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

# Import shared mpc modules (kept in controller/mpc/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)  # -> controller/
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from mpc.reference_builder import build_preview, extract_frenet_state  # noqa: E402
from mpc.vehicle_model import NU, NX, load_vehicle_params_from_ros  # noqa: E402


# Column indices for our "wpnts" ndarray (matches mpc_node.py expectations).
# [x, y, z, vx_mps, safety_ratio(unused=0), s_m, kappa_radpm, psi_rad, ax_mps2, d_m]
C_X, C_Y, C_Z, C_VX, _, C_S, C_KAPPA, C_PSI, C_AX, C_D = range(10)


class MpcMsNode:
    def __init__(self) -> None:
        rospy.init_node("mpc_ms_controller", anonymous=False)
        self.name = "mpc_ms_controller"
        self.lock = threading.Lock()

        # ---- Params (under /mpc_ms/* namespace) ----
        NS = "mpc_ms"
        self.loop_rate = float(rospy.get_param(f"{NS}/loop_rate_hz", 50))
        self.test_mode = bool(rospy.get_param(f"{NS}/test_mode", False))
        self.N = int(rospy.get_param(f"{NS}/N_horizon", 20))
        self.dt = float(rospy.get_param(f"{NS}/dt", 0.05))
        self.v_max = float(rospy.get_param(f"{NS}/v_max", 12.0))
        self.max_steer = float(rospy.get_param(f"{NS}/max_steer", 0.4))
        self.mu_default = float(rospy.get_param(f"{NS}/mu_default", 0.85))
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
        self.solver = None
        if not self.test_mode:
            try:
                from mpc.mpcc_ocp import build_tracking_ocp  # noqa: E402
                t0 = time.perf_counter()
                codegen_dir = rospy.get_param(
                    f"{NS}/codegen_dir", "/tmp/mpc_ms_c_generated"
                )
                self.solver = build_tracking_ocp(
                    self.vp, self.mpc_cfg, codegen_dir=codegen_dir
                )
                rospy.loginfo(
                    f"[{self.name}] acados OCP built in "
                    f"{(time.perf_counter() - t0):.1f}s → {codegen_dir}"
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
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=10)
        self.pred_pub = rospy.Publisher("/mpc_ms/prediction", MarkerArray, queue_size=1)
        self.ref_pub = rospy.Publisher("/mpc_ms/reference", MarkerArray, queue_size=1)
        self.solve_time_pub = rospy.Publisher("/mpc_ms/solve_ms", Float32, queue_size=1)
        self.mu_used_pub = rospy.Publisher("/mpc_ms/mu_used", Float32, queue_size=1)

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
            mu_topic = str(rospy.get_param(f"{NS}/mu_estimate_topic", "/mpc_ms/mu_estimate"))
            rospy.Subscriber(mu_topic, Float32, self._mu_cb, queue_size=1)
            rospy.loginfo(f"[{self.name}] μ source = {self.mu_source} (subscribing {mu_topic})")
        else:
            rospy.loginfo(f"[{self.name}] μ source = static (mu={self.mu_default})")
        # Runtime toggle for μ adaptation (independent of mu_source).
        from std_msgs.msg import Bool
        rospy.Subscriber("/mpc_ms/mu_adapt_enable", Bool, self._mu_enable_cb, queue_size=1)

        self.startup_delay_s = float(rospy.get_param(f"{NS}/startup_delay_s", 3.0))
        self._start_time = rospy.Time.now().to_sec()

        self.warmup_vx_min = float(rospy.get_param(f"{NS}/warmup_vx_min", 0.2))
        self.warmup_speed_cmd = float(rospy.get_param(f"{NS}/warmup_speed_cmd", 2.0))

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
        with self.lock:
            self.mu_adapt_enable = bool(msg.data)
        rospy.loginfo(f"[{self.name}] μ adaptation {'ENABLED' if self.mu_adapt_enable else 'DISABLED'}")

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

    def _cold_start_solver(self, x0: np.ndarray) -> None:
        if self.solver is None:
            return
        u_zero = np.zeros(NU, dtype=np.float64)
        for k in range(self.N + 1):
            self.solver.set(k, "x", x0)
        for k in range(self.N):
            self.solver.set(k, "u", u_zero)
        self.last_delta = 0.0

    # ---- Solve ----
    def _solve(self, x0: np.ndarray, wpnts: np.ndarray) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
        from mpc.mpcc_ocp import solve_once

        # Use runtime μ (patch/rls/gp) only if adaptation enabled; else fallback.
        if self.mu_source != "static" and self.mu_adapt_enable:
            mu_for_stage = self.mu_runtime
        else:
            mu_for_stage = self.mu_default
        params = build_preview(
            wpnts,
            s_ego=0.0,
            N=self.N,
            dt=self.dt,
            track_length=self.track_length,
            mu_default=mu_for_stage,
        )
        self.mu_used_pub.publish(Float32(data=float(mu_for_stage)))

        if not np.all(np.isfinite(x0)):
            rospy.logerr_throttle(1.0, f"[{self.name}] x0 non-finite: {x0}")
            return 0.0, 0.0, 0.0, 0.0, None
        if not np.all(np.isfinite(params)):
            bad = np.argwhere(~np.isfinite(params))
            rospy.logerr_throttle(1.0, f"[{self.name}] params non-finite at {bad[:5].tolist()}")
            return 0.0, 0.0, 0.0, 0.0, None

        u0, status, info = solve_once(self.solver, x0, params)

        if status == 0:
            self._stuck_count = 0
        else:
            self._stuck_count += 1
            if self._stuck_count >= self._stuck_thres:
                rospy.logwarn(
                    f"[{self.name}] RESCUE: status!=0 for {self._stuck_count} ticks → cold-start"
                )
                self._cold_start_solver(x0)
                self._stuck_count = 0
                u0, status, info = solve_once(self.solver, x0, params)

        x0_solver = self.solver.get(0, "x")
        x_end = self.solver.get(self.N, "x")
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
                if vx_now < self.warmup_vx_min:
                    speed = self.warmup_speed_cmd
                    accel = 1.0
                    steer = 0.0
                    self.last_delta = 0.0
                    rospy.loginfo_throttle(
                        1.0,
                        f"[{self.name}] WARMUP (vx={vx_now:.2f} < {self.warmup_vx_min}) → speed={speed} steer=0",
                    )
                self._publish_drive(speed, accel, jerk, steer)

                if traj is not None:
                    self._viz_trajectory(traj, self.pred_pub, (0.1, 0.9, 0.2))
                    self._viz_trajectory(wpnts[:, :2], self.ref_pub, (0.9, 0.6, 0.1))
            except Exception as e:
                rospy.logerr_throttle(1.0, f"[{self.name}] solve exception: {e}")
                self._publish_drive(0.0, 0.0, 0.0, 0.0)

            rate.sleep()


if __name__ == "__main__":
    MpcMsNode().run()
