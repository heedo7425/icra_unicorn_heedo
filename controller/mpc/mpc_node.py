#!/usr/bin/env python3
"""
MPC controller node — dynamic bicycle + Pacejka tracking MPC (Stage 1).

Pipeline per 50 Hz tick:
  1. Gather ego state (s, n, dpsi, vx, vy, omega, delta) from ROS topics.
  2. Build (N+1, 6) preview params from local waypoints.
  3. Call acados OCP solver. Get u0 = [delta_rate, a_x].
  4. Integrate delta_rate * dt to get new steering command.
  5. Publish AckermannDriveStamped + MarkerArray for RViz.

Stage 1 limitations (documented):
- μ is fixed from /mpc/mu_default (no estimator yet — Stage 2).
- θ (grade), κ_z (vertical curvature) are zero (no 3D — Stage 3).
- dpsi estimation uses waypoint tangent at current s (coarse).

RViz topics:
  /mpc/prediction       (MarkerArray)  — predicted xy trajectory (N+1 points)
  /mpc/reference        (MarkerArray)  — reference path preview
  /mpc/solve_debug      (Float32)      — last solve time (ms)
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
from f110_msgs.msg import BehaviorStrategy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

# Local module (same dir). Python auto-adds the script's own dir (controller/mpc)
# to sys.path, but we need its parent (controller/) so `from mpc.x import y` works.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from mpc.reference_builder import (  # noqa: E402
    WPT_KAPPA,
    WPT_S,
    build_preview,
    extract_frenet_state,
)
from mpc.vehicle_model import NX, NU, load_vehicle_params_from_ros  # noqa: E402


class MpcNode:
    def __init__(self) -> None:
        rospy.init_node("mpc_controller", anonymous=False)
        self.name = "mpc_controller"
        self.lock = threading.Lock()

        # ---- Params ----
        self.loop_rate = float(rospy.get_param("mpc/loop_rate_hz", 50))
        self.test_mode = bool(rospy.get_param("mpc/test_mode", False))
        self.N = int(rospy.get_param("mpc/N_horizon", 20))
        self.dt = float(rospy.get_param("mpc/dt", 0.05))
        self.v_max = float(rospy.get_param("mpc/v_max", 8.0))
        self.max_steer = float(rospy.get_param("mpc/max_steer", 0.4))
        self.mu_default = float(rospy.get_param("mpc/mu_default", 0.7))

        self.mpc_cfg = {
            "N_horizon": self.N,
            "dt": self.dt,
            "v_max": self.v_max,
            "v_min": float(rospy.get_param("mpc/v_min", 0.0)),
            "max_steer": self.max_steer,
            "max_steer_rate": float(rospy.get_param("mpc/max_steer_rate", 3.5)),
            "max_accel": float(rospy.get_param("mpc/max_accel", 5.0)),
            "max_decel": float(rospy.get_param("mpc/max_decel", -6.0)),
            "w_d": float(rospy.get_param("mpc/w_d", 10.0)),
            "w_dpsi": float(rospy.get_param("mpc/w_dpsi", 5.0)),
            "w_vx": float(rospy.get_param("mpc/w_vx", 1.0)),
            "w_vy": float(rospy.get_param("mpc/w_vy", 0.5)),
            "w_omega": float(rospy.get_param("mpc/w_omega", 0.1)),
            "w_steer": float(rospy.get_param("mpc/w_steer", 0.01)),
            "w_u_steer_rate": float(rospy.get_param("mpc/w_u_steer_rate", 0.5)),
            "w_u_accel": float(rospy.get_param("mpc/w_u_accel", 0.05)),
            "w_terminal_scale": float(rospy.get_param("mpc/w_terminal_scale", 10.0)),
        }
        self.vp = load_vehicle_params_from_ros(rospy)

        rospy.loginfo(
            f"[{self.name}] init N={self.N} dt={self.dt} v_max={self.v_max} "
            f"test_mode={self.test_mode}"
        )

        # ---- Build solver (unless test_mode — saves 10s codegen on pure topic test) ----
        self.solver = None
        if not self.test_mode:
            try:
                from mpc.mpcc_ocp import build_tracking_ocp  # noqa: E402
                t0 = time.perf_counter()
                codegen_dir = rospy.get_param(
                    "mpc/codegen_dir", "/tmp/mpc_c_generated"
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
        self.behavior: Optional[BehaviorStrategy] = None
        self.imu: Optional[Imu] = None
        self.track_length = float(rospy.get_param("/global_republisher/track_length", 0.0))

        # Track last issued steering (since model state includes delta)
        self.last_delta = 0.0

        # Reset detection: spawn_on_waypoint.py teleports the car via /initialpose
        # after a crash. Pose jumps ≫ 1 tick of real motion → the solver's
        # warm-start (stages 1..N) is stale and SQP_RTI (1 iter) can't recover.
        # Track prev car xy; on jump, force solver cold start.
        self._prev_car_xy: Optional[Tuple[float, float]] = None
        self._reset_jump_thres_m = float(
            rospy.get_param("mpc/reset_jump_thres_m", 1.0)
        )  # >1m in 20ms = 50 m/s → only a teleport

        # Warm-start rescue: SQP_RTI does 1 iter/tick. If previous solution is
        # infeasible (QP status≠0 repeatedly), solver gets stuck in bad local
        # minimum. After N consecutive bad-status ticks, wipe warm-start and
        # re-solve with fresh seed.
        self._stuck_count = 0
        self._stuck_thres = int(rospy.get_param("mpc/stuck_status_thres", 5))

        # ---- Pub / Sub ----
        self.drive_topic = rospy.get_param(
            "~drive_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_1"
        )
        self.drive_pub = rospy.Publisher(
            self.drive_topic, AckermannDriveStamped, queue_size=10
        )
        self.pred_pub = rospy.Publisher("/mpc/prediction", MarkerArray, queue_size=1)
        self.ref_pub = rospy.Publisher("/mpc/reference", MarkerArray, queue_size=1)
        self.solve_time_pub = rospy.Publisher("/mpc/solve_ms", Float32, queue_size=1)

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb)
        rospy.Subscriber("/car_state/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._frenet_cb)
        rospy.Subscriber("/behavior_strategy", BehaviorStrategy, self._behavior_cb)
        rospy.Subscriber("/imu/data", Imu, self._imu_cb)
        # Startup delay: hold zero cmd for a few seconds so spawn_on_waypoint
        # can teleport the sim car before MPC starts driving. Without this,
        # MPC computes a wild cmd from the invalid initial pose and the car
        # drifts away from where spawn tried to place it.
        self.startup_delay_s = float(rospy.get_param("mpc/startup_delay_s", 3.0))
        self._start_time = rospy.Time.now().to_sec()

        # Bootstrap (warm-up) params. Pacejka Jacobian is near-singular at vx≈0
        # (slip-angle change w.r.t. δ vanishes), so HPIPM returns ACADOS_MINSTEP
        # and MPC can't find an accel/steer that lifts the car out of standstill.
        # Override MPC with a gentle forward push until vx exceeds warmup_vx_min.
        self.warmup_vx_min = float(rospy.get_param("mpc/warmup_vx_min", 0.8))
        self.warmup_speed_cmd = float(rospy.get_param("mpc/warmup_speed_cmd", 1.2))

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

    def _behavior_cb(self, msg: BehaviorStrategy) -> None:
        with self.lock:
            self.behavior = msg

    def _imu_cb(self, msg: Imu) -> None:
        with self.lock:
            self.imu = msg

    def _ready(self) -> bool:
        return (
            self.odom is not None
            and self.pose is not None
            and self.frenet_odom is not None
            and self.behavior is not None
            and self.behavior.local_wpnts is not None
            and len(self.behavior.local_wpnts) > 1
        )

    # ---- State extraction ----
    def _current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x0, wpnts_array).

        Frenet state:
            s, n = from /car_state/odom_frenet
            dpsi = (yaw_world - psi_ref(s))  with psi_ref from nearest wpnt
            vx, vy = /car_state/odom body twist
            omega = /imu/data z-angular-vel
            delta = last commanded
        """
        with self.lock:
            od = self.odom
            po = self.pose
            fr = self.frenet_odom
            bh = self.behavior
            im = self.imu

        wpnts = np.array([
            [w.x_m, w.y_m, w.z_m, w.vx_mps, 0.0, w.s_m, w.kappa_radpm, w.psi_rad, w.ax_mps2, w.d_m]
            for w in bh.local_wpnts
        ])
        # --- Local Frenet state ---
        # x0[0] ≡ 0 (local s); params at stage k come from wpnts[k] index.
        # n is computed with the left-positive normal (-sin ψ, cos ψ) — this
        # matches MPC dynamics `dn/dt = vx sin(dpsi) + vy cos(dpsi)` where at
        # dpsi=0, positive body vy (ROS-left) increases n.
        # (Note: /car_state/odom_frenet uses the repo's right-positive d
        # convention — opposite sign. Don't consume that directly here.)

        car_x = float(po.pose.position.x)
        car_y = float(po.pose.position.y)
        d_sq = (wpnts[:, 0] - car_x) ** 2 + (wpnts[:, 1] - car_y) ** 2
        idx_near = int(np.argmin(d_sq))
        wpx, wpy, psi_ref = wpnts[idx_near, 0], wpnts[idx_near, 1], wpnts[idx_near, 7]
        dx, dy = car_x - wpx, car_y - wpy
        n_local = -math.sin(psi_ref) * dx + math.cos(psi_ref) * dy

        # Heading: world yaw from pose quaternion
        q = po.pose.orientation
        yaw_world = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        dpsi = self._wrap_pi(yaw_world - psi_ref)

        vx = float(od.twist.twist.linear.x)
        vy = float(od.twist.twist.linear.y)
        omega = float(im.angular_velocity.z) if im is not None else 0.0

        # cache nearest wpnt info for viz
        self._ego_idx_near = idx_near
        self._ego_yaw = yaw_world
        self._n_local_raw = n_local  # unclamped, for off-track detection

        # Clamp n only (to avoid `1 - n·kappa` singularity in dynamics). dpsi is
        # NOT clamped — MPC must see the true heading error to correct it.
        # A clamp on dpsi hides the correction need from the solver.
        n_clip = float(np.clip(n_local, -1.0, 1.0))

        x0 = extract_frenet_state(0.0, n_clip, dpsi, vx, vy, omega, self.last_delta)
        return x0, wpnts

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _cold_start_solver(self, x0: np.ndarray) -> None:
        """Wipe acados warm-start: seed every stage with x0, zero all inputs.

        Needed after a teleport (crash respawn). Without this, stages 1..N still
        carry the pre-reset horizon and SQP_RTI's 1 iteration can't bridge the
        gap, so predicted trajectory stays glued to the old pose.
        """
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

        params = build_preview(
            wpnts,
            s_ego=0.0,              # state is LOCAL; params indexed by stage k
            N=self.N,
            dt=self.dt,
            track_length=self.track_length,
            mu_default=self.mu_default,
        )

        # NaN sanity: log exactly what's bad before feeding solver.
        if not np.all(np.isfinite(x0)):
            rospy.logerr_throttle(1.0, f"[{self.name}] x0 has non-finite: {x0}")
            return 0.0, 0.0, 0.0, 0.0, None
        if not np.all(np.isfinite(params)):
            bad = np.argwhere(~np.isfinite(params))
            rospy.logerr_throttle(1.0, f"[{self.name}] params non-finite at {bad[:5].tolist()}")
            return 0.0, 0.0, 0.0, 0.0, None

        u0, status, info = solve_once(self.solver, x0, params)

        # Warm-start rescue: if solver keeps returning non-zero status, the
        # warm-start from previous solves is stuck. Cold-start and re-solve.
        if status == 0:
            self._stuck_count = 0
        else:
            self._stuck_count += 1
            if self._stuck_count >= self._stuck_thres:
                rospy.logwarn(
                    f"[{self.name}] RESCUE: status!=0 for {self._stuck_count} "
                    f"ticks → cold-start warm-start"
                )
                self._cold_start_solver(x0)
                self._stuck_count = 0
                u0, status, info = solve_once(self.solver, x0, params)

        # DEBUG: verify initial state constraint is honored by solver.
        x0_solver = self.solver.get(0, "x")
        x_end = self.solver.get(self.N, "x")
        rospy.loginfo_throttle(
            1.0,
            f"[{self.name}] x0_input={x0[:4].round(3)} "
            f"x0_solver={x0_solver[:4].round(3)} "
            f"x_end={x_end[:4].round(3)}",
        )
        self.solve_time_pub.publish(Float32(data=info["solve_time_ms"]))

        # acados QP status: 0 = success, 2 = max iter NLP, 4 = QP max iter (best-effort ok)
        if status not in (0, 2, 4):
            rospy.logwarn_throttle(
                1.0,
                f"[{self.name}] solve status={status} time={info['solve_time_ms']:.1f}ms — safing",
            )
            return 0.0, 0.0, 0.0, 0.0, None
        if status != 0:
            rospy.logwarn_throttle(
                2.0,
                f"[{self.name}] solve status={status} (suboptimal, accepted) time={info['solve_time_ms']:.1f}ms "
                f"x0[s,n,dpsi,vx]=[{x0[0]:.2f},{x0[1]:.2f},{x0[2]:.2f},{x0[3]:.2f}]",
            )

        u_ddelta = float(u0[0])
        u_ax = float(u0[1])

        # Steering command: integrate delta_rate over one dt, clip to max_steer.
        new_delta = float(np.clip(
            self.last_delta + u_ddelta * self.dt,
            -self.max_steer,
            self.max_steer,
        ))
        self.last_delta = new_delta

        # Speed command: forward-integrate vx (use current vx + u_ax * dt)
        vx_cmd = float(np.clip(x0[3] + u_ax * self.dt, 0.0, self.v_max))

        # Horizon trajectory in world frame.
        # MPC state is LOCAL (x0[0]=0), and params were sampled from wpnts[k].
        # So predicted stage k corresponds to wpnts[min(k, M-1)] — use that for
        # world xy + lateral n offset.
        M = wpnts.shape[0]
        traj = np.empty((self.N + 1, 2))
        for k in range(self.N + 1):
            xk = self.solver.get(k, "x")
            n_k = float(xk[1])
            idx = min(k, M - 1)
            wx, wy, wpsi = wpnts[idx, 0], wpnts[idx, 1], wpnts[idx, 7]
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

    # ---- Run ----
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
        warmup_warned = False
        while not rospy.is_shutdown():
            if self.test_mode:
                self._publish_drive(0.0, 0.0, 0.0, 0.0)
                rate.sleep()
                continue

            if not self._ready():
                if not warmup_warned:
                    rospy.loginfo_throttle(
                        2.0,
                        f"[{self.name}] waiting for inputs…",
                    )
                rate.sleep()
                continue
            warmup_warned = True

            # Hold zero cmd during startup_delay so spawn can settle.
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

                # Teleport / respawn detection. If car xy jumped > thres since
                # last tick, flush solver warm-start and zero last_delta.
                car_xy = (self.pose.pose.position.x, self.pose.pose.position.y)
                if self._prev_car_xy is not None:
                    ddx = car_xy[0] - self._prev_car_xy[0]
                    ddy = car_xy[1] - self._prev_car_xy[1]
                    jump = math.hypot(ddx, ddy)
                    if jump > self._reset_jump_thres_m:
                        rospy.logwarn(
                            f"[{self.name}] RESET detected (Δxy={jump:.2f}m > "
                            f"{self._reset_jump_thres_m}m) → cold-start solver"
                        )
                        self._cold_start_solver(x0)
                        # Also clear "last state" so next tick is a fresh start
                        self._prev_car_xy = car_xy
                        self._publish_drive(0.0, 0.0, 0.0, 0.0)
                        rate.sleep()
                        continue
                self._prev_car_xy = car_xy

                # Off-track detection: if car is too far laterally, MPC output
                # is meaningless and we're just making it worse. Publish zeros
                # and wait for supervisor / manual respawn.
                if abs(self._n_local_raw) > 1.5:
                    rospy.logwarn_throttle(
                        1.0,
                        f"[{self.name}] OFF-TRACK (n={self._n_local_raw:+.2f}m); MPC halted until respawn",
                    )
                    self._publish_drive(0.0, 0.0, 0.0, 0.0)
                    rate.sleep()
                    continue

                speed, accel, jerk, steer, traj = self._solve(x0, wpnts)

                # Warm-up push: MPC's QP collapses at vx≈0 (Pacejka singular).
                # Override with pure forward motion — steer=0 to avoid wall hits.
                # Once vx > warmup_vx_min, let MPC take over.
                vx_now = float(x0[3])
                if vx_now < self.warmup_vx_min:
                    speed = self.warmup_speed_cmd
                    accel = 1.0
                    steer = 0.0
                    self.last_delta = 0.0
                    rospy.loginfo_throttle(
                        1.0,
                        f"[{self.name}] WARMUP straight (vx={vx_now:.2f} < {self.warmup_vx_min}) → speed={speed} steer=0",
                    )
                self._publish_drive(speed, accel, jerk, steer)
                if traj is not None:
                    self._viz_trajectory(traj, self.pred_pub, (0.1, 0.9, 0.2))
                    # reference preview (from wpnts x, y)
                    ref_traj = wpnts[:, :2]
                    self._viz_trajectory(ref_traj, self.ref_pub, (0.9, 0.6, 0.1))
            except Exception as e:
                rospy.logerr_throttle(1.0, f"[{self.name}] solve exception: {e}")
                self._publish_drive(0.0, 0.0, 0.0, 0.0)

            rate.sleep()


if __name__ == "__main__":
    MpcNode().run()
