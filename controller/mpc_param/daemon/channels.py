"""channels.py — in-memory ring buffer subscriber.

Phase 1. Subscribes to the topics needed by the metric calculator, keeps a
thread-safe time-stamped ring buffer per channel. The daemon snapshots a
[t_start, t_end] window per lap.

ROS-dependent. Import from inside the icra2026 container.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import rospy
import tf.transformations as tft
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import LapData, WpntArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int8

# Wpnt column layout (mirror upenn_mpc_node.py)
C_X, C_Y, C_Z, C_VX, _, C_S, C_KAPPA, C_PSI, C_AX, C_D, C_MU_RAD = range(11)


@dataclass
class LapEvent:
    """Snapshot delivered by the lap watcher each time a lap closes."""
    lap_count: int
    lap_time: float            # from LapData
    avg_lat_err: float
    max_lat_err: float
    t_start: float             # ROS time (s) of lap start
    t_end: float               # ROS time (s) of lap end


class ChannelBuffer:
    """Single-channel ring buffer of (t, value)."""
    __slots__ = ("_buf", "_lock")

    def __init__(self, maxlen: int = 50_000):
        self._buf: Deque[Tuple[float, float]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, t: float, v) -> None:
        with self._lock:
            self._buf.append((t, v))

    def slice(self, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (times, values) within [t0, t1]. Inclusive."""
        with self._lock:
            data = [(t, v) for t, v in self._buf if t0 <= t <= t1]
        if not data:
            return np.empty(0), np.empty(0)
        ts, vs = zip(*data)
        return np.asarray(ts, dtype=np.float64), np.asarray(vs, dtype=np.float64)

    def latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else (None, None)


class ChannelHub:
    """Subscribes to all channels needed by the metric calculator.

    Public API:
        hub = ChannelHub(target='upenn_mpc')
        hub.set_lap_callback(fn)        # fn(LapEvent) at every lap close
        hub.snapshot(t0, t1) -> dict    # arrays per channel
        hub.raceline()        -> ndarray (M,11) or None
    """

    def __init__(self, target: str = "upenn_mpc",
                 lap_topic: str = "lap_data",
                 buf_seconds: float = 600.0,
                 expected_rate_hz: float = 100.0):
        self.target = target
        maxlen = int(buf_seconds * expected_rate_hz) + 1024

        # ---- Channels (HOT path: append on every msg) ----
        self.ch: Dict[str, ChannelBuffer] = {
            name: ChannelBuffer(maxlen=maxlen) for name in [
                "vx", "vy", "yaw", "x", "y",       # car_state
                "s", "n",                            # frenet
                "omega", "ay_imu",                   # imu
                "delta_cmd", "speed_cmd",            # drive
                "solve_ms",                          # mpc telemetry
                "mu_used",
                "infeasible",                        # 0 or 1 events; Phase 3 wires real source
            ]
        }

        self._raceline_lock = threading.Lock()
        self._raceline: Optional[np.ndarray] = None      # (M,11)
        self._track_length: float = 0.0

        self._lap_cb = None
        self._lap_t_anchor: Optional[float] = None       # rolling lap start time
        self._last_lap_count: Optional[int] = None

        # ---- Subscribers ----
        rospy.Subscriber("/car_state/odom",       Odometry,    self._cb_odom,    queue_size=50)
        rospy.Subscriber("/car_state/pose",       PoseStamped, self._cb_pose,    queue_size=50)
        rospy.Subscriber("/car_state/odom_frenet", Odometry,   self._cb_frenet,  queue_size=50)
        rospy.Subscriber("/imu/data",             Imu,         self._cb_imu,     queue_size=100)
        # Drive cmd: try common nav_1 path; user can override.
        drive_topic = rospy.get_param("~drive_topic",
                                      "/vesc/high_level/ackermann_cmd_mux/input/nav_1")
        rospy.Subscriber(drive_topic,             AckermannDriveStamped,
                                                     self._cb_drive, queue_size=50)
        rospy.Subscriber(f"/{target}/solve_ms",   Float32,     self._cb_solve,   queue_size=50)
        rospy.Subscriber(f"/{target}/solve_status", Int8,      self._cb_status,  queue_size=50)
        rospy.Subscriber(f"/{target}/mu_used",    Float32,     self._cb_mu,      queue_size=50)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self._cb_raceline, queue_size=1)
        rospy.Subscriber(lap_topic,               LapData,     self._cb_lap,     queue_size=10)

        rospy.loginfo(f"[mpc_param.channels] subscribed (target={target}, lap={lap_topic})")

    # ---- Callbacks ----
    def _now(self) -> float:
        return rospy.Time.now().to_sec()

    def _cb_odom(self, m: Odometry) -> None:
        t = m.header.stamp.to_sec() or self._now()
        self.ch["vx"].append(t, float(m.twist.twist.linear.x))
        self.ch["vy"].append(t, float(m.twist.twist.linear.y))

    def _cb_pose(self, m: PoseStamped) -> None:
        t = m.header.stamp.to_sec() or self._now()
        self.ch["x"].append(t, float(m.pose.position.x))
        self.ch["y"].append(t, float(m.pose.position.y))
        q = m.pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.ch["yaw"].append(t, float(yaw))

    def _cb_frenet(self, m: Odometry) -> None:
        t = m.header.stamp.to_sec() or self._now()
        # frenet odom convention in this stack: pose.position.x = s, .y = n
        self.ch["s"].append(t, float(m.pose.pose.position.x))
        self.ch["n"].append(t, float(m.pose.pose.position.y))

    def _cb_imu(self, m: Imu) -> None:
        t = m.header.stamp.to_sec() or self._now()
        self.ch["omega"].append(t, float(m.angular_velocity.z))
        self.ch["ay_imu"].append(t, float(m.linear_acceleration.y))

    def _cb_drive(self, m: AckermannDriveStamped) -> None:
        t = m.header.stamp.to_sec() or self._now()
        self.ch["delta_cmd"].append(t, float(m.drive.steering_angle))
        self.ch["speed_cmd"].append(t, float(m.drive.speed))

    def _cb_solve(self, m: Float32) -> None:
        t = self._now()
        self.ch["solve_ms"].append(t, float(m.data))

    def _cb_mu(self, m: Float32) -> None:
        self.ch["mu_used"].append(self._now(), float(m.data))

    def _cb_status(self, m: Int8) -> None:
        # 0=ok, 2=max_iter (often acceptable), 4=qp_fail, etc.
        # 비정상 (status not in {0,2}) 만 infeasible 카운트.
        s = int(m.data)
        if s not in (0, 2):
            self.ch["infeasible"].append(self._now(), 1.0)

    def _cb_raceline(self, m: WpntArray) -> None:
        if not m.wpnts:
            return
        arr = np.array([
            [w.x_m, w.y_m, w.z_m, w.vx_mps, 0.0, w.s_m,
             w.kappa_radpm, w.psi_rad, w.ax_mps2, w.d_m,
             getattr(w, "mu_rad", 0.0)]
            for w in m.wpnts
        ], dtype=np.float64)
        with self._raceline_lock:
            self._raceline = arr
            self._track_length = float(arr[-1, C_S])

    def _cb_lap(self, m: LapData) -> None:
        t_end = m.header.stamp.to_sec() or self._now()
        if self._lap_t_anchor is None:
            # first lap_data: anchor; emit nothing.
            self._lap_t_anchor = t_end
            self._last_lap_count = m.lap_count
            rospy.loginfo(f"[mpc_param.channels] anchored at lap #{m.lap_count}")
            return
        evt = LapEvent(
            lap_count=int(m.lap_count),
            lap_time=float(m.lap_time),
            avg_lat_err=float(m.average_lateral_error_to_global_waypoints),
            max_lat_err=float(m.max_lateral_error_to_global_waypoints),
            t_start=self._lap_t_anchor,
            t_end=t_end,
        )
        self._lap_t_anchor = t_end
        self._last_lap_count = m.lap_count
        if self._lap_cb is not None:
            try:
                self._lap_cb(evt)
            except Exception as e:
                rospy.logerr(f"[mpc_param.channels] lap callback error: {e}")

    # ---- Public ----
    def set_lap_callback(self, fn) -> None:
        self._lap_cb = fn

    def raceline(self) -> Optional[np.ndarray]:
        with self._raceline_lock:
            return None if self._raceline is None else self._raceline.copy()

    def track_length(self) -> float:
        with self._raceline_lock:
            return self._track_length

    def snapshot(self, t0: float, t1: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Per-channel (times, values) within window."""
        return {name: buf.slice(t0, t1) for name, buf in self.ch.items()}

    def report_infeasible(self) -> None:
        """Daemon calls this when the node logs a solve failure (Phase 3)."""
        self.ch["infeasible"].append(self._now(), 1.0)
