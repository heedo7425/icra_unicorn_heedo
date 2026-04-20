"""Ring-buffer recorder for sysid data.

Subscribes to car state and commands, appends timestamped tuples into a fixed
circular buffer, and flushes to CSV on request. Phase 1: basic captures.
"""
from __future__ import annotations

import csv
import os
from collections import deque
from typing import Deque, List, Tuple


class Recorder:
    """Maintain synchronized samples [t, vx, vy, omega, delta, ax]."""

    COLUMNS = ("t", "vx", "vy", "omega", "delta", "ax")

    def __init__(self, rospy_mod, buffer_s: float = 120.0, hz: float = 70.0):
        self.rospy = rospy_mod
        self._buf: Deque[Tuple[float, ...]] = deque(maxlen=int(buffer_s * hz))

        # Latched latest measurements; ticker samples them.
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0
        self._delta = 0.0
        self._ax = 0.0
        self._have_odom = False

        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Float64
        from ackermann_msgs.msg import AckermannDriveStamped

        rospy_mod.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=10)
        rospy_mod.Subscriber("/vesc/sensors/imu/raw", Imu, self._imu_cb, queue_size=10)
        rospy_mod.Subscriber("/vesc/high_level/ackermann_cmd_mux/input/nav_1",
                             AckermannDriveStamped, self._cmd_cb, queue_size=10)

        self._recording = False

    # ---- subs ----
    def _odom_cb(self, msg):
        tw = msg.twist.twist
        self._vx = tw.linear.x
        self._vy = tw.linear.y
        self._omega = tw.angular.z
        self._have_odom = True

    def _imu_cb(self, msg):
        self._ax = msg.linear_acceleration.x

    def _cmd_cb(self, msg):
        self._delta = msg.drive.steering_angle

    # ---- control ----
    def start(self):
        self._buf.clear()
        self._recording = True

    def stop(self):
        self._recording = False

    def sample(self):
        if not self._recording or not self._have_odom:
            return
        t = self.rospy.get_time()
        self._buf.append((t, self._vx, self._vy, self._omega, self._delta, self._ax))

    def flush_to_csv(self, path: str) -> int:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.COLUMNS)
            w.writerows(self._buf)
        return len(self._buf)

    def as_arrays(self):
        import numpy as np
        if not self._buf:
            return None
        data = np.array(self._buf, dtype=float)
        return {c: data[:, i] for i, c in enumerate(self.COLUMNS)}
