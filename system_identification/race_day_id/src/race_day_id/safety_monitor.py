"""Latched safety abort.

Any of:
  |omega| > abort_omega_rad
  |alpha_r| > abort_alpha_r_rad  (computed with current rosparam geometry)
  external trigger via /race_day_id/abort
sets the latch.
"""
from __future__ import annotations

import math


class SafetyMonitor:
    def __init__(self, rospy_mod, params: dict):
        self.rospy = rospy_mod
        self._omega_max = float(params.get("abort_omega_rad", 8.0))
        self._alpha_r_max = float(params.get("abort_alpha_r_rad", 0.35))
        self._latched = False
        self._reason = ""

        from nav_msgs.msg import Odometry
        from std_msgs.msg import Empty
        rospy_mod.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=10)
        rospy_mod.Subscriber("/race_day_id/abort", Empty, self._abort_cb, queue_size=1)

        self._l_r = float(rospy_mod.get_param("/vehicle/l_r", 0.145))

    def _odom_cb(self, msg):
        if self._latched:
            return
        omega = msg.twist.twist.angular.z
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        if abs(omega) > self._omega_max:
            self._latch(f"|omega|>{self._omega_max}")
            return
        if vx > 0.5:
            alpha_r = -math.atan2(vy - self._l_r * omega, vx)
            if abs(alpha_r) > self._alpha_r_max:
                self._latch(f"|alpha_r|>{self._alpha_r_max}")

    def _abort_cb(self, _msg):
        self._latch("external /race_day_id/abort")

    def _latch(self, reason):
        self._latched = True
        self._reason = reason
        try:
            self.rospy.logwarn(f"[safety_monitor] LATCH: {reason}")
        except Exception:
            pass

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def reason(self) -> str:
        return self._reason
