"""Slalom / figure-8 maneuver.

Slalom mode: sinusoidal steering with frequency chirp f0 -> f1 over sweep_s.
Figure-8 mode: piecewise-constant steering alternating +/-delta to trace two
tangent circles. Both abort if |n| > lateral_abort_m where n is the Frenet
lateral offset (when available) or the euclidean distance from the start pose.
"""
from __future__ import annotations

import math

from .base_maneuver import BaseManeuver, ManeuverStatus


class SlalomManeuver(BaseManeuver):
    name = "slalom"

    def __init__(self, profile, rospy_mod):
        super().__init__(profile, rospy_mod)
        from ackermann_msgs.msg import AckermannDriveStamped
        from nav_msgs.msg import Odometry

        self._Msg = AckermannDriveStamped
        self._pub = rospy_mod.Publisher(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_1",
            AckermannDriveStamped, queue_size=10)

        self._A = float(profile.get("amplitude_rad", 0.20))
        self._f0 = float(profile.get("freq_start_hz", 0.5))
        self._f1 = float(profile.get("freq_end_hz", 2.0))
        self._T = float(profile.get("sweep_s", 10.0))
        self._v = float(profile.get("v_fixed", 2.5))
        self._fig8 = bool(profile.get("figure_eight", False))
        self._lat_abort = float(profile.get("lateral_abort_m", 2.5))

        self._t0 = None
        self._x0 = None
        self._y0 = None
        self._x = 0.0
        self._y = 0.0
        # Prefer frenet n when published.
        self._have_frenet = False
        self._n = 0.0

        rospy_mod.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=10)
        try:
            rospy_mod.Subscriber("/car_state/odom_frenet", Odometry,
                                 self._frenet_cb, queue_size=10)
        except Exception:
            pass

    def _odom_cb(self, msg):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y

    def _frenet_cb(self, msg):
        self._have_frenet = True
        # odom_frenet convention used in upenn_trainer: pose.position.x = s,
        # pose.position.y = n.
        self._n = msg.pose.pose.position.y

    def start(self):
        super().start()
        self._t0 = self.rospy.get_time()

    def _lateral_offset(self):
        if self._have_frenet:
            return abs(self._n)
        if self._x0 is None:
            self._x0, self._y0 = self._x, self._y
            return 0.0
        return math.hypot(self._x - self._x0, self._y - self._y0)

    def step(self):
        if self._status != ManeuverStatus.RUNNING:
            return
        t = self.rospy.get_time() - self._t0
        if t >= self._T:
            self.stop()
            return

        if self._lateral_offset() > self._lat_abort:
            self.stop(f"lateral_abort n={self._lateral_offset():.2f}>"
                      f"{self._lat_abort:.2f}")
            return

        if self._fig8:
            half = self._T / 2.0
            sign = 1.0 if t < half else -1.0
            delta = sign * self._A
        else:
            f = self._f0 + (self._f1 - self._f0) * (t / self._T)
            delta = self._A * math.sin(2.0 * math.pi * f * t)

        msg = self._Msg()
        msg.header.stamp = self.rospy.Time.now()
        msg.drive.steering_angle = delta
        msg.drive.speed = self._v
        self._pub.publish(msg)
