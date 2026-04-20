"""Steady-state circle maneuver.

Fixed steering delta, stepwise increase speed through v_levels, each held
dwell_s seconds. Abort when |ay| stays below ay_sat_ratio * expected across the
dwell window (implies tire already saturated — no sysid info beyond this).
"""
from __future__ import annotations

import math

from .base_maneuver import BaseManeuver, ManeuverStatus


class SteadyCircleManeuver(BaseManeuver):
    name = "steady_circle"

    def __init__(self, profile, rospy_mod):
        super().__init__(profile, rospy_mod)
        from ackermann_msgs.msg import AckermannDriveStamped
        from sensor_msgs.msg import Imu
        from nav_msgs.msg import Odometry

        self._Msg = AckermannDriveStamped
        self._pub = rospy_mod.Publisher(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_1",
            AckermannDriveStamped, queue_size=10)

        self._delta = float(profile.get("delta_fixed", 0.25))
        self._levels = list(profile.get("v_levels", [1.0, 2.0, 3.0]))
        self._dwell = float(profile.get("dwell_s", 3.0))
        self._ay_sat_ratio = float(profile.get("ay_sat_ratio", 0.3))
        self._radius_min = float(profile.get("radius_min_m", 2.0))

        self._ay = 0.0
        self._vx = 0.0
        rospy_mod.Subscriber("/vesc/sensors/imu/raw", Imu, self._imu_cb, queue_size=10)
        rospy_mod.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=10)

        self._t0 = None
        self._idx = -1
        self._level_t0 = 0.0
        self._level_ay_max = 0.0

    def _imu_cb(self, msg):
        self._ay = msg.linear_acceleration.y

    def _odom_cb(self, msg):
        self._vx = msg.twist.twist.linear.x

    def start(self):
        super().start()
        self._t0 = self.rospy.get_time()
        self._level_t0 = self._t0
        self._idx = 0
        self._level_ay_max = 0.0

    def step(self):
        if self._status != ManeuverStatus.RUNNING:
            return
        now = self.rospy.get_time()
        t_in_level = now - self._level_t0

        # Advance level when dwell elapsed.
        if t_in_level >= self._dwell:
            # Evaluate saturation for this level before advancing.
            v = self._levels[self._idx]
            expected_ay = v * v / max(self._radius_min, 1e-3)
            if self._level_ay_max < self._ay_sat_ratio * expected_ay:
                self.stop(f"ay_sat_abort lvl={v:.1f} ay_max={self._level_ay_max:.2f} "
                          f"expected≈{expected_ay:.2f}")
                return
            self._idx += 1
            if self._idx >= len(self._levels):
                self.stop()
                return
            self._level_t0 = now
            self._level_ay_max = 0.0

        self._level_ay_max = max(self._level_ay_max, abs(self._ay))

        msg = self._Msg()
        msg.header.stamp = self.rospy.Time.now()
        msg.drive.steering_angle = self._delta
        msg.drive.speed = float(self._levels[self._idx])
        self._pub.publish(msg)
