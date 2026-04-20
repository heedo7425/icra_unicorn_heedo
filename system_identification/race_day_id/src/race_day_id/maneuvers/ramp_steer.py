"""Ramp-steer maneuver.

Phase 1: stub — publishes AckermannDriveStamped with linearly-ramped delta,
matching id_controller Exp5 (controller_node.py:130). Phase 2 can switch to
re-using id_controller launch directly if desired.
"""
from __future__ import annotations

from .base_maneuver import BaseManeuver, ManeuverStatus


class RampSteerManeuver(BaseManeuver):
    name = "ramp_steer"

    def __init__(self, profile, rospy_mod):
        super().__init__(profile, rospy_mod)
        from ackermann_msgs.msg import AckermannDriveStamped
        self._Msg = AckermannDriveStamped
        topic = profile.get("drive_topic",
                            "/vesc/high_level/ackermann_cmd_mux/input/nav_1")
        self._pub = rospy_mod.Publisher(topic, AckermannDriveStamped, queue_size=10)
        self._t0 = None

    def start(self):
        super().start()
        self._t0 = self.rospy.get_time()

    def step(self):
        if self._status != ManeuverStatus.RUNNING:
            return
        t = self.rospy.get_time() - self._t0
        T = float(self.profile["angle_time"])
        frac = t / T
        if frac >= 1.0:
            self.stop()
            return
        a0 = float(self.profile["start_angle"])
        a1 = float(self.profile["end_angle"])
        delta = a0 + (a1 - a0) * frac

        msg = self._Msg()
        msg.header.stamp = self.rospy.Time.now()
        msg.drive.steering_angle = delta
        msg.drive.speed = float(self.profile["v_target"])
        self._pub.publish(msg)
