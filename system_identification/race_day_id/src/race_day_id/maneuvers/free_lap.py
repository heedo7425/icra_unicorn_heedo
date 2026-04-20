"""Free-lap passive collection.

No actuation — the car is expected to be driven by the existing MPC/FTG stack.
Lap-end detection mirrors `controller/upenn_mpc/scripts/upenn_trainer.py:218-233`:
s wraps from >0.75·L to <0.15·L where L is /global_republisher/track_length.
Maneuver completes after `max_laps` wrap events.
"""
from __future__ import annotations

from .base_maneuver import BaseManeuver, ManeuverStatus


class FreeLapManeuver(BaseManeuver):
    name = "free_lap"

    def __init__(self, profile, rospy_mod):
        super().__init__(profile, rospy_mod)
        from nav_msgs.msg import Odometry

        self._max_laps = int(profile.get("max_laps", 1))
        self._min_vx = float(profile.get("min_vx", 1.0))
        self._lap_cooldown_s = 2.0

        self._track_length = float(rospy_mod.get_param(
            "/global_republisher/track_length", 0.0))
        self._prev_s = None
        self._laps = 0
        self._vx = 0.0
        self._last_lap_t = 0.0
        self._t0 = None

        rospy_mod.Subscriber("/car_state/odom_frenet", Odometry,
                             self._frenet_cb, queue_size=10)
        rospy_mod.Subscriber("/car_state/odom", Odometry,
                             self._odom_cb, queue_size=10)

    def _odom_cb(self, msg):
        self._vx = msg.twist.twist.linear.x

    def _frenet_cb(self, msg):
        if self._status != ManeuverStatus.RUNNING:
            return
        s = msg.pose.pose.position.x
        if self._track_length <= 0.1:
            self._track_length = float(self.rospy.get_param(
                "/global_republisher/track_length", 0.0))
        if (self._prev_s is not None and self._track_length > 0.1
                and self._vx >= self._min_vx):
            if (self._prev_s > 0.75 * self._track_length
                    and s < 0.15 * self._track_length):
                now = self.rospy.get_time()
                if now - self._last_lap_t > self._lap_cooldown_s:
                    self._last_lap_t = now
                    self._laps += 1
                    self.rospy.loginfo(
                        f"[free_lap] LAP END #{self._laps}/{self._max_laps} at s={s:.1f}")
                    if self._laps >= self._max_laps:
                        self.stop()
        self._prev_s = s

    def start(self):
        super().start()
        self._t0 = self.rospy.get_time()

    def step(self):
        if self._status != ManeuverStatus.RUNNING:
            return
        # Passive: no publish. Guard against forever-hang if frenet topic dead.
        if self._track_length <= 0.1 and self.rospy.get_time() - self._t0 > 10.0:
            self.stop("no_track_length")
