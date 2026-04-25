#!/usr/bin/env python3
"""Minimal pure-pursuit controller for Isaac-sim smoke testing.

Bypasses race_stack state_machine/sector-tuner/planner infrastructure so we can
check whether the Isaac ↔ rosbridge ↔ vehicle pipeline is healthy independent
of the full MPC stack.

- Reads `/car_state/odom`       (pose)
- Reads `/global_waypoints`     (raceline)
- Publishes AckermannDriveStamped on `/vesc/high_level/ackermann_cmd_mux/input/nav_1`

Constant speed (configurable) and geometric pure-pursuit steering to a
look-ahead point on the raceline (interpolated from xy closest then +L).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rospy
import tf.transformations as tft
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray


class SimplePP:
    def __init__(self):
        rospy.init_node("simple_pp")
        self.lookahead = float(rospy.get_param("~lookahead", 0.6))           # m
        self.speed     = float(rospy.get_param("~speed", 2.0))               # m/s
        self.wheelbase = float(rospy.get_param("~wheelbase", 0.307))         # m
        self.max_steer = float(rospy.get_param("~max_steer", 0.4))           # rad
        self.drive_topic = rospy.get_param("~drive_topic",
                                            "/vesc/high_level/ackermann_cmd_mux/input/nav_1")

        self.wpnts_xy: Optional[np.ndarray] = None   # shape (N, 2)
        self.odom: Optional[Odometry] = None

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=5)
        rospy.Subscriber("/global_waypoints", WpntArray, self._gw_cb, queue_size=1)
        self.pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=5)

        rospy.loginfo(
            "[simple_pp] L=%.2f v=%.1f wb=%.3f → %s",
            self.lookahead, self.speed, self.wheelbase, self.drive_topic,
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom = msg

    def _gw_cb(self, msg: WpntArray) -> None:
        if len(msg.wpnts) < 2:
            return
        self.wpnts_xy = np.array([[w.x_m, w.y_m] for w in msg.wpnts], dtype=np.float64)
        # ### HJ : raceline 의 vx_mps 캐시 → constant speed 가 아닌 velocity profile 추종.
        self.wpnts_vx = np.array([w.vx_mps for w in msg.wpnts], dtype=np.float64)
        rospy.loginfo_once(
            f"[simple_pp] raceline cached: {len(self.wpnts_xy)} wpnts, "
            f"vx∈[{self.wpnts_vx.min():.2f},{self.wpnts_vx.max():.2f}]"
        )

    def step(self) -> None:
        if self.odom is None or self.wpnts_xy is None:
            return
        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        q  = self.odom.pose.pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        dxy = self.wpnts_xy - np.array([px, py])
        dists = np.hypot(dxy[:, 0], dxy[:, 1])
        nearest = int(np.argmin(dists))

        # walk forward along wpnts from nearest, accumulating arc length, until
        # we exceed lookahead
        N = len(self.wpnts_xy)
        target_idx = nearest
        acc = 0.0
        for k in range(1, N):
            i = (nearest + k) % N
            j = (nearest + k - 1) % N
            seg = float(np.hypot(*(self.wpnts_xy[i] - self.wpnts_xy[j])))
            acc += seg
            if acc >= self.lookahead:
                target_idx = i
                break

        tx, ty = self.wpnts_xy[target_idx]
        # transform target into body frame
        dx = tx - px
        dy = ty - py
        xb =  math.cos(-yaw) * dx - math.sin(-yaw) * dy
        yb =  math.sin(-yaw) * dx + math.cos(-yaw) * dy
        if xb < 1e-3:
            # target behind / at same x — avoid /0, just steer hard toward sign(yb)
            delta = math.copysign(self.max_steer, yb)
        else:
            # geometric pure pursuit
            L = math.hypot(xb, yb)
            delta = math.atan2(2.0 * self.wheelbase * yb, L * L)
        delta = max(-self.max_steer, min(self.max_steer, delta))

        # ### HJ : raceline vx_mps 사용 (현재 nearest waypoint 의 속도).
        # ~speed 가 0 이면 velocity profile 추종, >0 이면 상수속도 (legacy).
        if self.speed > 0.0:
            cmd_speed = self.speed
        elif getattr(self, "wpnts_vx", None) is not None:
            cmd_speed = float(self.wpnts_vx[nearest])
        else:
            cmd_speed = 0.0
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = cmd_speed
        msg.drive.steering_angle = delta
        self.pub.publish(msg)

    def spin(self, hz: float = 50.0) -> None:
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()


if __name__ == "__main__":
    SimplePP().spin()
