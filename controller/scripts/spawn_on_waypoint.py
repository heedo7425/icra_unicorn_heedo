#!/usr/bin/env python3
"""Spawn the racecar at the first global waypoint (sim only) and optionally
supervise for stuck conditions → re-teleport.

Referenced by upenn_mpc_sim.launch (and rls/ekf variants). Publishes a
PoseWithCovarianceStamped on /initialpose — f1tenth_simulator subscribes to
this and teleports the car.

Params (~):
    supervise   : bool  (default True)  — if True, monitor /car_state/odom and
                                           re-teleport when vx < stuck_vx_min
                                           for stuck_sec seconds
    stuck_sec   : float (default 1.5)   — seconds of near-zero vx before reset
    stuck_vx_min: float (default 0.1)   — m/s threshold
    wpnt_idx    : int   (default 0)     — which waypoint to spawn on
    tangent_idx : int   (default 5)     — waypoint used to compute tangent yaw
"""
from __future__ import annotations

import math
from typing import Optional

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray


class SpawnOnWaypoint:
    def __init__(self):
        rospy.init_node("spawn_on_waypoint")
        self.supervise    = bool(rospy.get_param("~supervise",    True))
        self.stuck_sec    = float(rospy.get_param("~stuck_sec",   1.5))
        self.stuck_vx_min = float(rospy.get_param("~stuck_vx_min", 0.1))
        self.wpnt_idx     = int(rospy.get_param("~wpnt_idx",      0))
        self.tangent_idx  = int(rospy.get_param("~tangent_idx",   5))

        self.pose_pub = rospy.Publisher("/initialpose",
                                        PoseWithCovarianceStamped, queue_size=1, latch=True)

        self.wpnts = None
        self.last_ok_time = rospy.Time.now()
        self.spawned = False

        rospy.Subscriber("/global_waypoints", WpntArray, self._gw_cb, queue_size=1)
        if self.supervise:
            rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=5)

    def _gw_cb(self, msg: WpntArray) -> None:
        if len(msg.wpnts) <= max(self.wpnt_idx, self.tangent_idx):
            return
        self.wpnts = msg.wpnts
        if not self.spawned:
            self._spawn()
            self.spawned = True

    def _odom_cb(self, msg: Odometry) -> None:
        if not self.spawned or self.wpnts is None:
            return
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.hypot(vx, vy)
        if speed >= self.stuck_vx_min:
            self.last_ok_time = rospy.Time.now()
            return
        if (rospy.Time.now() - self.last_ok_time).to_sec() > self.stuck_sec:
            rospy.logwarn(
                f"[spawn_on_waypoint] stuck for {self.stuck_sec:.1f}s "
                f"(speed={speed:.2f}) — re-teleporting"
            )
            self._spawn()
            self.last_ok_time = rospy.Time.now()

    def _spawn(self) -> None:
        if self.wpnts is None:
            return
        w0 = self.wpnts[self.wpnt_idx]
        wt = self.wpnts[self.tangent_idx % len(self.wpnts)]
        yaw = math.atan2(wt.y_m - w0.y_m, wt.x_m - w0.x_m)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = w0.x_m
        msg.pose.pose.position.y = w0.y_m
        msg.pose.pose.position.z = getattr(w0, "z_m", 0.0)
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_pub.publish(msg)
        rospy.loginfo(
            f"[spawn_on_waypoint] spawned at wpnt[{self.wpnt_idx}] "
            f"({w0.x_m:.2f},{w0.y_m:.2f}) yaw={yaw:.3f}"
        )


if __name__ == "__main__":
    SpawnOnWaypoint()
    rospy.spin()
