#!/usr/bin/env python3
"""
μ patch publisher (raceline-s based).

Patches are defined in Frenet (s_start, s_end, d_min, d_max). This node
subscribes to /global_waypoints, walks the wpnts in each patch's s-range,
and builds:
  (1) TRIANGLE_LIST marker for rviz showing the patch as a colored band
      along the raceline (width = d_max-d_min per wpnt).
  (2) /mu_ground_truth (Float32) per /car_state/odom_frenet callback.
      If car's current (s, d) falls inside any patch range → use patch μ.
      Else → default_mu.

When launched with source=ground_truth, a topic_tools relay forwards
/mu_ground_truth → /ekf_mpc/mu_estimate (done in ekf_mpc_sim.launch).
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np
import rospy
import yaml
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray


class MuPatchPublisher:
    def __init__(self) -> None:
        rospy.init_node("mu_patch_publisher", anonymous=False)
        self.patches_file = rospy.get_param("~patches_file", "")
        self.default_mu = float(rospy.get_param("~default_mu", 0.85))
        self.frame_id = rospy.get_param("~frame_id", "map")

        self.patches: List[dict] = []
        if self.patches_file and os.path.exists(self.patches_file):
            with open(self.patches_file) as f:
                data = yaml.safe_load(f) or {}
            self.patches = list(data.get("patches", []))
            rospy.loginfo(
                f"[mu_patch_publisher] loaded {len(self.patches)} patches from {self.patches_file}"
            )
        else:
            rospy.logwarn(
                f"[mu_patch_publisher] no patches file ('{self.patches_file}'); "
                f"uniform mu={self.default_mu}"
            )

        self.gb_wpnts: Optional[np.ndarray] = None  # (M, 4): s, x, y, psi
        self.track_length: float = 0.0

        self.mrk_pub = rospy.Publisher(
            "/mu_patches/markers", MarkerArray, queue_size=1, latch=True
        )
        self.mu_pub = rospy.Publisher("/mu_ground_truth", Float32, queue_size=1)

        rospy.Subscriber("/global_waypoints", WpntArray, self._gb_cb, queue_size=1)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self._gb_cb, queue_size=1)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._frenet_cb, queue_size=1)

        rospy.loginfo("[mu_patch_publisher] waiting for /global_waypoints...")

    def _gb_cb(self, msg: WpntArray) -> None:
        if self.gb_wpnts is not None:
            return
        arr = np.array([[w.s_m, w.x_m, w.y_m, w.psi_rad] for w in msg.wpnts], dtype=np.float64)
        if arr.shape[0] < 2:
            return
        self.gb_wpnts = arr
        self.track_length = float(arr[-1, 0])
        rospy.loginfo(
            f"[mu_patch_publisher] raceline cached: {arr.shape[0]} wpnts, L={self.track_length:.2f}m"
        )
        self._publish_markers()

    def _frenet_cb(self, msg: Odometry) -> None:
        if self.gb_wpnts is None:
            return
        s = float(msg.pose.pose.position.x)
        d = float(msg.pose.pose.position.y)
        mu = self._lookup_mu_frenet(s, d)
        self.mu_pub.publish(Float32(data=mu))

    def _lookup_mu_frenet(self, s: float, d: float) -> float:
        for p in self.patches:
            s_start = float(p.get("s_start", 0.0))
            s_end = float(p.get("s_end", 0.0))
            d_min = float(p.get("d_min", -10.0))
            d_max = float(p.get("d_max", +10.0))
            if self._s_in_range(s, s_start, s_end) and (d_min <= d <= d_max):
                return float(p["mu"])
        return self.default_mu

    @staticmethod
    def _s_in_range(s: float, s_start: float, s_end: float) -> bool:
        if s_start <= s_end:
            return s_start <= s <= s_end
        return (s >= s_start) or (s <= s_end)

    def _publish_markers(self) -> None:
        ma = MarkerArray()
        gb = self.gb_wpnts
        for i, p in enumerate(self.patches):
            s_start = float(p["s_start"])
            s_end = float(p["s_end"])
            d_min = float(p.get("d_min", -0.5))
            d_max = float(p.get("d_max", +0.5))
            rgba = p.get("color", [0.5, 0.5, 0.5, 0.45])

            if s_start <= s_end:
                mask = (gb[:, 0] >= s_start) & (gb[:, 0] <= s_end)
            else:
                mask = (gb[:, 0] >= s_start) | (gb[:, 0] <= s_end)
            seg = gb[mask]
            if seg.shape[0] < 2:
                continue

            tri = Marker()
            tri.header.frame_id = self.frame_id
            tri.header.stamp = rospy.Time.now()
            tri.ns = "mu_patches"
            tri.id = i
            tri.type = Marker.TRIANGLE_LIST
            tri.action = Marker.ADD
            tri.scale.x = 1.0
            tri.scale.y = 1.0
            tri.scale.z = 1.0
            tri.pose.orientation.w = 1.0
            tri.color.r = float(rgba[0])
            tri.color.g = float(rgba[1])
            tri.color.b = float(rgba[2])
            tri.color.a = float(rgba[3]) if len(rgba) > 3 else 0.45
            for k in range(seg.shape[0] - 1):
                s_k, x_k, y_k, psi_k = seg[k]
                s_k1, x_k1, y_k1, psi_k1 = seg[k + 1]
                nx_k, ny_k = -math.sin(psi_k), math.cos(psi_k)
                nx_k1, ny_k1 = -math.sin(psi_k1), math.cos(psi_k1)
                p1 = Point(x=x_k + d_min * nx_k, y=y_k + d_min * ny_k, z=-0.005)
                p2 = Point(x=x_k + d_max * nx_k, y=y_k + d_max * ny_k, z=-0.005)
                p3 = Point(x=x_k1 + d_min * nx_k1, y=y_k1 + d_min * ny_k1, z=-0.005)
                p4 = Point(x=x_k1 + d_max * nx_k1, y=y_k1 + d_max * ny_k1, z=-0.005)
                tri.points.extend([p1, p2, p3, p3, p2, p4])
            ma.markers.append(tri)

            mid = seg[seg.shape[0] // 2]
            txt = Marker()
            txt.header.frame_id = self.frame_id
            txt.header.stamp = rospy.Time.now()
            txt.ns = "mu_patches_label"
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(mid[1])
            txt.pose.position.y = float(mid[2])
            txt.pose.position.z = 0.4
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.35
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 1.0
            txt.text = f"{p.get('name','?')}\n s={s_start:.1f}-{s_end:.1f} mu={p['mu']:.2f}"
            ma.markers.append(txt)
        self.mrk_pub.publish(ma)


if __name__ == "__main__":
    try:
        MuPatchPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
