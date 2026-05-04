#!/usr/bin/env python3
"""mu_patch_node — sim 의 friction_coeff 를 sector-별로 override.

차량 s 위치 → /friction_map_params/Sector*/friction 룩업 → /sim_friction_coeff
(Float32) @ 50Hz publish. 패치된 f110-simulator 가 받아 매 step 의 타이어 모델
friction 갱신. 알고리즘 검증 환경 일관성용 (sim grip 균일 1.0 → sector-별 변동).
"""
from __future__ import annotations

from typing import List, Optional

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Int32


class MuPatchNode:
    def __init__(self):
        rospy.init_node("mu_patch_node")
        self.default_mu = float(rospy.get_param("~default_mu", 1.0))
        self.rate_hz = float(rospy.get_param("~publish_rate", 50.0))
        self.sectors: List[dict] = []
        self.global_limit: float = 1.0
        self.s_now: Optional[float] = None

        self.pub = rospy.Publisher("/sim_friction_coeff", Float32, queue_size=1)
        self.sec_pub = rospy.Publisher("/mu_patch/active_sector", Int32, queue_size=1)

        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._frenet_cb, queue_size=5)
        rospy.Subscriber("/mu_ppc/sectors_loaded", Bool,
                         lambda _: self._reload(), queue_size=1)
        self._reload()
        rospy.loginfo("[mu_patch_node] default_mu=%.2f rate=%.0fHz",
                      self.default_mu, self.rate_hz)

    def _frenet_cb(self, msg: Odometry) -> None:
        self.s_now = float(msg.pose.pose.position.x)

    def _reload(self) -> None:
        try:
            n = int(rospy.get_param("/friction_map_params/n_sectors", 0))
        except Exception:
            n = 0
        secs = []
        for i in range(n):
            secs.append({
                "s_start": float(rospy.get_param(f"/friction_map_params/Sector{i}/s_start", -1.0)),
                "s_end": float(rospy.get_param(f"/friction_map_params/Sector{i}/s_end", -1.0)),
                "friction": float(rospy.get_param(f"/friction_map_params/Sector{i}/friction", 1.0)),
            })
        self.sectors = secs
        self.global_limit = float(
            rospy.get_param("/friction_map_params/global_friction_limit", 1.5))
        if secs:
            rospy.loginfo("[mu_patch_node] %d sectors loaded", len(secs))

    def _lookup(self):
        if self.s_now is None or not self.sectors:
            return self.default_mu, -1
        for i, sec in enumerate(self.sectors):
            if sec["s_start"] >= 0 and sec["s_start"] <= self.s_now <= sec["s_end"]:
                return min(sec["friction"], self.global_limit), i
        return self.default_mu, -1

    def spin(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        last = -2
        while not rospy.is_shutdown():
            mu, idx = self._lookup()
            self.pub.publish(Float32(data=float(mu)))
            self.sec_pub.publish(Int32(data=int(idx)))
            if idx != last:
                rospy.loginfo("[mu_patch_node] s=%.1f → sector %d, mu=%.2f",
                              self.s_now if self.s_now else -1, idx, mu)
                last = idx
            rate.sleep()


if __name__ == "__main__":
    MuPatchNode().spin()
