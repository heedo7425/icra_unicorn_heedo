#!/usr/bin/env python3
"""
GP-MPC rviz HUD — ground-truth μ (패치 색) + GP correction 강도.

Publishes (10 Hz) to /mu_hud/markers:
  - SPHERE @ car: ground-truth μ 로 색칠 (패치 인지 확인)
  - SPHERE @ car + z: GP correction magnitude 로 크기·색. READY 전엔 작은 회색
  - TEXT over car: "μ_gt=X.XX  corr=YY%"
  - BADGE: correction > 70% 면 빨간 경고

Subscribes:
  /car_state/pose
  /mu_ground_truth           (패치 μ)
  /upenn_mpc/residual           (GP 잔차 — correction 계산용)
  /upenn_mpc/gp_ready           (GP 학습 완료 여부)
"""

from __future__ import annotations

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


# Must match upenn_mpc_srx1.yaml::gp.residual_clip
RESIDUAL_CLIP = (10.0, 5.0, 12.0)


def mu_to_color(mu: float):
    """Red → yellow → green over μ ∈ [0.3, 1.3]."""
    mu = max(0.3, min(1.3, mu))
    t = (mu - 0.3) / 1.0
    if t < 0.5:
        r, g, b = 1.0, t * 2.0, 0.0
    else:
        r, g, b = (1.0 - t) * 2.0, 1.0, 0.0
    return r, g, b


def corr_to_color(corr: float):
    """corr ∈ [0, 1]: gray → green → yellow → red."""
    if corr < 0.05:
        return 0.5, 0.5, 0.5
    if corr < 0.3:
        return 0.4, 1.0, 0.4
    if corr < 0.7:
        return 1.0, 0.85, 0.2
    return 1.0, 0.3, 0.3


class GpMuHud:
    def __init__(self) -> None:
        rospy.init_node("mu_hud", anonymous=False)
        self.frame_id = rospy.get_param("~frame_id", "map")

        self.mu_gt = 0.85
        self.pose_xy = (0.0, 0.0)
        self.residual = [0.0, 0.0, 0.0]
        self.gp_ready = False

        self.pub = rospy.Publisher("/mu_hud/markers", MarkerArray, queue_size=1)

        rospy.Subscriber("/car_state/pose", PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber("/mu_ground_truth", Float32, self._gt_cb, queue_size=1)
        rospy.Subscriber("/upenn_mpc/residual", Float32MultiArray, self._res_cb, queue_size=1)
        rospy.Subscriber("/upenn_mpc/gp_ready", Bool, self._ready_cb, queue_size=1)

        rospy.Timer(rospy.Duration(0.1), self._tick)
        rospy.loginfo("[mu_hud] ready (gt μ + GP correction viz)")

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.pose_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _gt_cb(self, msg: Float32) -> None:
        self.mu_gt = float(msg.data)

    def _res_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 3:
            self.residual = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]

    def _ready_cb(self, msg: Bool) -> None:
        self.gp_ready = bool(msg.data)

    def _tick(self, _evt) -> None:
        x, y = self.pose_xy
        ma = MarkerArray()

        # GT μ sphere (낮은 z)
        s_gt = Marker()
        s_gt.header.frame_id = self.frame_id
        s_gt.header.stamp = rospy.Time.now()
        s_gt.ns = "mu_hud_gt"
        s_gt.id = 0
        s_gt.type = Marker.SPHERE
        s_gt.action = Marker.ADD
        s_gt.pose.position.x = x
        s_gt.pose.position.y = y
        s_gt.pose.position.z = 0.35
        s_gt.pose.orientation.w = 1.0
        s_gt.scale.x = s_gt.scale.y = s_gt.scale.z = 0.35
        r, g, b = mu_to_color(self.mu_gt)
        s_gt.color.r = r; s_gt.color.g = g; s_gt.color.b = b; s_gt.color.a = 0.85
        ma.markers.append(s_gt)

        # GP correction sphere (scale·color 로 강도 표시)
        corr = 0.0
        if self.gp_ready:
            for i in range(3):
                c = abs(self.residual[i]) / RESIDUAL_CLIP[i]
                if c > corr:
                    corr = c
        s_gp = Marker()
        s_gp.header.frame_id = self.frame_id
        s_gp.header.stamp = rospy.Time.now()
        s_gp.ns = "mu_hud_gp_corr"
        s_gp.id = 0
        s_gp.type = Marker.SPHERE
        s_gp.action = Marker.ADD
        s_gp.pose.position.x = x
        s_gp.pose.position.y = y
        s_gp.pose.position.z = 0.95
        s_gp.pose.orientation.w = 1.0
        size = 0.25 + 0.55 * corr           # 0.25 (no corr) → 0.80 (100%)
        s_gp.scale.x = s_gp.scale.y = s_gp.scale.z = size
        cr, cg, cb = corr_to_color(corr)
        s_gp.color.r = cr; s_gp.color.g = cg; s_gp.color.b = cb
        s_gp.color.a = 0.4 + 0.5 * corr     # corr 높을수록 불투명
        ma.markers.append(s_gp)

        # Text: "μ_gt=X.XX  corr=YY%"
        txt = Marker()
        txt.header.frame_id = self.frame_id
        txt.header.stamp = rospy.Time.now()
        txt.ns = "mu_hud_text"
        txt.id = 0
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = x
        txt.pose.position.y = y
        txt.pose.position.z = 1.7
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.45
        txt.color.r = 1.0; txt.color.g = 1.0; txt.color.b = 1.0; txt.color.a = 1.0
        status = "READY" if self.gp_ready else "COLD"
        txt.text = (
            f"μ_gt={self.mu_gt:.2f}\n"
            f"GP {status}  corr={corr*100:.0f}%"
        )
        ma.markers.append(txt)

        # BADGE: correction 강함 → 빨간 경고 (패치 강도 확인)
        if corr > 0.70:
            badge = Marker()
            badge.header.frame_id = self.frame_id
            badge.header.stamp = rospy.Time.now()
            badge.ns = "mu_hud_badge"
            badge.id = 0
            badge.type = Marker.TEXT_VIEW_FACING
            badge.action = Marker.ADD
            badge.pose.position.x = x
            badge.pose.position.y = y
            badge.pose.position.z = 2.4
            badge.pose.orientation.w = 1.0
            badge.scale.z = 0.45
            badge.color.r = 1.0; badge.color.g = 0.2; badge.color.b = 0.2; badge.color.a = 1.0
            badge.text = "!! GP strong correction !!"
            ma.markers.append(badge)

        self.pub.publish(ma)


if __name__ == "__main__":
    try:
        GpMuHud()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
