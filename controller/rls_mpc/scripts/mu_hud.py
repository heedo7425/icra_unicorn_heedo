#!/usr/bin/env python3
"""
μ HUD — rviz 직관 viz: ground truth vs estimated μ 비교.

Publishes (10 Hz):
  /mu_hud/markers (MarkerArray):
    - SPHERE @ car xy, color = estimated μ (rls_mpc/mu_used)
    - SPHERE @ car xy + z offset, color = ground truth μ (mu_ground_truth)
    - TEXT "gt=X.XX | est=Y.YY" over car
    - TEXT "Δ=|gt-est|" in red if |diff|>0.1

Subscribes:
  /car_state/pose
  /rls_mpc/mu_used      (MPC 가 실제 OCP 에 주입한 μ — estimator 출력)
  /mu_ground_truth     (패치 lookup)
"""

from __future__ import annotations

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray


def mu_to_color(mu: float):
    """Red → yellow → green over μ ∈ [0.3, 1.0]."""
    mu = max(0.3, min(1.0, mu))
    t = (mu - 0.3) / 0.7
    if t < 0.5:
        r, g, b = 1.0, t * 2.0, 0.0
    else:
        r, g, b = (1.0 - t) * 2.0, 1.0, 0.0
    return r, g, b


class MuHud:
    def __init__(self) -> None:
        rospy.init_node("mu_hud", anonymous=False)
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.mu_est = 0.85
        self.mu_gt = 0.85
        self.pose_xy = (0.0, 0.0)

        self.pub = rospy.Publisher("/mu_hud/markers", MarkerArray, queue_size=1)

        rospy.Subscriber("/car_state/pose", PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber("/rls_mpc/mu_used", Float32, self._est_cb, queue_size=1)
        rospy.Subscriber("/mu_ground_truth", Float32, self._gt_cb, queue_size=1)

        rospy.Timer(rospy.Duration(0.1), self._tick)
        rospy.loginfo("[mu_hud] ready (gt + estimated μ 비교)")

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.pose_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _est_cb(self, msg: Float32) -> None:
        self.mu_est = float(msg.data)

    def _gt_cb(self, msg: Float32) -> None:
        self.mu_gt = float(msg.data)

    def _tick(self, _evt) -> None:
        x, y = self.pose_xy
        ma = MarkerArray()

        # GT sphere (아래쪽)
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

        # EST sphere (위쪽, 테두리 효과)
        s_est = Marker()
        s_est.header.frame_id = self.frame_id
        s_est.header.stamp = rospy.Time.now()
        s_est.ns = "mu_hud_est"
        s_est.id = 0
        s_est.type = Marker.SPHERE
        s_est.action = Marker.ADD
        s_est.pose.position.x = x
        s_est.pose.position.y = y
        s_est.pose.position.z = 0.85
        s_est.pose.orientation.w = 1.0
        s_est.scale.x = s_est.scale.y = s_est.scale.z = 0.30
        r, g, b = mu_to_color(self.mu_est)
        s_est.color.r = r; s_est.color.g = g; s_est.color.b = b; s_est.color.a = 0.85
        ma.markers.append(s_est)

        # Text: gt/est 비교
        txt = Marker()
        txt.header.frame_id = self.frame_id
        txt.header.stamp = rospy.Time.now()
        txt.ns = "mu_hud_text"
        txt.id = 0
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = x
        txt.pose.position.y = y
        txt.pose.position.z = 1.4
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.50
        txt.color.r = 1.0; txt.color.g = 1.0; txt.color.b = 1.0; txt.color.a = 1.0
        diff = abs(self.mu_est - self.mu_gt)
        txt.text = (
            f"gt={self.mu_gt:.2f}  est={self.mu_est:.2f}\n"
            f"Δ={diff:+.2f}"
        )
        ma.markers.append(txt)

        # Error badge (큰 차이 시 빨간 경고)
        if diff > 0.10:
            badge = Marker()
            badge.header.frame_id = self.frame_id
            badge.header.stamp = rospy.Time.now()
            badge.ns = "mu_hud_badge"
            badge.id = 0
            badge.type = Marker.TEXT_VIEW_FACING
            badge.action = Marker.ADD
            badge.pose.position.x = x
            badge.pose.position.y = y
            badge.pose.position.z = 2.1
            badge.pose.orientation.w = 1.0
            badge.scale.z = 0.45
            badge.color.r = 1.0; badge.color.g = 0.2; badge.color.b = 0.2; badge.color.a = 1.0
            badge.text = "!! mu mismatch !!"
            ma.markers.append(badge)

        self.pub.publish(ma)


if __name__ == "__main__":
    try:
        MuHud()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
