#!/usr/bin/env python3
"""
μ applier — 2D sim 에서 '가상 마찰' 효과.

Design:
  MPC belief μ (mpc_used) 와 ground-truth μ 의 mismatch 에 따라 cmd 를
  변형해 sim 차량 거동 변화를 만든다.

  ratio = clamp(mu_gt / reference_mu, ratio_min, ratio_max)

  1) 기본 스케일: speed *= ratio, steer *= sqrt(ratio)
     → μ 낮을 땐 속도·조향 감소 (제한적 grip 효과)

  2) Slip 효과: mismatch = mu_used - mu_gt  (MPC belief - true)
     mismatch > 0 이면 MPC 가 실제보다 고마찰 가정 → 실제로는 grip 부족
       → steer 에 랜덤 노이즈 (진동) + 추가 속도 감쇄
     mismatch ≤ 0 이면 MPC 가 보수적 → 영향 없음

Subscribes:
  /upenn_mpc/cmd_raw                — MPC 원 명령
  /mu_ground_truth               — 패치 lookup μ
  /upenn_mpc/mu_used                — MPC 가 실제 OCP 에 주입한 μ

Publishes:
  /vesc/high_level/ackermann_cmd_mux/input/nav_1
  /upenn_mpc/cmd_scaled_debug      — Float32 ratio
  /upenn_mpc/slip_indicator        — Float32 slip 강도 (0=없음, 1=심각)
"""

from __future__ import annotations

import random

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32


class MuApplier:
    def __init__(self) -> None:
        rospy.init_node("mu_applier", anonymous=False)
        self.reference_mu = float(rospy.get_param("~reference_mu", 0.85))
        # Softened defaults: 극단 스케일링이 MPC planning 과 어긋나 차가 코너에서
        # 멈추는 현상을 줄이기 위해 ratio_min 상향 + slip 효과 약화.
        self.ratio_min = float(rospy.get_param("~ratio_min", 0.5))
        self.ratio_max = float(rospy.get_param("~ratio_max", 1.3))
        self.enable = bool(rospy.get_param("~enable", True))
        self.slip_gain = float(rospy.get_param("~slip_gain", 0.10))
        self.slip_speed_penalty = float(rospy.get_param("~slip_speed_penalty", 0.25))

        self.mu_gt = self.reference_mu
        self.mu_used = self.reference_mu

        self.pub = rospy.Publisher(
            rospy.get_param("~out_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_1"),
            AckermannDriveStamped, queue_size=10,
        )
        self.ratio_pub = rospy.Publisher("/upenn_mpc/cmd_scaled_debug", Float32, queue_size=1)
        self.slip_pub = rospy.Publisher("/upenn_mpc/slip_indicator", Float32, queue_size=1)

        rospy.Subscriber(
            rospy.get_param("~in_topic", "/upenn_mpc/cmd_raw"),
            AckermannDriveStamped, self._cmd_cb, queue_size=10,
        )
        rospy.Subscriber("/mu_ground_truth", Float32, self._gt_cb, queue_size=1)
        rospy.Subscriber("/upenn_mpc/mu_used", Float32, self._used_cb, queue_size=1)

        rospy.loginfo(
            f"[mu_applier] ref_mu={self.reference_mu} ratio=[{self.ratio_min},{self.ratio_max}] "
            f"slip_gain={self.slip_gain} enable={self.enable}"
        )

    def _gt_cb(self, msg: Float32) -> None:
        self.mu_gt = float(msg.data)

    def _used_cb(self, msg: Float32) -> None:
        self.mu_used = float(msg.data)

    def _cmd_cb(self, msg: AckermannDriveStamped) -> None:
        out = AckermannDriveStamped()
        out.header = msg.header
        if not self.enable:
            out.drive = msg.drive
            self.pub.publish(out)
            return

        # Base ratio from ground-truth μ.
        ratio = max(self.ratio_min, min(self.ratio_max, self.mu_gt / self.reference_mu))
        speed = msg.drive.speed * ratio
        steer = msg.drive.steering_angle * (ratio ** 0.5)

        # Slip injection when MPC belief > ground truth.
        mismatch = self.mu_used - self.mu_gt
        slip = max(0.0, mismatch)  # only when over-confident
        if slip > 0.05:
            # Steer chatter proportional to mismatch.
            steer_noise = random.gauss(0.0, self.slip_gain * slip)
            steer = steer + steer_noise
            # Extra speed penalty (tire can't transmit torque).
            speed = speed * (1.0 - self.slip_speed_penalty * min(slip, 1.0))

        out.drive.speed = speed
        out.drive.acceleration = msg.drive.acceleration * ratio
        out.drive.jerk = msg.drive.jerk
        out.drive.steering_angle = steer
        out.drive.steering_angle_velocity = msg.drive.steering_angle_velocity * (ratio ** 0.5)
        self.pub.publish(out)

        self.ratio_pub.publish(Float32(data=float(ratio)))
        self.slip_pub.publish(Float32(data=float(slip)))


if __name__ == "__main__":
    try:
        MuApplier()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
