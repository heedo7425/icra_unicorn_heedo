#!/usr/bin/env python3
"""
Online μ estimator — peak-tracking RLS (physically more reactive).

핵심 아이디어:
  단순 ay/g 는 non-saturation 시 μ 과소추정. 하지만 **최근 윈도우 내
  peak ay** 는 tire 가 해당 구간에서 도달한 최대 횡가속 → μ·g 의 실제
  추정치로 더 적절.

Pipeline per /car_state/odom callback:
  ay_k = vx · ω                      # centripetal (body-frame ay proxy)
  peak_window 에 ay 넣고 가장 큰 값 추출
  gated update: peak 가 ay_threshold 이상이고 vx > 1 → Kalman update

Bounds: μ̂ ∈ [mu_min, mu_max].

Publishes /rls_mpc/mu_estimate, /rls_mpc/mu_sigma, /rls_mpc/rls_ay_peak.
"""

from __future__ import annotations

import math
from collections import deque

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


class MuEstimatorRLS:
    def __init__(self) -> None:
        rospy.init_node("mu_estimator_rls", anonymous=False)
        NS = "rls_mpc/rls"

        self.forgetting = float(rospy.get_param(f"{NS}/forgetting", 0.95))
        self.init_mu = float(rospy.get_param(f"{NS}/init_mu", 0.85))
        self.init_sigma = float(rospy.get_param(f"{NS}/init_sigma", 0.18))
        self.meas_sigma = float(rospy.get_param(f"{NS}/meas_sigma", 0.22))
        self.ay_threshold = float(rospy.get_param(f"{NS}/ay_threshold", 2.0))
        self.mu_min = float(rospy.get_param(f"{NS}/mu_min", 0.2))
        self.mu_max = float(rospy.get_param(f"{NS}/mu_max", 1.3))
        self.peak_window_sec = float(rospy.get_param(f"{NS}/peak_window_sec", 1.0))

        self.mu_hat = self.init_mu
        self.P = self.init_sigma ** 2
        self.R = self.meas_sigma ** 2
        self.g = 9.81

        # Peak ay over a short rolling window.
        max_samples = int(max(5, self.peak_window_sec * 100))
        self.ay_history: deque = deque(maxlen=max_samples)

        self.mu_pub = rospy.Publisher("/rls_mpc/mu_estimate", Float32, queue_size=1)
        self.sigma_pub = rospy.Publisher("/rls_mpc/mu_sigma", Float32, queue_size=1)
        self.peak_pub = rospy.Publisher("/rls_mpc/rls_ay_peak", Float32, queue_size=1)

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=1)

        rospy.loginfo(
            f"[mu_estimator_rls] peak-tracking μ̂=init{self.init_mu} "
            f"init_σ={self.init_sigma} meas_σ={self.meas_sigma} "
            f"ay_thres={self.ay_threshold} forget={self.forgetting} "
            f"peak_win={self.peak_window_sec}s clip=[{self.mu_min},{self.mu_max}]"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        vx = float(msg.twist.twist.linear.x)
        omega = float(msg.twist.twist.angular.z)
        ay = vx * omega  # body-frame centripetal proxy
        abs_ay = abs(ay)

        self.ay_history.append(abs_ay)
        peak_ay = max(self.ay_history) if self.ay_history else 0.0
        self.peak_pub.publish(Float32(data=float(peak_ay)))

        # Gate: peak within window reaches threshold + moving forward.
        if peak_ay >= self.ay_threshold and vx > 1.0:
            z = peak_ay / self.g
            z = max(self.mu_min, min(self.mu_max, z))
            K = self.P / (self.P + self.R)
            self.mu_hat = self.mu_hat + K * (z - self.mu_hat)
            self.mu_hat = max(self.mu_min, min(self.mu_max, self.mu_hat))
            self.P = (1 - K) * self.P

        # Forgetting: P inflates over time so estimator can track change.
        self.P = self.P / self.forgetting
        self.P = min(self.P, self.init_sigma ** 2)

        self.mu_pub.publish(Float32(data=float(self.mu_hat)))
        self.sigma_pub.publish(Float32(data=float(math.sqrt(max(self.P, 0.0)))))


if __name__ == "__main__":
    try:
        MuEstimatorRLS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
