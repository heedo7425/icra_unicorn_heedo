#!/usr/bin/env python3
"""
Online μ estimator — Pacejka-based EKF + longitudinal slip + s-memory prior.

Level 2 확장 (단순 Pacejka-EKF 대비):
  (a) Longitudinal channel — |u_ax| 이 |a_x_meas| 보다 크면 saturation 간주,
      z_long = |a_x_meas|/g 를 두 번째 measurement 로 사용.
  (b) s-memory prior — 2m 간격 s_bin 에 μ̂ 저장 (EMA). 업데이트 없는 직선
      구간에선 이 memory 를 prior target 으로 써서 이전 lap 값 복원.

Pipeline per /car_state/odom callback:
  1. Lateral EKF step (기존 Pacejka): h_y(μ) = μ·K_y
  2. Longitudinal update if saturation detected
  3. If neither update fires (직선), s-memory 로 soft pull
  4. K>min_K 로 실질 업데이트 발생 시 현 s_bin 에 기록

Publishes /ekf_mpc/mu_estimate, /ekf_mpc/mu_sigma,
         /ekf_mpc/ekf_K, /ekf_mpc/ekf_innov, /ekf_mpc/ekf_long_active.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


def pacejka_unit(alpha: float, B: float, C: float, D: float, E: float) -> float:
    x = B * alpha
    phi = x - E * (x - math.atan(x))
    return D * math.sin(C * math.atan(phi))


class MuEstimatorEKF:
    def __init__(self) -> None:
        rospy.init_node("mu_estimator_ekf", anonymous=False)
        NS = "ekf_mpc/ekf"
        gp = rospy.get_param

        # ---- Vehicle / tire params ----
        self.m   = float(gp("/vehicle/m",    3.54))
        self.l_f = float(gp("/vehicle/l_f",  0.162))
        self.l_r = float(gp("/vehicle/l_r",  0.145))
        self.l_wb= float(gp("/vehicle/l_wb", self.l_f + self.l_r))
        self.Bf  = float(gp("/tire_front/B", 4.80))
        self.Cf  = float(gp("/tire_front/C", 2.16))
        self.Df  = float(gp("/tire_front/D", 0.65))
        self.Ef  = float(gp("/tire_front/E", 0.37))
        self.Br  = float(gp("/tire_rear/B",  20.0))
        self.Cr  = float(gp("/tire_rear/C",  1.50))
        self.Dr  = float(gp("/tire_rear/D",  0.62))
        self.Er  = float(gp("/tire_rear/E",  0.0))
        self.g   = 9.81

        # ---- EKF tuning ----
        self.init_mu    = float(gp(f"{NS}/init_mu",    0.85))
        self.init_sigma = float(gp(f"{NS}/init_sigma", 0.30))
        self.proc_sigma = float(gp(f"{NS}/proc_sigma", 0.04))
        self.meas_sigma = float(gp(f"{NS}/meas_sigma", 0.55))
        self.min_speed  = float(gp(f"{NS}/min_speed",  1.0))
        self.min_K      = float(gp(f"{NS}/min_K",      0.8))
        self.mu_min     = float(gp(f"{NS}/mu_min",     0.2))
        self.mu_max     = float(gp(f"{NS}/mu_max",     1.3))
        self.vy_smooth  = float(gp(f"{NS}/vy_smooth_alpha", 0.35))
        self.prior_pull = float(gp(f"{NS}/prior_pull_rate", 0.005))
        # --- Level 2 long channel ---
        self.long_enable     = bool(gp(f"{NS}/long_enable", True))
        self.long_ax_thresh  = float(gp(f"{NS}/long_ax_thresh",  2.0))   # |u_ax| 최소
        self.long_slip_thresh= float(gp(f"{NS}/long_slip_thresh", 0.8))  # |u_ax|-|ax_meas| 이상
        self.long_meas_sigma = float(gp(f"{NS}/long_meas_sigma", 0.45))
        # --- Level 2 s-memory ---
        self.mem_enable    = bool(gp(f"{NS}/mem_enable", True))
        self.mem_bin_width = float(gp(f"{NS}/mem_bin_width", 2.0))     # meters
        self.mem_ema_alpha = float(gp(f"{NS}/mem_ema_alpha", 0.2))
        self.mem_pull_rate = float(gp(f"{NS}/mem_pull_rate", 0.03))    # K-skip 시 memory 로 pull

        self.mu_hat = self.init_mu
        self.P      = self.init_sigma ** 2
        self.Q      = self.proc_sigma ** 2
        self.R_lat  = self.meas_sigma ** 2
        self.R_long = self.long_meas_sigma ** 2

        self.prev_vy: Optional[float] = None
        self.prev_vx: Optional[float] = None
        self.prev_t:  Optional[float] = None
        self.dvy_dt = 0.0
        self.dvx_dt = 0.0
        self.delta = 0.0
        self.u_ax  = 0.0
        self.car_s: Optional[float] = None

        self.s_memory: Dict[int, float] = {}

        # ---- Pubs / Subs ----
        self.mu_pub    = rospy.Publisher("/ekf_mpc/mu_estimate", Float32, queue_size=1)
        self.sigma_pub = rospy.Publisher("/ekf_mpc/mu_sigma",    Float32, queue_size=1)
        self.K_pub     = rospy.Publisher("/ekf_mpc/ekf_K",       Float32, queue_size=1)
        self.innov_pub = rospy.Publisher("/ekf_mpc/ekf_innov",   Float32, queue_size=1)
        self.long_pub  = rospy.Publisher("/ekf_mpc/ekf_long_active", Float32, queue_size=1)

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._frenet_cb, queue_size=1)
        rospy.Subscriber("/ekf_mpc/cmd_raw", AckermannDriveStamped, self._cmd_cb, queue_size=1)

        rospy.loginfo(
            f"[mu_estimator_ekf] L2 start — Q={self.proc_sigma}² R_lat={self.meas_sigma}² "
            f"R_long={self.long_meas_sigma}² min_K={self.min_K} "
            f"long={self.long_enable} mem={self.mem_enable} bin={self.mem_bin_width}m"
        )

    def _cmd_cb(self, msg: AckermannDriveStamped) -> None:
        self.delta = float(msg.drive.steering_angle)
        self.u_ax = float(msg.drive.acceleration)

    def _frenet_cb(self, msg: Odometry) -> None:
        self.car_s = float(msg.pose.pose.position.x)

    # ---- s-memory helpers ----
    def _s_bin(self, s: float) -> int:
        return int(s // self.mem_bin_width)

    def _memory_update(self) -> None:
        if not self.mem_enable or self.car_s is None:
            return
        sb = self._s_bin(self.car_s)
        prev = self.s_memory.get(sb)
        if prev is None:
            self.s_memory[sb] = self.mu_hat
        else:
            a = self.mem_ema_alpha
            self.s_memory[sb] = (1 - a) * prev + a * self.mu_hat

    def _memory_lookup(self) -> Optional[float]:
        if not self.mem_enable or self.car_s is None:
            return None
        return self.s_memory.get(self._s_bin(self.car_s))

    def _kalman_update(self, H: float, z: float, R: float) -> float:
        """Scalar EKF update. Returns innovation."""
        h_pred = self.mu_hat * H  # both channels have form h(μ) = μ·H_i
        innovation = z - h_pred
        S = H * self.P * H + R
        if abs(S) < 1e-9:
            return innovation
        gain = self.P * H / S
        self.mu_hat += gain * innovation
        self.mu_hat = max(self.mu_min, min(self.mu_max, self.mu_hat))
        self.P = (1.0 - gain * H) * self.P
        self.P = max(self.P, 1e-6)
        return innovation

    def _odom_cb(self, msg: Odometry) -> None:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        omega = float(msg.twist.twist.angular.z)
        t = msg.header.stamp.to_sec()

        # Finite-diff LPF for dvy/dt, dvx/dt.
        dvy_raw = 0.0
        dvx_raw = 0.0
        if self.prev_t is not None and t > self.prev_t:
            dt = t - self.prev_t
            if 1e-4 < dt < 0.1:
                if self.prev_vy is not None:
                    dvy_raw = (vy - self.prev_vy) / dt
                if self.prev_vx is not None:
                    dvx_raw = (vx - self.prev_vx) / dt
        self.prev_vy, self.prev_vx, self.prev_t = vy, vx, t
        a = self.vy_smooth
        self.dvy_dt = a * dvy_raw + (1 - a) * self.dvy_dt
        self.dvx_dt = a * dvx_raw + (1 - a) * self.dvx_dt

        # Predict (inflate P).
        self.P = self.P + self.Q

        long_used = 0.0
        lateral_updated = False

        if vx > self.min_speed:
            # --- Lateral Pacejka channel ---
            vx_reg = math.sqrt(vx * vx + 1.0)
            alpha_f = self.delta - math.atan2(vy + self.l_f * omega, vx_reg)
            alpha_r = -math.atan2(vy - self.l_r * omega, vx_reg)
            pf = pacejka_unit(alpha_f, self.Bf, self.Cf, self.Df, self.Ef)
            pr = pacejka_unit(alpha_r, self.Br, self.Cr, self.Dr, self.Er)
            Nz = self.m * self.g
            Nf = Nz * self.l_r / self.l_wb
            Nr = Nz * self.l_f / self.l_wb
            K = (Nf * pf * math.cos(self.delta) + Nr * pr) / self.m
            ay_meas = self.dvy_dt + vx * omega

            if abs(K) >= self.min_K:
                innovation_lat = self._kalman_update(H=K, z=ay_meas, R=self.R_lat)
                lateral_updated = True
                innov_for_pub = innovation_lat
            else:
                innov_for_pub = 0.0
                K_for_pub = K   # 보존

            # --- Longitudinal channel: saturation detection ---
            if self.long_enable and abs(self.u_ax) >= self.long_ax_thresh:
                ax_meas = self.dvx_dt - vy * omega  # body-frame
                # Slip detection: commanded > measured by threshold
                slip = abs(self.u_ax) - abs(ax_meas)
                if slip >= self.long_slip_thresh:
                    # Saturation → measurement z ≈ μ·g. H = g (derivative of h w.r.t. μ)
                    z_long = abs(ax_meas)
                    self._kalman_update(H=self.g, z=z_long, R=self.R_long)
                    long_used = 1.0
        else:
            K = 0.0

        # --- s-memory fallback when no update fired ---
        if not lateral_updated and long_used == 0.0:
            mem_mu = self._memory_lookup()
            if mem_mu is not None:
                # Pull toward memory of this s_bin
                self.mu_hat += self.mem_pull_rate * (mem_mu - self.mu_hat)
            elif self.prior_pull > 0:
                # No memory yet → pull toward init prior softly
                self.mu_hat += self.prior_pull * (self.init_mu - self.mu_hat)

        # --- Store to memory when lateral K strong ---
        if lateral_updated:
            self._memory_update()

        # --- Publish ---
        self.mu_pub.publish(Float32(data=float(self.mu_hat)))
        self.sigma_pub.publish(Float32(data=float(math.sqrt(max(self.P, 0.0)))))
        self.K_pub.publish(Float32(data=float(K)))
        self.innov_pub.publish(Float32(data=float(innov_for_pub if lateral_updated else 0.0)))
        self.long_pub.publish(Float32(data=float(long_used)))


if __name__ == "__main__":
    try:
        MuEstimatorEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
