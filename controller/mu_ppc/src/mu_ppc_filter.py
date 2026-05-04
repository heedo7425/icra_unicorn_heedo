#!/usr/bin/env python3
"""mu_ppc_filter — friction-aware overlay on top of race_stack PPC.

기존 controller_manager 의 출력을 가로채 (drive_raw) μ_eff 기반으로:
  1) friction-circle 코너 속도 한계 (v ≤ sqrt(μ·g·R)·k_safe)
  2) max_steer 스케일 (low-μ 에서 hard cap 보호)
  3) μ_eff = α · μ_prior(s) + (1-α) · μ_hat (런타임 잔차)

PPC 본체는 **손대지 않음.** 기존 Controller.py 의 모든 보정 (Stanley, future
heading, AEB, accel-ellipse) 그대로 살아있고 그 위에 hard-saturation 보호만 얹음.

Topology:
  controller_manager  ──/mu_ppc/drive_raw──▶  (this filter)  ──nav_1──▶  ackermann_mux
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np
import rospy
import tf.transformations as tft
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32MultiArray, String, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


G = 9.81


class MuPPCFilter:
    def __init__(self):
        rospy.init_node("mu_ppc_filter")

        # --- params ---
        self.mu_ref = float(rospy.get_param("~mu_ref", 1.0))
        self.mu_floor = float(rospy.get_param("~mu_floor", 0.4))
        self.mu_ceil = float(rospy.get_param("~mu_ceil", 1.5))
        self.safety_margin = float(rospy.get_param("~safety_margin", 0.85))
        self.max_steer = float(rospy.get_param("~max_steer", 0.4))
        self.steer_scale_min = float(rospy.get_param("~steer_scale_min", 0.7))
        self.kappa_lookahead_m = float(rospy.get_param("~kappa_lookahead_m", 3.0))
        self.alpha_prior = float(rospy.get_param("~alpha_prior", 0.7))
        self.ema_lambda = float(rospy.get_param("~ema_lambda", 0.05))
        self.sat_ay_thres = float(rospy.get_param("~sat_ay_thres", 3.0))
        self.sat_v_thres = float(rospy.get_param("~sat_v_thres", 1.5))
        self.min_samples = int(rospy.get_param("~min_samples", 30))

        self.drive_in_topic = rospy.get_param("~drive_in",  "/mu_ppc/drive_raw")
        self.drive_out_topic = rospy.get_param("~drive_out",
            "/vesc/high_level/ackermann_cmd_mux/input/nav_1")

        # --- state ---
        self.s_now: Optional[float] = None
        self.v_body: float = 0.0
        self.a_y: float = 0.0
        self.kappa_arr: Optional[np.ndarray] = None
        self.s_arr: Optional[np.ndarray] = None
        self.sectors: List[dict] = []
        self.global_limit = 1.0
        self.mu_hat: Optional[float] = None
        self.n_samples = 0
        self.frozen = False

        # --- pubs ---
        self.pub = rospy.Publisher(self.drive_out_topic, AckermannDriveStamped, queue_size=5)
        self.mu_eff_pub = rospy.Publisher("/mu_ppc/mu_eff", Float32, queue_size=5)
        self.mu_hat_pub = rospy.Publisher("/mu_ppc/mu_hat", Float32, queue_size=5)
        self.diag_pub = rospy.Publisher("/mu_ppc/diag", Float32MultiArray, queue_size=5)
        self.overlay_pub = rospy.Publisher("/mu_ppc/friction_overlay",
                                           MarkerArray, queue_size=2, latch=True)

        # --- subs ---
        rospy.Subscriber(self.drive_in_topic, AckermannDriveStamped,
                         self._drive_cb, queue_size=10)
        rospy.Subscriber("/global_waypoints", WpntArray, self._gw_cb, queue_size=1)
        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=5)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._frenet_cb, queue_size=5)
        rospy.Subscriber("/imu/data", Imu, self._imu_cb, queue_size=5)
        rospy.Subscriber("/mu_ppc/sectors_loaded", Bool,
                         lambda _: self._reload_sectors(), queue_size=1)
        rospy.Subscriber("/launch_mode/active", Bool, self._launch_cb, queue_size=1)

        self._reload_sectors()
        self._overlay_timer = rospy.Timer(rospy.Duration(1.0), self._publish_overlay)

        # f110-simulator 의 /mux 가 nav_mux_idx 비활성으로 시작.
        # behavior_controller 의 /key 콜백이 'n' 받으면 nav slot 토글 (켜짐 ↔ 꺼짐).
        # 직접 /mux 를 publish 해도 behavior_controller 가 다시 자기 상태로 덮어씀.
        # /key 토픽으로 한 번 'n' publish → behavior_controller 가 mux 켜고 유지.
        if rospy.get_param("/sim", False):
            self._key_pub = rospy.Publisher("/key", String, queue_size=1, latch=False)
            rospy.Timer(rospy.Duration(2.0), self._enable_nav_mux, oneshot=True)

        rospy.loginfo(
            "[mu_ppc_filter] in=%s  out=%s  mu_ref=%.2f safety=%.2f",
            self.drive_in_topic, self.drive_out_topic, self.mu_ref, self.safety_margin)

    # ------------ callbacks ------------
    def _drive_cb(self, msg: AckermannDriveStamped) -> None:
        v_raw = float(msg.drive.speed)
        d_raw = float(msg.drive.steering_angle)
        mu = self.mu_eff()

        # friction-circle speed clamp (preview window kappa max)
        kappa_max = self._kappa_max_lookahead()
        v_circle = self._v_circle(mu, kappa_max)
        v_cmd = min(v_raw, v_circle) if kappa_max > 1e-3 else v_raw

        # max_steer scale
        scale = float(np.clip(mu / self.mu_ref, self.steer_scale_min, 1.0))
        max_steer_eff = self.max_steer * scale
        d_cmd = float(np.clip(d_raw, -max_steer_eff, max_steer_eff))

        out = AckermannDriveStamped()
        out.header = msg.header
        out.drive = msg.drive
        out.drive.speed = v_cmd
        out.drive.steering_angle = d_cmd
        self.pub.publish(out)

        self.mu_eff_pub.publish(Float32(data=float(mu)))
        if self.mu_hat is not None:
            self.mu_hat_pub.publish(Float32(data=float(self.mu_hat)))
        diag = Float32MultiArray()
        diag.data = [v_cmd, v_raw, v_circle, d_cmd, d_raw, max_steer_eff, mu, kappa_max]
        self.diag_pub.publish(diag)

    def _gw_cb(self, msg: WpntArray) -> None:
        if len(msg.wpnts) < 2:
            return
        try:
            self.kappa_arr = np.array([w.kappa_radpm for w in msg.wpnts], dtype=np.float64)
        except AttributeError:
            xy = np.array([[w.x_m, w.y_m] for w in msg.wpnts])
            d1 = np.gradient(xy, axis=0); d2 = np.gradient(d1, axis=0)
            den = (d1[:, 0]**2 + d1[:, 1]**2)**1.5
            den = np.where(den < 1e-6, 1e-6, den)
            self.kappa_arr = (d1[:, 0]*d2[:, 1] - d1[:, 1]*d2[:, 0]) / den
        self.s_arr = np.array([w.s_m for w in msg.wpnts], dtype=np.float64)
        self.xy_arr = np.array([[w.x_m, w.y_m] for w in msg.wpnts], dtype=np.float64)

    def _odom_cb(self, msg: Odometry) -> None:
        self.v_body = float(msg.twist.twist.linear.x)

    def _frenet_cb(self, msg: Odometry) -> None:
        self.s_now = float(msg.pose.pose.position.x)

    def _imu_cb(self, msg: Imu) -> None:
        # vesc 90deg rotation: a_y meas ≈ -linear_acceleration.x
        self.a_y = -float(msg.linear_acceleration.x)
        # update mu_hat residual
        if not self.frozen and self.v_body > self.sat_v_thres \
                and abs(self.a_y) > self.sat_ay_thres:
            mu_inst = min(abs(self.a_y) / G, self.mu_ceil)
            if self.mu_hat is None:
                self.mu_hat = mu_inst
            else:
                self.mu_hat = (1 - self.ema_lambda) * self.mu_hat + self.ema_lambda * mu_inst
            self.n_samples += 1

    def _launch_cb(self, msg: Bool) -> None:
        self.frozen = bool(msg.data)

    # ------------ μ ------------
    def _reload_sectors(self) -> None:
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
        self.global_limit = float(rospy.get_param("/friction_map_params/global_friction_limit", 1.5))
        if secs:
            rospy.loginfo("[mu_ppc_filter] reloaded %d sectors", len(secs))

    def mu_prior(self) -> float:
        if self.s_now is None or not self.sectors:
            return 1.0
        for sec in self.sectors:
            if sec["s_start"] >= 0 and sec["s_start"] <= self.s_now <= sec["s_end"]:
                return min(sec["friction"], self.global_limit)
        return 1.0

    def mu_eff(self) -> float:
        prior = self.mu_prior()
        if self.mu_hat is None or self.n_samples < self.min_samples:
            return float(np.clip(prior, self.mu_floor, self.mu_ceil))
        a = self.alpha_prior
        return float(np.clip(a * prior + (1 - a) * self.mu_hat,
                             self.mu_floor, self.mu_ceil))

    # ------------ kinematics ------------
    def _kappa_max_lookahead(self) -> float:
        if self.kappa_arr is None or self.s_arr is None or self.s_now is None:
            return 0.0
        s = self.s_now
        s_max = float(self.s_arr[-1]) + (float(self.s_arr[1]) - float(self.s_arr[0]))
        # window [s, s + base + v*dt_preview]
        win = self.kappa_lookahead_m + 0.3 * max(self.v_body, 0.0)
        s_end = s + win
        # wrap
        idxs = []
        s_arr = self.s_arr
        N = len(s_arr)
        if s_end <= s_max:
            mask = (s_arr >= s) & (s_arr <= s_end)
            idxs = np.where(mask)[0]
        else:
            mask1 = (s_arr >= s)
            mask2 = (s_arr <= (s_end - s_max))
            idxs = np.concatenate([np.where(mask1)[0], np.where(mask2)[0]])
        if len(idxs) == 0:
            return 0.0
        return float(np.max(np.abs(self.kappa_arr[idxs])))

    def _v_circle(self, mu: float, kappa_max: float) -> float:
        R = 1.0 / max(abs(kappa_max), 1e-3)
        return math.sqrt(max(mu, self.mu_floor) * G * R) * self.safety_margin

    # ------------ sim mux enable ------------
    def _enable_nav_mux(self, _evt) -> None:
        nav_char = str(rospy.get_param("/nav_key_char", "n"))
        # 한 번 보내고 끝. behavior_controller 가 mux 토글 → on 상태로 유지.
        self._key_pub.publish(String(data=nav_char))
        rospy.loginfo("[mu_ppc_filter] /key '%s' published → nav mux toggle", nav_char)

    # ------------ overlay ------------
    def _publish_overlay(self, _evt) -> None:
        if not hasattr(self, "xy_arr") or self.xy_arr is None or not self.sectors:
            return
        xy = self.xy_arr
        N = len(xy)
        mu_per = np.ones(N)
        for sec in self.sectors:
            try:
                a = int(rospy.get_param(
                    f"/friction_map_params/Sector{self.sectors.index(sec)}/start", 0))
                b = int(rospy.get_param(
                    f"/friction_map_params/Sector{self.sectors.index(sec)}/end", 0))
                if 0 <= a < N and 0 <= b < N and a <= b:
                    mu_per[a:b+1] = sec["friction"]
            except Exception:
                continue
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"; m.header.stamp = rospy.Time.now()
        m.ns = "mu_ppc_friction_overlay"; m.id = 0
        m.type = Marker.SPHERE_LIST; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.18
        for idx in range(N):
            mu = float(mu_per[idx])
            p = Point(); p.x = float(xy[idx, 0]); p.y = float(xy[idx, 1]); p.z = 0.05
            m.points.append(p)
            if mu < 0.9:
                t = max(0.0, min(1.0, (0.9 - mu)/0.5))
                m.colors.append(ColorRGBA(r=0.1, g=0.2, b=0.5+0.5*t, a=0.9))
            elif mu > 1.1:
                t = max(0.0, min(1.0, (mu-1.1)/0.4))
                m.colors.append(ColorRGBA(r=0.5+0.5*t, g=0.1, b=0.1, a=0.9))
            else:
                m.colors.append(ColorRGBA(r=0.55, g=0.55, b=0.55, a=0.5))
        ma.markers.append(m)
        self.overlay_pub.publish(ma)


if __name__ == "__main__":
    MuPPCFilter()
    rospy.spin()
