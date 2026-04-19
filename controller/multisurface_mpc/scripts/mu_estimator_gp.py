#!/usr/bin/env python3
"""
Online μ estimator — GP ensemble (Nagy 2023 pattern, simplified).

Design:
  - At init, load an ensemble of N surface-specific GPs from disk. Each GP
    takes feature x_t = [v_x, v_y, ω, δ, a_x_cmd] and outputs predicted μ_i
    for that surface's friction coefficient.
  - Maintain a rolling buffer of (feature, observed μ_proxy) pairs; every
    tick compute per-GP prediction error over the buffer. Convex weights
    w_i = softmax(-β · err_i).
  - Publish blended μ = Σ w_i · μ_i.

This is a **stub** — a full implementation requires:
  1. Offline data collection on ≥2 surfaces (we have only μ=1.0 sim today)
  2. Training scripts (GPy/sklearn with inducing points)
  3. Saved ensemble pickle.

Current behavior when no ensemble file exists (default):
  - Falls back to a single "observed μ proxy" estimator identical in spirit
    to RLS but moving-average style. This lets the launch param `mu_source:=gp`
    produce *some* reasonable output so the end-to-end pipeline can be
    exercised before real GPs are trained.

When the ensemble pickle exists at `/mpc_ms/gp/ensemble_path`, it uses real
GP prediction + convex blending.
"""

from __future__ import annotations

import math
import os
import pickle
from collections import deque
from typing import List, Optional

import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


class MuEstimatorGP:
    def __init__(self) -> None:
        rospy.init_node("mu_estimator_gp", anonymous=False)
        NS = "mpc_ms/gp"

        self.ensemble_path = rospy.get_param(f"{NS}/ensemble_path",
                                             "/tmp/gp_ensemble_srx1.pkl")
        self.init_mu = float(rospy.get_param(f"{NS}/init_mu", 0.85))
        self.beta = float(rospy.get_param(f"{NS}/blend_temperature", 5.0))
        self.buf_size = int(rospy.get_param(f"{NS}/buffer_size", 50))

        # Ensemble: list of (predict_fn(x)->mu, name). Load if exists.
        self.ensemble: List[dict] = []
        self._load_ensemble()

        # Rolling buffer of (feature, observed mu_proxy).
        self.buffer = deque(maxlen=self.buf_size)

        # Latest state cache.
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.delta = 0.0
        self.ax_cmd = 0.0
        self.mu_proxy = self.init_mu   # fallback when no ensemble

        # --- Pub / Sub ---
        self.mu_pub = rospy.Publisher("/mpc_ms/mu_estimate", Float32, queue_size=1)
        self.weights_pub = rospy.Publisher("/mpc_ms/gp_weights", Float32, queue_size=1)  # publish first weight as summary

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/imu/data", Imu, self._imu_cb, queue_size=1)
        rospy.Subscriber("/vesc/high_level/ackermann_cmd_mux/input/nav_1",
                         AckermannDriveStamped, self._cmd_cb, queue_size=1)

        rospy.loginfo(
            f"[mu_estimator_gp] init mu={self.init_mu} beta={self.beta} "
            f"buf={self.buf_size} ensemble={len(self.ensemble)} surfaces"
        )

        self.rate_hz = 50.0
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)

    # ---- Ensemble I/O ----
    def _load_ensemble(self) -> None:
        if not os.path.exists(self.ensemble_path):
            rospy.logwarn(
                f"[mu_estimator_gp] no ensemble at {self.ensemble_path}; "
                f"using moving-average proxy (stub mode)"
            )
            return
        try:
            with open(self.ensemble_path, "rb") as f:
                data = pickle.load(f)
            # Expected format: list of dicts {name: str, gp: sklearn/GPy model}
            self.ensemble = list(data)
            rospy.loginfo(
                f"[mu_estimator_gp] loaded {len(self.ensemble)} surface GPs "
                f"from {self.ensemble_path}"
            )
        except Exception as e:
            rospy.logerr(f"[mu_estimator_gp] ensemble load failed: {e}")
            self.ensemble = []

    # ---- Callbacks ----
    def _odom_cb(self, msg: Odometry) -> None:
        self.vx = float(msg.twist.twist.linear.x)
        self.vy = float(msg.twist.twist.linear.y)

    def _imu_cb(self, msg: Imu) -> None:
        self.omega = float(msg.angular_velocity.z)
        ay = float(msg.linear_acceleration.y)
        # Moving-average μ proxy (fallback when no ensemble).
        if abs(ay) >= 2.0:
            z = abs(ay) / 9.81
            alpha = 0.05
            self.mu_proxy = (1 - alpha) * self.mu_proxy + alpha * z

    def _cmd_cb(self, msg: AckermannDriveStamped) -> None:
        self.ax_cmd = float(msg.drive.acceleration)
        self.delta = float(msg.drive.steering_angle)

    # ---- Main tick ----
    def _tick(self, _evt) -> None:
        feat = np.array([self.vx, self.vy, self.omega, self.delta, self.ax_cmd])

        if not self.ensemble:
            # Stub mode: publish moving-average μ proxy.
            self.mu_pub.publish(Float32(data=float(self.mu_proxy)))
            self.weights_pub.publish(Float32(data=1.0))
            return

        # GP ensemble prediction per member.
        mus = []
        for member in self.ensemble:
            gp = member["gp"]
            mu_i = float(gp.predict(feat.reshape(1, -1))[0])
            mus.append(mu_i)
        mus = np.array(mus)

        # Per-member error over recent buffer (if enough samples).
        errs = np.zeros(len(self.ensemble))
        if len(self.buffer) >= 10:
            for i, member in enumerate(self.ensemble):
                gp = member["gp"]
                buf_feats = np.array([b[0] for b in self.buffer])
                buf_mus = np.array([b[1] for b in self.buffer])
                preds = gp.predict(buf_feats)
                errs[i] = float(np.mean((preds - buf_mus) ** 2))
        # Convex weights via softmax(-β·err).
        logits = -self.beta * errs
        logits -= logits.max()
        w = np.exp(logits)
        w = w / w.sum()

        mu_blend = float(np.dot(w, mus))

        # Update buffer with (feature, proxy-observed μ).
        self.buffer.append((feat.copy(), float(self.mu_proxy)))

        self.mu_pub.publish(Float32(data=mu_blend))
        self.weights_pub.publish(Float32(data=float(w[0])))


if __name__ == "__main__":
    try:
        MuEstimatorGP()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
