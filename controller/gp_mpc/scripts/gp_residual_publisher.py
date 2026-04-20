#!/usr/bin/env python3
"""
gp_residual_publisher — 50Hz GP eval → /gp_mpc/residual.

Hot-reloads GP checkpoint at /tmp/gp_mpc_models/latest.pth (mtime watch).
Subscribes to /car_state/odom + /gp_mpc/cmd_raw for 6D input.
Publishes /gp_mpc/residual (Float32MultiArray, [Δvx, Δvy, Δω]).

While gp_ready=False (no model yet) → publishes zeros.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty, Float32, Float32MultiArray

import torch
import gpytorch

from gp_trainer import BatchIndependentMultitaskGPModel, GP_INPUT_DIM, GP_TASK_DIM


class GPResidualPublisher:
    def __init__(self) -> None:
        rospy.init_node("gp_residual_publisher", anonymous=False)
        NS = "gp_mpc/gp"

        self.model_path = str(rospy.get_param(f"{NS}/model_path", "/tmp/gp_mpc_models/latest.pth"))
        self.clip = np.array(
            rospy.get_param(f"{NS}/residual_clip", [10.0, 5.0, 8.0]),
            dtype=np.float64,
        )
        loop_hz = float(rospy.get_param("gp_mpc/loop_rate_hz", 50.0))
        self.period = 1.0 / loop_hz

        torch.set_num_threads(int(rospy.get_param(f"{NS}/torch_num_threads", 2)))

        self.lock = threading.Lock()
        self.model = None
        self.likelihood = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.model_mtime = 0.0
        self.gp_ready = False

        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.delta = 0.0
        self.u_ax = 0.0
        self.u_ddelta = 0.0
        self.s = 0.0   # Frenet arc length (GP input 7th feature)

        self.res_pub = rospy.Publisher("/gp_mpc/residual", Float32MultiArray, queue_size=1)
        self.sigma_pub = rospy.Publisher("/gp_mpc/gp_sigma", Float32MultiArray, queue_size=1)
        self.ready_pub = rospy.Publisher("/gp_mpc/gp_ready_pub_echo", Bool, queue_size=1)

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/cmd_raw", AckermannDriveStamped, self._cmd_cb, queue_size=1)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._frenet_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/gp_ready", Bool, self._gp_ready_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/mu_adapt_enable", Bool, self._enable_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/gp_reset", Empty, self._reset_cb, queue_size=1)
        self.enable = True

        rospy.Timer(rospy.Duration(self.period), self._tick)

        rospy.loginfo(
            f"[gp_residual_publisher] start — watch={self.model_path} clip={self.clip.tolist()}"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self.vx = float(msg.twist.twist.linear.x)
        self.vy = float(msg.twist.twist.linear.y)
        self.omega = float(msg.twist.twist.angular.z)

    def _cmd_cb(self, msg: AckermannDriveStamped) -> None:
        self.delta = float(msg.drive.steering_angle)
        self.u_ax = float(msg.drive.acceleration)
        self.u_ddelta = float(msg.drive.steering_angle_velocity)

    def _frenet_cb(self, msg: Odometry) -> None:
        self.s = float(msg.pose.pose.position.x)

    def _gp_ready_cb(self, msg: Bool) -> None:
        self.gp_ready = bool(msg.data)

    def _enable_cb(self, msg: Bool) -> None:
        self.enable = bool(msg.data)

    def _reset_cb(self, _msg) -> None:
        with self.lock:
            self.model = None
            self.likelihood = None
            self.model_mtime = 0.0
        rospy.loginfo("[gp_residual_publisher] RESET — model unloaded")

    def _maybe_reload(self) -> None:
        if not os.path.isfile(self.model_path):
            return
        try:
            mtime = os.path.getmtime(self.model_path)
        except OSError:
            return
        if mtime <= self.model_mtime:
            return
        try:
            payload = torch.load(self.model_path, map_location="cpu")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[gp_residual_publisher] load fail: {e}")
            return
        with self.lock:
            train_x = payload["train_x"]
            train_y = payload["train_y"]
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                batch_shape=torch.Size([GP_TASK_DIM])
            )
            self.model = BatchIndependentMultitaskGPModel(
                train_x, train_y, self.likelihood, GP_TASK_DIM
            )
            self.model.load_state_dict(payload["state_dict"])
            self.likelihood.load_state_dict(payload["likelihood_state_dict"])
            self.model.eval()
            self.likelihood.eval()
            self.x_mean = np.asarray(payload["x_mean"], dtype=np.float64)
            self.x_std = np.asarray(payload["x_std"], dtype=np.float64)
            self.y_mean = np.asarray(payload["y_mean"], dtype=np.float64)
            self.y_std = np.asarray(payload["y_std"], dtype=np.float64)
            self.model_mtime = mtime
        rospy.loginfo(
            f"[gp_residual_publisher] loaded model N={int(payload.get('num_samples', 0))} "
            f"mtime={mtime:.0f}"
        )

    def _tick(self, _evt) -> None:
        self._maybe_reload()
        if self.model is None or not self.gp_ready or not self.enable:
            # Zero output.
            self.res_pub.publish(Float32MultiArray(data=[0.0, 0.0, 0.0]))
            return
        x_np = np.array(
            [self.vx, self.vy, self.omega, self.delta, self.u_ax, self.u_ddelta, self.s],
            dtype=np.float64,
        )
        xs = (x_np - self.x_mean) / self.x_std
        xt = torch.from_numpy(xs).float().unsqueeze(0)  # (1, 6)
        xt_b = xt.unsqueeze(0).expand(GP_TASK_DIM, -1, -1).contiguous()  # (3, 1, 6)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out = self.model(xt_b)
            mean = out.mean.squeeze(-1).numpy()        # (3,)
            var = out.variance.squeeze(-1).numpy()     # (3,)

        # Unscale, σ-attenuate, clip.
        residual = mean * self.y_std + self.y_mean
        sigma = np.sqrt(np.maximum(var, 0.0)) * self.y_std
        # Attenuate when uncertainty is high: w = 1 / (1 + (σ/σ0)²) where
        # σ0 ≈ y_std (training-time label scale). High σ → residual suppressed,
        # preventing OCP-infeasible overshoot (RESCUE 이벤트 원인).
        sigma0 = np.maximum(self.y_std, 1e-3)
        att = 1.0 / (1.0 + (sigma / sigma0) ** 2)
        residual = residual * att
        residual = np.clip(residual, -self.clip, self.clip)

        self.res_pub.publish(Float32MultiArray(data=residual.astype(np.float32).tolist()))
        self.sigma_pub.publish(Float32MultiArray(data=sigma.astype(np.float32).tolist()))


if __name__ == "__main__":
    # Make scripts/ importable so 'from gp_trainer import ...' works.
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    try:
        GPResidualPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
