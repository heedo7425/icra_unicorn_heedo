#!/usr/bin/env python3
"""
gp_trainer — lap-end online GP residual trainer (UPenn-style).

Collects (x_k, u_k, x_{k+1}) pairs from /car_state/odom + /gp_mpc/cmd_raw,
computes residual = measured_derivative - pacejka_numpy(...) on (vx, vy, ω)
channels, then at every lap-end trains a GPyTorch BatchIndependentMultitaskGP
(3 tasks, ExactGP, ScaleKernel(RBFKernel(ard=6))) and atomically saves to
/tmp/gp_mpc_models/latest.pth.

Lap-end trigger: s wrap-around (|s_prev - s_now| > track_length/2, s_now small).

Inputs (6D): [vx, vy, omega, delta, u_ax, u_ddelta]
Outputs (3D): residual on [dvx/dt, dvy/dt, domega/dt]

Publishes:
  /gp_mpc/gp_ready       (Bool)   — true after first successful train
  /gp_mpc/train_time_s   (Float32) — wall time of last train
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty, Float32

import torch
import gpytorch


GP_INPUT_DIM = 7   # [vx, vy, ω, δ, u_ax, u_ddelta, s]
GP_TASK_DIM = 3


# ------------------------------------------------------------------------
# Numpy Pacejka RHS (mirrors mpc/vehicle_model.py::build_dynamic_bicycle_model
# for the (dvx, dvy, domega) channels).
# ------------------------------------------------------------------------
def pacejka_derivs_np(
    vx: float, vy: float, omega: float, delta: float,
    u_ax: float,
    vp: dict, mu: float, theta: float = 0.0, kappa_z: float = 0.0,
) -> np.ndarray:
    """Return [dvx, dvy, domega] from base Pacejka model (no residual)."""
    m = vp["m"]; l_f = vp["l_f"]; l_r = vp["l_r"]; l_wb = vp["l_wb"]; I_z = vp["I_z"]
    Bf, Cf, Df, Ef = vp["Bf"], vp["Cf"], vp["Df"], vp["Ef"]
    Br, Cr, Dr, Er = vp["Br"], vp["Cr"], vp["Dr"], vp["Er"]
    g = 9.81

    eps_vx = 1.5
    vx_reg = math.sqrt(vx * vx + eps_vx * eps_vx)

    Nz = m * g * math.cos(theta) - m * vx * vx * kappa_z
    Nf = Nz * l_r / l_wb
    Nr = Nz * l_f / l_wb

    alpha_f = delta - math.atan2(vy + l_f * omega, vx_reg)
    alpha_r = -math.atan2(vy - l_r * omega, vx_reg)

    def _pac(alpha, N, B, C, D, E):
        x = B * alpha
        phi = x - E * (x - math.atan(x))
        return mu * N * D * math.sin(C * math.atan(phi))

    Fyf = _pac(alpha_f, Nf, Bf, Cf, Df, Ef)
    Fyr = _pac(alpha_r, Nr, Br, Cr, Dr, Er)

    dvx = u_ax - g * math.sin(theta) - Fyf * math.sin(delta) / m + vy * omega
    dvy = (Fyf * math.cos(delta) + Fyr) / m - vx * omega
    domega = (l_f * Fyf * math.cos(delta) - l_r * Fyr) / I_z
    return np.array([dvx, dvy, domega], dtype=np.float64)


# ------------------------------------------------------------------------
# GPyTorch model — 3 independent tasks, single shared RBF kernel per task.
# ------------------------------------------------------------------------
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=3):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_tasks])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([num_tasks]),
                ard_num_dims=GP_INPUT_DIM,
            ),
            batch_shape=torch.Size([num_tasks]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ------------------------------------------------------------------------
# Trainer node
# ------------------------------------------------------------------------
class GPTrainer:
    def __init__(self) -> None:
        rospy.init_node("gp_trainer", anonymous=False)
        NS = "gp_mpc/gp"

        self.model_path = str(rospy.get_param(f"{NS}/model_path", "/tmp/gp_mpc_models/latest.pth"))
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.buffer_size = int(rospy.get_param(f"{NS}/buffer_size", 5000))
        self.train_epochs = int(rospy.get_param(f"{NS}/train_epochs", 300))
        self.train_lr = float(rospy.get_param(f"{NS}/train_lr", 0.05))
        self.train_min_samples = int(rospy.get_param(f"{NS}/train_min_samples", 500))
        self.train_max_samples = int(rospy.get_param(f"{NS}/train_max_samples", 1500))
        self.outlier_q = float(rospy.get_param(f"{NS}/outlier_quantile", 1.0))
        self.skip_first_sec = float(rospy.get_param(f"{NS}/skip_first_sec", 3.0))

        self.mu_train = float(rospy.get_param("gp_mpc/mu_default", 0.85))

        # Vehicle params (flat dict).
        vp_keys = ["m", "l_f", "l_r", "l_wb", "I_z"]
        tire_f = ["Bf", "Cf", "Df", "Ef"]
        tire_r = ["Br", "Cr", "Dr", "Er"]
        self.vp = {
            "m":    rospy.get_param("/vehicle/m", 3.54),
            "l_f":  rospy.get_param("/vehicle/l_f", 0.162),
            "l_r":  rospy.get_param("/vehicle/l_r", 0.145),
            "l_wb": rospy.get_param("/vehicle/l_wb", 0.307),
            "I_z":  rospy.get_param("/vehicle/I_z", 0.05797),
            "Bf":   rospy.get_param("/tire_front/B", 4.80),
            "Cf":   rospy.get_param("/tire_front/C", 2.16),
            "Df":   rospy.get_param("/tire_front/D", 0.65),
            "Ef":   rospy.get_param("/tire_front/E", 0.37),
            "Br":   rospy.get_param("/tire_rear/B", 20.0),
            "Cr":   rospy.get_param("/tire_rear/C", 1.50),
            "Dr":   rospy.get_param("/tire_rear/D", 0.62),
            "Er":   rospy.get_param("/tire_rear/E", 0.0),
        }

        # State buffer: list of (t, vx, vy, omega, delta, u_ax, u_ddelta, s).
        self.lock = threading.Lock()
        self.buffer = deque(maxlen=self.buffer_size)
        self.last_cmd: Optional[tuple] = None  # (delta, u_ax, u_ddelta)
        self.last_s: float = 0.0               # 최신 Frenet s (_frenet_cb 에서 갱신)

        self.start_time = rospy.Time.now().to_sec()
        self.prev_s: Optional[float] = None
        self.track_length = float(rospy.get_param("/global_republisher/track_length", 0.0))
        self.lap_end_cooldown_s = 2.0
        self.last_lap_end_t = 0.0

        self.training_in_progress = False

        torch.set_num_threads(int(rospy.get_param(f"{NS}/torch_num_threads", 2)))

        self.ready_pub = rospy.Publisher("/gp_mpc/gp_ready", Bool, queue_size=1, latch=True)
        self.train_time_pub = rospy.Publisher("/gp_mpc/train_time_s", Float32, queue_size=1)
        self.buffer_size_pub = rospy.Publisher("/gp_mpc/buffer_size", Float32, queue_size=1)

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/cmd_raw", AckermannDriveStamped, self._cmd_cb, queue_size=1)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._frenet_cb, queue_size=1)
        rospy.Subscriber("/gp_mpc/gp_reset", Empty, self._reset_cb, queue_size=1)

        # Initial ready=False.
        self.ready_pub.publish(Bool(data=False))

        rospy.loginfo(
            f"[gp_trainer] start — buffer={self.buffer_size} min={self.train_min_samples} "
            f"epochs={self.train_epochs} model_path={self.model_path}"
        )

    def _reset_cb(self, _msg) -> None:
        """/gp_mpc/gp_reset → buffer · checkpoint · ready 모두 초기화."""
        with self.lock:
            self.buffer.clear()
        try:
            if os.path.isfile(self.model_path):
                os.remove(self.model_path)
        except OSError as e:
            rospy.logwarn(f"[gp_trainer] reset: model remove failed: {e}")
        self.last_lap_end_t = rospy.Time.now().to_sec()  # 즉시 재학습 금지
        self.ready_pub.publish(Bool(data=False))
        self.buffer_size_pub.publish(Float32(data=0.0))
        self.train_time_pub.publish(Float32(data=0.0))
        rospy.loginfo("[gp_trainer] RESET — buffer + model cleared, ready=False")

    # ---- Callbacks ----
    def _cmd_cb(self, msg: AckermannDriveStamped) -> None:
        self.last_cmd = (
            float(msg.drive.steering_angle),
            float(msg.drive.acceleration),
            float(msg.drive.steering_angle_velocity),
        )

    def _odom_cb(self, msg: Odometry) -> None:
        if self.last_cmd is None:
            return
        t = msg.header.stamp.to_sec()
        if t - self.start_time < self.skip_first_sec:
            return
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        omega = float(msg.twist.twist.angular.z)
        delta, u_ax, u_ddelta = self.last_cmd
        s = self.last_s
        with self.lock:
            self.buffer.append((t, vx, vy, omega, delta, u_ax, u_ddelta, s))

    def _frenet_cb(self, msg: Odometry) -> None:
        s = float(msg.pose.pose.position.x)
        self.last_s = s
        if self.track_length <= 0.1:
            # Try lazy refresh.
            self.track_length = float(rospy.get_param("/global_republisher/track_length", 0.0))
        if self.prev_s is not None and self.track_length > 0.1:
            # Lap-end: wrap-around (prev near end, now near 0).
            if self.prev_s > 0.75 * self.track_length and s < 0.15 * self.track_length:
                now = rospy.Time.now().to_sec()
                if now - self.last_lap_end_t > self.lap_end_cooldown_s and not self.training_in_progress:
                    self.last_lap_end_t = now
                    rospy.loginfo(f"[gp_trainer] LAP END at s={s:.1f} → trigger train "
                                  f"(buffer={len(self.buffer)})")
                    threading.Thread(target=self._train_and_save, daemon=True).start()
        self.prev_s = s

    # ---- Training ----
    def _build_dataset(self) -> Optional[tuple]:
        with self.lock:
            samples = list(self.buffer)
        if len(samples) < self.train_min_samples:
            rospy.logwarn(f"[gp_trainer] buffer={len(samples)} < min={self.train_min_samples} → skip")
            return None

        X_list = []
        Y_list = []
        for i in range(len(samples) - 1):
            t0, vx0, vy0, w0, d0, a0, dr0, s0 = samples[i]
            t1, vx1, vy1, w1, d1, a1, dr1, s1 = samples[i + 1]
            dt = t1 - t0
            if not (0.005 < dt < 0.1):
                continue
            # Measured derivatives.
            dvx_m = (vx1 - vx0) / dt
            dvy_m = (vy1 - vy0) / dt
            dw_m = (w1 - w0) / dt
            # Base Pacejka derivatives at (x_k, u_k).
            dvx_p, dvy_p, dw_p = pacejka_derivs_np(
                vx0, vy0, w0, d0, a0, self.vp, mu=self.mu_train,
            )
            y = np.array([dvx_m - dvx_p, dvy_m - dvy_p, dw_m - dw_p], dtype=np.float64)
            # 7D feature: (vx, vy, ω, δ, u_ax, u_ddelta, s)
            x = np.array([vx0, vy0, w0, d0, a0, dr0, s0], dtype=np.float64)
            X_list.append(x)
            Y_list.append(y)

        if len(X_list) < self.train_min_samples:
            rospy.logwarn(f"[gp_trainer] filtered pairs={len(X_list)} < min → skip")
            return None

        X = np.stack(X_list, axis=0)  # (N, 6)
        Y = np.stack(Y_list, axis=0)  # (N, 3)

        # Outlier filter — 채널별 |y| 상위 (1-q)% 샘플 제거.
        # Pacejka 가 설명 못하는 급변 조향/IMU spike 가 tail 을 지배하는 것 방지.
        if 0.5 < self.outlier_q < 1.0:
            thr = np.quantile(np.abs(Y), self.outlier_q, axis=0)
            keep = np.all(np.abs(Y) <= thr[None, :], axis=1)
            if keep.sum() >= self.train_min_samples:
                X = X[keep]; Y = Y[keep]

        # Subsample if over max.
        if len(X) > self.train_max_samples:
            idx = np.random.default_rng(0).choice(len(X), self.train_max_samples, replace=False)
            X = X[idx]
            Y = Y[idx]

        # Standardize.
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0) + 1e-6
        y_mean = Y.mean(axis=0)
        y_std = Y.std(axis=0) + 1e-6
        Xs = (X - x_mean) / x_std
        Ys = (Y - y_mean) / y_std
        self.buffer_size_pub.publish(Float32(data=float(len(X))))
        return (
            torch.from_numpy(Xs).float(),
            torch.from_numpy(Ys).float(),
            x_mean, x_std, y_mean, y_std,
        )

    def _train_and_save(self) -> None:
        self.training_in_progress = True
        try:
            t0 = time.perf_counter()
            data = self._build_dataset()
            if data is None:
                return
            X, Y, x_mean, x_std, y_mean, y_std = data
            N = X.shape[0]
            # GPyTorch batch format: train_y shape [num_tasks, N], train_x [num_tasks, N, D].
            train_x = X.unsqueeze(0).expand(GP_TASK_DIM, -1, -1).contiguous()
            train_y = Y.t().contiguous()
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                batch_shape=torch.Size([GP_TASK_DIM])
            )
            model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, GP_TASK_DIM)
            model.train()
            likelihood.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            opt = torch.optim.Adam(model.parameters(), lr=self.train_lr)

            for i in range(self.train_epochs):
                opt.zero_grad()
                out = model(train_x)
                loss = -mll(out, train_y).sum()
                loss.backward()
                opt.step()

            # Save atomically.
            payload = {
                "state_dict": model.state_dict(),
                "likelihood_state_dict": likelihood.state_dict(),
                "train_x": train_x.detach(),
                "train_y": train_y.detach(),
                "x_mean": x_mean.astype(np.float32),
                "x_std": x_std.astype(np.float32),
                "y_mean": y_mean.astype(np.float32),
                "y_std": y_std.astype(np.float32),
                "num_tasks": GP_TASK_DIM,
                "input_dim": GP_INPUT_DIM,
                "num_samples": N,
            }
            tmp_dir = os.path.dirname(self.model_path)
            fd, tmp_path = tempfile.mkstemp(prefix=".latest_", suffix=".pth", dir=tmp_dir)
            os.close(fd)
            torch.save(payload, tmp_path)
            os.replace(tmp_path, self.model_path)

            elapsed = time.perf_counter() - t0
            self.train_time_pub.publish(Float32(data=float(elapsed)))
            self.ready_pub.publish(Bool(data=True))
            rospy.loginfo(
                f"[gp_trainer] trained N={N} epochs={self.train_epochs} "
                f"in {elapsed:.2f}s → {self.model_path}"
            )
        except Exception as e:
            rospy.logerr(f"[gp_trainer] train failed: {e}")
        finally:
            self.training_in_progress = False


if __name__ == "__main__":
    try:
        GPTrainer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
