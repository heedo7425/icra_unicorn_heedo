#!/usr/bin/env python3
"""
GP Training — CSV 데이터로 steering correction GP를 학습.

사용법:
    python3 gp_train.py --data ../data/run01.csv ../data/run02.csv

학습 내용:
    입력: [v, delta_cmd, kappa, yaw_rate, ax]
    타겟: yaw_rate_residual → steering correction
    출력: ../models/gp_residual_model.pkl
"""

import argparse
import numpy as np
import csv
import pickle
import os


class SparseGPModel:
    """Sparse GP using Random Fourier Features (RFF) approximation.
    numpy only — no torch/GPyTorch dependency.
    Inference: ~0.1ms on CPU.
    """

    def __init__(self, n_features=200, length_scales=None, noise_var=0.01):
        self.n_features = n_features
        self.noise_var = noise_var
        self.length_scales = length_scales
        self.W = None
        self.b = None
        self.alpha = None
        self.A_inv = None

    def _rff_transform(self, X):
        Z = np.cos(X @ self.W + self.b) * np.sqrt(2.0 / self.n_features)
        return Z

    def fit(self, X, y):
        N, D = X.shape
        if self.length_scales is None:
            self.length_scales = np.std(X, axis=0) + 1e-6
        X_scaled = X / self.length_scales

        np.random.seed(42)
        self.W = np.random.randn(D, self.n_features)
        self.b = np.random.uniform(0, 2 * np.pi, self.n_features)

        Z = self._rff_transform(X_scaled)
        A = Z.T @ Z + self.noise_var * np.eye(self.n_features)
        self.A_inv = np.linalg.inv(A)
        self.alpha = self.A_inv @ Z.T @ y

        y_pred = Z @ self.alpha
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        return rmse

    def predict(self, X):
        X_scaled = X / self.length_scales
        Z = self._rff_transform(X_scaled)
        pred = Z @ self.alpha
        sigma = np.sqrt(self.noise_var * (1.0 + np.sum((Z @ self.A_inv) * Z, axis=1)))
        return pred, sigma


# CSV columns: t, v, delta_cmd, kappa, yaw_rate, ax, lat_error, s_position
COL = {'t': 0, 'v': 1, 'delta_cmd': 2, 'kappa': 3, 'yaw_rate': 4,
       'ax': 5, 'lat_error': 6, 's_position': 7}


def load_csv(paths):
    """Load CSV files into numpy array (skip header)."""
    rows = []
    for path in paths:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                rows.append([float(x) for x in row])
        print(f"  {path}: {len(rows)} samples (cumulative)")
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser(description='Train GP residual model')
    parser.add_argument('--data', nargs='+', required=True, help='CSV data files')
    parser.add_argument('--output', default=None, help='Output model file')
    parser.add_argument('--wheelbase', type=float, default=0.324)
    parser.add_argument('--n_features', type=int, default=200)
    parser.add_argument('--noise_var', type=float, default=0.01)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), '..', 'models',
                                   'gp_model.pkl')

    print("=== GP Residual Training ===\n")

    # 1. Load
    print("[1] Loading data...")
    data = load_csv(args.data)

    # filter v > 0.5
    mask = data[:, COL['v']] > 0.5
    data = data[mask]
    print(f"  After v>0.5 filter: {len(data)}")

    # 2. Targets: yaw rate residual → steering correction
    print("[2] Computing targets...")
    v = data[:, COL['v']]
    delta = data[:, COL['delta_cmd']]
    yr_actual = data[:, COL['yaw_rate']]
    yr_expected = v * np.tan(delta) / args.wheelbase
    yr_residual = yr_expected - yr_actual
    targets = np.arctan(yr_residual * args.wheelbase / np.maximum(v, 0.5))

    # outlier removal (3 sigma)
    t_mean, t_std = np.mean(targets), np.std(targets)
    mask = np.abs(targets - t_mean) < 3 * t_std
    data = data[mask]
    targets = targets[mask]
    print(f"  After outlier removal: {len(data)}")
    print(f"  Target: mean={np.mean(targets):.6f} rad, std={np.std(targets):.6f} rad")

    # 3. Features: [v, delta_cmd, kappa, yaw_rate, ax]
    X = np.column_stack([
        data[:, COL['v']],
        data[:, COL['delta_cmd']],
        data[:, COL['kappa']],
        data[:, COL['yaw_rate']],
        data[:, COL['ax']],
    ])
    y = targets

    print(f"  Features: {X.shape}")

    # 4. Train
    print(f"[3] Training (n_features={args.n_features})...")
    model = SparseGPModel(n_features=args.n_features, noise_var=args.noise_var)
    rmse = model.fit(X, y)
    print(f"  Train RMSE: {rmse:.6f} rad ({np.degrees(rmse):.3f} deg)")

    # 5. Validation (last 20%)
    split = int(len(X) * 0.8)
    pred_val, sigma_val = model.predict(X[split:])
    val_rmse = np.sqrt(np.mean((y[split:] - pred_val) ** 2))
    print(f"  Val RMSE:   {val_rmse:.6f} rad ({np.degrees(val_rmse):.3f} deg)")
    print(f"  Mean sigma: {np.mean(sigma_val):.6f}")

    # 6. Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n[OK] Saved to {args.output}")


if __name__ == '__main__':
    main()
