#!/usr/bin/env python3
"""
GP Model Evaluator — 학습된 모델 성능 검증 + 시각화.

사용법:
    python3 gp_eval.py --model ../models/gp_residual_model.pkl --data ../data/run01.csv
"""

import argparse
import numpy as np
import csv
import pickle
import os

COL = {'t': 0, 'v': 1, 'delta_cmd': 2, 'kappa': 3, 'yaw_rate': 4,
       'ax': 5, 'lat_error': 6, 's_position': 7}


def load_csv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([float(x) for x in row])
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate GP residual model')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--wheelbase', type=float, default=0.324)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    data = load_csv(args.data)
    data = data[data[:, COL['v']] > 0.5]

    X = np.column_stack([
        data[:, COL['v']], data[:, COL['delta_cmd']],
        data[:, COL['kappa']], data[:, COL['yaw_rate']],
        data[:, COL['ax']],
    ])

    v = data[:, COL['v']]
    delta = data[:, COL['delta_cmd']]
    yr_actual = data[:, COL['yaw_rate']]
    yr_expected = v * np.tan(delta) / args.wheelbase
    y_actual = np.arctan((yr_expected - yr_actual) * args.wheelbase / np.maximum(v, 0.5))

    y_pred, sigma = model.predict(X)
    error = y_actual - y_pred
    rmse = np.sqrt(np.mean(error ** 2))

    print(f"=== GP Residual Evaluation ===")
    print(f"RMSE:       {np.degrees(rmse):.3f} deg")
    print(f"Max |pred|: {np.degrees(np.max(np.abs(y_pred))):.3f} deg")
    print(f"Mean sigma: {np.mean(sigma):.6f}")

    for lo, hi in [(0.5, 3), (3, 6), (6, 9), (9, 12)]:
        m = (v >= lo) & (v < hi)
        if np.sum(m) > 0:
            print(f"  v=[{lo}-{hi}]: mean|corr|={np.degrees(np.mean(np.abs(y_pred[m]))):.3f} deg, n={np.sum(m)}")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(np.degrees(y_actual), np.degrees(y_pred), alpha=0.3, s=2)
    lim = max(np.max(np.abs(y_actual)), np.max(np.abs(y_pred)))
    ax.plot([-np.degrees(lim), np.degrees(lim)],
            [-np.degrees(lim), np.degrees(lim)], 'r--')
    ax.set_xlabel('Actual (deg)')
    ax.set_ylabel('Predicted (deg)')
    ax.set_title(f'RMSE={np.degrees(rmse):.3f} deg')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    t = np.arange(len(y_pred)) / 40.0
    ax.plot(t, np.degrees(y_actual), alpha=0.5, label='actual', lw=0.5)
    ax.plot(t, np.degrees(y_pred), alpha=0.7, label='predicted', lw=0.5)
    ax.fill_between(t, np.degrees(y_pred - 2*sigma), np.degrees(y_pred + 2*sigma),
                    alpha=0.2, label='2sigma')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Correction (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(v, np.degrees(np.abs(y_pred)), alpha=0.3, s=2)
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('|Correction| (deg)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(v, sigma, alpha=0.3, s=2)
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Uncertainty')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = args.model.replace('.pkl', '_eval.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot: {plot_path}")
    plt.show()


if __name__ == '__main__':
    main()
