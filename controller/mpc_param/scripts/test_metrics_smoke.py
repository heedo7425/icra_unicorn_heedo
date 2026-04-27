#!/usr/bin/env python3
"""Smoke test for metrics.py — no ROS needed.

Synthesizes one fake lap of data and checks that compute_lap_metrics returns
sensible numbers + sectors. Run from repo root:
    python3 controller/mpc_param/scripts/test_metrics_smoke.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # mpc_param/

from daemon.metrics import (
    auto_sectors_by_curvature, compute_lap_metrics, trail_row, TRAIL_HEADER,
)


def make_raceline(L=80.0, ds=0.1):
    s = np.arange(0, L, ds)
    # Two corners at s=20 and s=55, otherwise straight.
    kappa = np.zeros_like(s)
    kappa += 0.6 * np.exp(-((s - 20) ** 2) / 6.0)
    kappa += 0.5 * np.exp(-((s - 55) ** 2) / 8.0)
    psi = np.cumsum(kappa) * ds
    x = np.cumsum(np.cos(psi)) * ds
    y = np.cumsum(np.sin(psi)) * ds
    z = np.zeros_like(s)
    vx = 5.0 + 3.0 * (1 - kappa / max(kappa.max(), 1e-6))
    ax = np.zeros_like(s); d = np.zeros_like(s); mu = np.zeros_like(s)
    safety = np.zeros_like(s)
    return np.stack([x, y, z, vx, safety, s, kappa, psi, ax, d, mu], axis=1)


def make_snapshot(t0=0.0, t1=20.0, dt=0.02):
    t = np.arange(t0, t1, dt)
    n = len(t)
    s = np.linspace(0, 70, n)
    n_lat = 0.05 * np.sin(t * 0.5)
    n_lat[(s > 18) & (s < 22)] = 0.45            # corner-1 understeer drift
    vx = 6.0 + 0.3 * np.sin(t * 0.2)
    vy = 0.05 * np.sin(t * 1.0)
    yaw = np.cumsum(0.02 * np.sin(t * 0.4)) * dt
    omega = 0.5 * np.sin(t * 1.0)
    delta = 0.1 * np.sin(t * 0.7)
    speed_cmd = vx + 0.2
    solve_ms = 5 + 0.5 * np.random.randn(n)
    ay = 0.05 * np.sin(t)
    mu_used = np.full(n, 0.85)

    def pair(arr): return (t.copy(), arr.astype(np.float64))
    inf_t = np.array([], dtype=np.float64)
    return {
        "vx": pair(vx), "vy": pair(vy), "yaw": pair(yaw),
        "x":  pair(np.cumsum(vx) * dt), "y": pair(np.cumsum(vy) * dt),
        "s":  pair(s), "n": pair(n_lat),
        "omega": pair(omega), "ay_imu": pair(ay),
        "delta_cmd": pair(delta), "speed_cmd": pair(speed_cmd),
        "solve_ms": pair(solve_ms), "mu_used": pair(mu_used),
        "infeasible": (inf_t, inf_t.copy()),
    }


def main():
    raceline = make_raceline()
    sectors = auto_sectors_by_curvature(raceline, kappa_thr=0.4, min_len_m=1.5)
    print(f"sectors auto-split: {len(sectors)}")
    for s in sectors:
        print(f"  {s.name:>5} {s.type:<8} s=[{s.s_start:6.2f},{s.s_end:6.2f}] "
              f"κ_peak={s.kappa_peak:.3f}")

    snap = make_snapshot()
    vp = {"l_f": 0.162, "l_r": 0.145}
    lap_event = {"lap_count": 7, "lap_time": 19.85,
                 "avg_lat_err": 0.06, "max_lat_err": 0.45,
                 "t_start": 0.0, "t_end": 20.0}
    metrics = compute_lap_metrics(snap, raceline, vp, sectors, lap_event,
                                  mu_for_friction=0.85)
    lap = metrics["lap"]
    print("\nLAP:")
    for k in ("lap_time", "samples", "mean_abs_n", "max_abs_n",
              "max_ay_usage", "u_steer_rate_rms", "solve_p99_ms",
              "omega_oscillation_hz", "infeasible_count"):
        print(f"  {k:>22}: {lap[k]}")
    print(f"\nCORNERS ({len(metrics['corners'])}):")
    for c in metrics["corners"]:
        print(f"  {c['name']} max_n_exit_abs={c['max_n_exit_abs']:.3f} "
              f"n_exit_signed_mean={c['n_exit_signed_mean']:+.3f} "
              f"max_ay_usage={c['max_ay_usage']:.2f}")

    # csv row check
    print("\ntrail_row header has", len(TRAIL_HEADER), "cols")
    row = trail_row(metrics, applied_diff={}, ts=0.0)
    assert len(row) == len(TRAIL_HEADER)
    print("trail_row OK")


if __name__ == "__main__":
    main()
