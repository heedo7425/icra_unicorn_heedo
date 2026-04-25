"""
Build MPC horizon preview from f110_msgs waypoints.

Given the ego Frenet state (s_ego, d_ego) and a sequence of local waypoints
(from /behavior_strategy.local_wpnts or /local_waypoints), produce the per-stage
parameter vector for the acados OCP:

    p[k] = [kappa, theta, kappa_z, mu, vx_ref, n_ref]   for k = 0..N

Waypoints are spaced by arbitrary Δs; we resample to dt-scaled stages by
stepping forward along the s-axis at roughly `vx_ref * dt` per stage.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# Waypoint tuple index (matches controller_manager behavior_cb extraction):
# [x, y, z, speed, safety_ratio, s, kappa, psi, ax, d]
WPT_S = 5
WPT_KAPPA = 6
WPT_SPEED = 3
WPT_D = 9


def build_preview(
    wpnts: np.ndarray,            # (M, 10) waypoint array from behavior_cb
    s_ego: float,
    N: int,
    dt: float,
    track_length: float,
    mu_default: float,
    mu_rad: np.ndarray | None = None,      # per-waypoint pitch (theta), same M
    kappa_z: np.ndarray | None = None,     # per-waypoint vertical curvature, same M
) -> np.ndarray:
    """Return (N+1, 6) param matrix: [kappa, theta, kappa_z, mu, vx_ref, n_ref] per stage.

    Uses piecewise-constant kappa along s (step = vx_ref * dt).
    """
    if wpnts.shape[0] < 2:
        # Fallback: zeros + mu_default.
        return np.tile(
            [0.0, 0.0, 0.0, mu_default, 0.0, 0.0],
            (N + 1, 1),
        )

    # local_wpnts 는 state_machine 이 차량 앞쪽으로 잘라준 sliding window.
    # s_m 값은 글로벌 track 좌표라 wrap-around 경계 (track_length 근처) 에서는
    # monotonic 이 아니다 — np.interp 하면 엉뚱한 값이 나온다. 대신 wpnts 의
    # index 를 직접 쓴다: wpnts[0] 이 차량 현재 위치, wpnts[k*stride] 가 k 스텝 앞.

    kappa_wp = wpnts[:, WPT_KAPPA]
    vx_wp = wpnts[:, WPT_SPEED]
    d_wp = wpnts[:, WPT_D]
    M = wpnts.shape[0]
    theta_wp = mu_rad if mu_rad is not None else np.zeros(M)
    kz_wp = kappa_z if kappa_z is not None else np.zeros(M)

    # 평균 wpnt 간격 (m). wrap 지점이 있으면 큰 음수 점프가 끼어 들어서
    # mean 이 왜곡되므로 absolute diff 중 중간값 사용.
    s_wp = wpnts[:, WPT_S]
    if M >= 2:
        ds_diffs = np.abs(np.diff(s_wp))
        ds_med = float(np.median(ds_diffs)) if len(ds_diffs) > 0 else 0.1
        ds_med = max(ds_med, 0.05)
    else:
        ds_med = 0.1

    P = np.empty((N + 1, 6), dtype=np.float64)
    for k in range(N + 1):
        # 예상 앞쪽 거리 (m) → wpnt index.
        # vx_ref 로 적분하지 않고, 현재 속도 기반 일정 stride 로 샘플.
        if k == 0:
            idx = 0
        else:
            step_m = max(P[k - 1, 4], 0.5) * dt  # use previous vx_ref
            idx_float = (k * step_m) / ds_med
            idx = min(int(idx_float), M - 1)

        P[k, 0] = float(kappa_wp[idx])
        P[k, 1] = float(theta_wp[idx])
        P[k, 2] = float(kz_wp[idx])
        P[k, 3] = mu_default
        P[k, 4] = max(float(vx_wp[idx]), 0.5)
        # n_ref=0: local_wpnts.x_m/y_m are ALREADY the target path (raceline or
        # shifted avoidance path). state_machine's d_m is a label of how far
        # the shifted wpnt sits from the raceline, NOT a target offset for the
        # MPC. MPC's n is measured perpendicular to wpnt xy → we want n=0 to
        # sit exactly on that path.
        P[k, 5] = 0.0

    return P


def extract_frenet_state(
    ego_s: float,
    ego_n: float,
    ego_dpsi: float,
    ego_vx: float,
    ego_vy: float,
    ego_omega: float,
    ego_delta: float,
) -> np.ndarray:
    """Pack current state into the acados OCP x vector."""
    return np.array(
        [ego_s, ego_n, ego_dpsi, ego_vx, ego_vy, ego_omega, ego_delta],
        dtype=np.float64,
    )
