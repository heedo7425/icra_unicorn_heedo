"""
MPCC reference parameter builder.

Per-stage params (8): [x_ref, y_ref, psi_ref, kappa_ref, inner_x, inner_y, outer_x, outer_y].

Two linearization modes:
    - theta_pred (N+1,) given → eval at those θ values (Liniger SQP shift, prev solve's
      θ state).
    - else fallback: theta_k = theta_0 + k·dt·max(vs_init, 0.5).

Track bounds: centerline ± d_left/d_right (closed track wrap).
bound_inset shrinks d's by car half-width (geometric — point-mass plan vs car body).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def build_preview(
    centerline: np.ndarray,    # (M, 11) wpnt cols [X,Y,Z,VX,_,S,KAPPA,PSI,AX,D,MU_RAD]
    N: int,
    dt: float,
    theta_0: float,
    vs_init: float,
    cl_x_idx: int = 0, cl_y_idx: int = 1, cl_s_idx: int = 5,
    cl_kappa_idx: int = 6, cl_psi_idx: int = 7,
    d_right: Optional[np.ndarray] = None,
    d_left: Optional[np.ndarray] = None,
    theta_pred: Optional[np.ndarray] = None,
    bound_inset: float = 0.10,   # car half-width
) -> np.ndarray:
    """Return (N+1, 8) param matrix."""
    M = centerline.shape[0]
    if M < 2:
        return np.zeros((N + 1, 8), dtype=np.float64)
    s_arr = centerline[:, cl_s_idx].astype(np.float64)
    s_total = float(s_arr[-1] - s_arr[0])
    if s_total <= 1e-6:
        return np.zeros((N + 1, 8), dtype=np.float64)

    use_warm = (
        theta_pred is not None
        and len(theta_pred) == N + 1
        and np.all(np.isfinite(theta_pred))
    )

    P = np.zeros((N + 1, 8), dtype=np.float64)

    for k in range(N + 1):
        theta_k = float(theta_pred[k]) if use_warm else theta_0 + k * dt * max(vs_init, 0.5)
        s_local = (theta_k - s_arr[0]) % s_total + s_arr[0]
        idx = int(np.searchsorted(s_arr, s_local, side="right") - 1)
        idx = max(0, min(idx, M - 2))
        ds = s_arr[idx + 1] - s_arr[idx]
        alpha = 0.0 if ds < 1e-9 else max(0.0, min(1.0, (s_local - s_arr[idx]) / ds))

        x0_, x1_ = centerline[idx, cl_x_idx], centerline[idx + 1, cl_x_idx]
        y0_, y1_ = centerline[idx, cl_y_idx], centerline[idx + 1, cl_y_idx]
        psi0, psi1 = centerline[idx, cl_psi_idx], centerline[idx + 1, cl_psi_idx]
        # unwrap psi for interp
        if psi1 - psi0 > math.pi:
            psi1 -= 2 * math.pi
        elif psi1 - psi0 < -math.pi:
            psi1 += 2 * math.pi
        kp0, kp1 = centerline[idx, cl_kappa_idx], centerline[idx + 1, cl_kappa_idx]

        x_ref = x0_ + alpha * (x1_ - x0_)
        y_ref = y0_ + alpha * (y1_ - y0_)
        psi_ref = math.atan2(math.sin(psi0 + alpha * (psi1 - psi0)),
                             math.cos(psi0 + alpha * (psi1 - psi0)))
        kappa_ref = kp0 + alpha * (kp1 - kp0)

        if d_right is not None and d_left is not None and len(d_right) == M and len(d_left) == M:
            dr = float(d_right[idx]) + alpha * (float(d_right[idx + 1]) - float(d_right[idx]))
            dl = float(d_left[idx]) + alpha * (float(d_left[idx + 1]) - float(d_left[idx]))
        else:
            dr = dl = 0.6
        # bound inset (car half-width). prevents car-body wall collision.
        dr = max(dr - bound_inset, 0.05)
        dl = max(dl - bound_inset, 0.05)

        # left normal = (-sin, cos). outer = ref + dl*left_normal, inner = ref - dr*left_normal.
        nx_l = -math.sin(psi_ref); ny_l = math.cos(psi_ref)
        outer_x = x_ref + dl * nx_l
        outer_y = y_ref + dl * ny_l
        inner_x = x_ref - dr * nx_l
        inner_y = y_ref - dr * ny_l

        P[k, 0] = x_ref;    P[k, 1] = y_ref
        P[k, 2] = psi_ref;  P[k, 3] = kappa_ref
        P[k, 4] = inner_x;  P[k, 5] = inner_y
        P[k, 6] = outer_x;  P[k, 7] = outer_y

    return P


def project_to_centerline(
    car_xy: tuple,
    centerline: np.ndarray,
    cl_x_idx: int = 0, cl_y_idx: int = 1, cl_s_idx: int = 5,
) -> float:
    """Project (X, Y) onto centerline → return s."""
    cx, cy = car_xy
    dx = centerline[:, cl_x_idx] - cx
    dy = centerline[:, cl_y_idx] - cy
    i = int(np.argmin(dx * dx + dy * dy))
    return float(centerline[i, cl_s_idx])
