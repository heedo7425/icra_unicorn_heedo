"""Synthetic ground-truth generator for solver validation.

Produce (vx, vy, omega, delta, ax) that are fully self-consistent with the
steady-state bicycle equations used inside
`on_track_sys_id/src/helpers/solve_pacejka.analyse_tires`.

For each (vx, R) grid point on a constant-radius turn:

    omega  = vx / R
    F_yf   = m * vx^2 * l_r / (R * l_wb * cos(delta))
    F_yr   = m * vx^2 * l_f / (R * l_wb)

Inverting the GT Pacejka on these F_y values gives the slip angles:

    alpha_f = Pacejka^{-1}_f (F_yf, F_zf)
    alpha_r = Pacejka^{-1}_r (F_yr, F_zr)

From the slip definitions:

    alpha_r = -atan((vy - l_r*omega) / vx)   ->   vy = l_r*omega - vx*tan(alpha_r)
    alpha_f = -atan((vy + l_f*omega) / vx) + delta
              ->   delta = alpha_f + atan(vy/vx + l_f/R)

Because delta enters F_yf through cos(delta), we iterate delta a few times.
This gives a perfectly self-consistent fixed point — if fit_pacejka is given
this data and the GT as warm start, it must return the GT (to solver
tolerance) modulo identifiability.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def pacejka(params, alpha, F_z):
    B, C, D, E = params
    return F_z * D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))


def invert_pacejka(params, F_z, F_y_target, n_iter: int = 100):
    """Bisection on the positive-alpha monotonic branch."""
    sign = 1.0 if F_y_target >= 0 else -1.0
    target = abs(F_y_target)
    lo, hi = 0.0, 0.45
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if pacejka(params, mid, F_z) < target:
            lo = mid
        else:
            hi = mid
    return sign * 0.5 * (lo + hi)


def generate_dataset(model: Dict[str, float],
                     C_Pf_gt, C_Pr_gt,
                     vx_levels=(1.5, 2.0, 2.5, 3.0, 3.5),
                     radii=(1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0),
                     noise_frac: float = 0.01,
                     seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = model["m"]
    l_f = model["l_f"]
    l_r = model["l_r"]
    l_wb = model["l_wb"]
    g = 9.81

    F_zf = m * g * l_r / l_wb
    F_zr = m * g * l_f / l_wb

    vx_l, vy_l, omega_l, delta_l, ax_l = [], [], [], [], []

    for vx in vx_levels:
        for R in radii:
            omega = vx / R
            F_yr = m * vx * vx * l_f / (R * l_wb)
            alpha_r = invert_pacejka(C_Pr_gt, F_zr, F_yr)
            vy = l_r * omega - vx * np.tan(alpha_r)

            # Iterate delta <-> F_yf (cos(delta) coupling).
            delta = 0.0
            for _ in range(6):
                F_yf = m * vx * vx * l_r / (R * l_wb * np.cos(delta))
                alpha_f = invert_pacejka(C_Pf_gt, F_zf, F_yf)
                delta = alpha_f + np.arctan(vy / vx + l_f / R)

            # Skip samples that blew past solver bounds.
            if abs(alpha_f) > 0.45 or abs(alpha_r) > 0.18:
                continue

            # Noise (small, multiplicative on vx/omega + additive on vy/delta).
            if noise_frac > 0:
                vx_n = vx * (1.0 + noise_frac * rng.standard_normal())
                omega_n = omega * (1.0 + noise_frac * rng.standard_normal())
                vy_n = vy + noise_frac * 0.05 * rng.standard_normal()
                delta_n = delta + noise_frac * 0.005 * rng.standard_normal()
            else:
                vx_n, omega_n, vy_n, delta_n = vx, omega, vy, delta

            vx_l.append(vx_n)
            vy_l.append(vy_n)
            omega_l.append(omega_n)
            delta_l.append(delta_n)
            ax_l.append(0.0)

            # Mirror to the other side (negative turn) for symmetric coverage.
            vx_l.append(vx_n)
            vy_l.append(-vy_n)
            omega_l.append(-omega_n)
            delta_l.append(-delta_n)
            ax_l.append(0.0)

    return dict(
        t=np.arange(len(vx_l)) / 70.0,
        vx=np.asarray(vx_l),
        vy=np.asarray(vy_l),
        omega=np.asarray(omega_l),
        delta=np.asarray(delta_l),
        ax=np.asarray(ax_l),
    )
