"""Fit pipeline.

Reuses `on_track_sys_id/src/helpers/solve_pacejka.py` for the LSQ fit and
implements the race-day quality gate on top.

Inputs come as a dict of numpy arrays (recorder.as_arrays()).
"""
from __future__ import annotations

import sys
import os
from typing import Any, Dict, Tuple


def _import_solver():
    """solve_pacejka.py lives under on_track_sys_id/src/helpers. Insert on path."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))  # system_identification/
    helpers = os.path.join(root, "on_track_sys_id", "src", "helpers")
    if helpers not in sys.path:
        sys.path.insert(0, helpers)
    import solve_pacejka as sp  # noqa: E402
    return sp


def butterworth_filter(x, hz: float = 70.0, cutoff_hz: float = 5.0, order: int = 3):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * hz
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, x)


def fit_pacejka(arrays: Dict[str, Any], model: Dict[str, float],
                hz: float = 70.0, cutoff_hz: float = 5.0,
                apply_filter: bool = True) -> Tuple:
    """Return (C_Pf, C_Pr, diag) where diag has alphas/Fys for gate evaluation.

    Set apply_filter=False when samples are not time-series (e.g. steady-state
    grid from synthetic_gt.generate_dataset) — filtfilt on unordered samples
    introduces spurious artifacts.
    """
    import numpy as np
    sp = _import_solver()

    if apply_filter:
        vx = butterworth_filter(arrays["vx"], hz, cutoff_hz)
        vy = butterworth_filter(arrays["vy"], hz, cutoff_hz)
        omega = butterworth_filter(arrays["omega"], hz, cutoff_hz)
        ax = butterworth_filter(arrays["ax"], hz, cutoff_hz)
    else:
        vx = np.asarray(arrays["vx"], dtype=float).copy()
        vy = np.asarray(arrays["vy"], dtype=float).copy()
        omega = np.asarray(arrays["omega"], dtype=float).copy()
        ax = np.asarray(arrays["ax"], dtype=float).copy()
    delta = np.asarray(arrays["delta"], dtype=float).copy()

    # solve_pacejka internally filters ω/α bounds.
    alpha_f, alpha_r, F_zf, F_zr, F_yf, F_yr = sp.analyse_tires(
        model, vx, vy, omega, delta, ax)
    C_Pf, C_Pr = sp.solve_pacejka(model, vx, vy, omega, delta, ax)

    diag = dict(alpha_f=alpha_f, alpha_r=alpha_r,
                F_zf=F_zf, F_zr=F_zr, F_yf=F_yf, F_yr=F_yr)
    return C_Pf, C_Pr, diag


def _bin_coverage(alpha, bin_width):
    import numpy as np
    if alpha.size == 0:
        return 0
    bins = np.arange(0.0, float(np.max(np.abs(alpha))) + bin_width, bin_width)
    counts, _ = np.histogram(np.abs(alpha), bins=bins)
    return int((counts > 0).sum())


def evaluate_fit(C_Pf, C_Pr, diag, model, prior_yaml, gate: Dict[str, Any]):
    """Return (accept: bool, metrics: dict)."""
    import numpy as np
    sp = _import_solver()

    def _rmse(Cp, alpha, F_z, F_y):
        pred = sp.pacejka_formula(Cp, alpha, F_z)
        denom = np.mean(np.abs(F_y)) if F_y.size else 1.0
        return float(np.sqrt(np.mean((pred - F_y) ** 2)) / max(denom, 1e-6))

    rmse_f = _rmse(C_Pf, diag["alpha_f"], diag["F_zf"], diag["F_yf"])
    rmse_r = _rmse(C_Pr, diag["alpha_r"], diag["F_zr"], diag["F_yr"])

    cov_f = _bin_coverage(diag["alpha_f"], float(gate.get("coverage_front_bin_rad", 0.05)))
    cov_r = _bin_coverage(diag["alpha_r"], float(gate.get("coverage_rear_bin_rad", 0.02)))

    # solver bounds — match solve_pacejka.py:84
    lo, hi = [1.0, 0.1, 0.1, 0.0], [20.0, 20.0, 20.0, 5.0]
    def _bound_vio(p):
        return any(abs(v - lo[i]) < 0.01 * max(1.0, hi[i]) or
                   abs(hi[i] - v) < 0.01 * max(1.0, hi[i]) for i, v in enumerate(p))
    bvio = _bound_vio(C_Pf) or _bound_vio(C_Pr)

    m, l_f, l_wb = model["m"], model["l_f"], model["l_wb"]
    mu_est = C_Pr[2] / (m * 9.81 * l_f / l_wb)

    prior_Df = float(prior_yaml.get("C_Pf", [0, 0, 1.0, 0])[2]) or 1e-6
    df_err = abs(C_Pf[2] - prior_Df) / prior_Df

    metrics = dict(
        rmse_front=rmse_f, rmse_rear=rmse_r,
        coverage_front_bins=cov_f, coverage_rear_bins=cov_r,
        bound_violation=bvio, mu=mu_est, df_err=df_err,
    )

    accept = (
        rmse_f <= gate["rmse_front_max"] and
        rmse_r <= gate["rmse_rear_max"] and
        cov_f >= gate["coverage_front_bins_min"] and
        cov_r >= gate["coverage_rear_bins_min"] and
        (not bvio or gate.get("allow_bound_violation", False)) and
        gate["mu_min"] <= mu_est <= gate["mu_max"] and
        df_err <= gate["df_sanity_frac"]
    )
    return bool(accept), metrics
