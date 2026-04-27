"""metrics.py — per-lap metric calculator.

Pure numpy. ROS-free. Testable offline.

Input: snapshot dict from ChannelHub (time-stamped arrays per channel)
       + raceline ndarray (M,11) + vehicle_params dict + sector_def (optional)

Output: nested dict
    {
      "lap":     {lap_time, mean_n, max_n, ay_usage_max, solve_p99_ms, ...},
      "sectors": [ {name,type,s_start,s_end, mean_n, max_n, dpsi_in, ...}, ... ],
      "corners": [ ... ],   # subset of sectors with type=='corner'
    }

Derived channels (computed inside):
    slip_f = atan2(vy + lf*omega, vx) - delta
    slip_r = atan2(vy - lr*omega, vx)
    ay     = vx*omega + d/dt(vy)        (or IMU.ay_imu fallback)
    ay_usage = |ay| / (mu*g)
    u_steer_rate = d/dt(delta_cmd)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

G = 9.81
C_X, C_Y, C_Z, C_VX, _, C_S, C_KAPPA, C_PSI, C_AX, C_D, C_MU_RAD = range(11)


# --------------------------------------------------------------------------
# Sector definition
# --------------------------------------------------------------------------
@dataclass
class Sector:
    name: str
    type: str            # "corner" | "straight"
    s_start: float
    s_end: float
    kappa_peak: float = 0.0


def auto_sectors_by_curvature(raceline: np.ndarray,
                              kappa_thr: float = 0.4,
                              min_len_m: float = 1.5) -> List[Sector]:
    """Curvature-based auto split. |κ| > thr → corner.

    Adjacent corner samples merged. Short segments (< min_len_m) absorbed
    into the surrounding type to avoid noise sectors.
    """
    s = raceline[:, C_S]
    k = np.abs(raceline[:, C_KAPPA])
    is_corner = k > kappa_thr

    sectors: List[Sector] = []
    n = len(s)
    if n == 0:
        return sectors

    i = 0
    cnt = 0
    while i < n:
        j = i
        cur = is_corner[i]
        while j < n and is_corner[j] == cur:
            j += 1
        s0 = float(s[i]); s1 = float(s[min(j, n - 1)])
        peak = float(np.max(k[i:j])) if cur else 0.0
        sectors.append(Sector(
            name=f"{'C' if cur else 'S'}{cnt:02d}",
            type="corner" if cur else "straight",
            s_start=s0, s_end=s1, kappa_peak=peak,
        ))
        cnt += 1
        i = j

    # Merge tiny segments into neighbors.
    merged: List[Sector] = []
    for sec in sectors:
        length = sec.s_end - sec.s_start
        if length < min_len_m and merged:
            merged[-1].s_end = sec.s_end
            merged[-1].kappa_peak = max(merged[-1].kappa_peak, sec.kappa_peak)
            # type kept as the (longer) prior segment
        else:
            merged.append(sec)
    # Renumber.
    out: List[Sector] = []
    cc = sc = 0
    for sec in merged:
        if sec.type == "corner":
            sec.name = f"C{cc:02d}"; cc += 1
        else:
            sec.name = f"S{sc:02d}"; sc += 1
        out.append(sec)
    return out


def load_sector_override(sectors_yaml: List[dict]) -> List[Sector]:
    """sectors/<map>.yaml format:
        - {name: C00, type: corner, s_start: 12.3, s_end: 18.7}
    """
    return [Sector(name=d["name"], type=d["type"],
                   s_start=float(d["s_start"]), s_end=float(d["s_end"]))
            for d in sectors_yaml]


# --------------------------------------------------------------------------
# Resampling + alignment
# --------------------------------------------------------------------------
def _resample(t: np.ndarray, v: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linear interp onto t_grid. Empty input → NaN."""
    if t.size == 0:
        return np.full_like(t_grid, np.nan, dtype=np.float64)
    return np.interp(t_grid, t, v, left=np.nan, right=np.nan)


def _common_grid(snapshot: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 dt: float = 0.02) -> np.ndarray:
    t_start = max((arr[0][0] for arr in snapshot.values() if arr[0].size), default=0.0)
    t_end = min((arr[0][-1] for arr in snapshot.values() if arr[0].size), default=0.0)
    if t_end <= t_start:
        return np.empty(0)
    n = int((t_end - t_start) / dt) + 1
    return t_start + np.arange(n) * dt


def _ddt(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    if t.size < 2:
        return np.zeros_like(v)
    dv = np.gradient(v, t)
    return dv


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
def compute_lap_metrics(snapshot: Dict[str, Tuple[np.ndarray, np.ndarray]],
                        raceline: np.ndarray,
                        vehicle_params: dict,
                        sectors: List[Sector],
                        lap_event: dict,
                        mu_for_friction: float = 0.85,
                        grid_dt: float = 0.02) -> dict:
    """Return per-lap + per-sector metric dict.

    `lap_event` carries lap_time, avg/max_lat_err from LapData.
    """
    grid = _common_grid(snapshot, dt=grid_dt)
    if grid.size < 10:
        return {"lap": {**lap_event, "valid": False, "reason": "too_few_samples"},
                "sectors": [], "corners": []}

    # Resample core channels onto common grid.
    R = lambda name: _resample(*snapshot[name], grid)
    vx = R("vx"); vy = R("vy"); yaw = R("yaw")
    s = R("s");   n = R("n")
    omega = R("omega"); ay_imu = R("ay_imu")
    delta = R("delta_cmd"); speed_cmd = R("speed_cmd")
    solve_ms = R("solve_ms")

    # Derived channels.
    lf = float(vehicle_params.get("l_f", 0.162))
    lr = float(vehicle_params.get("l_r", 0.145))
    # Slip angles. delta may have NaN early; ignore.
    with np.errstate(invalid="ignore"):
        slip_f = np.arctan2(vy + lf * omega, np.maximum(vx, 0.05)) - delta
        slip_r = np.arctan2(vy - lr * omega, np.maximum(vx, 0.05))
    # Lateral accel: prefer model-based vx*omega (clean) + IMU as fallback.
    ay_model = vx * omega
    ay = np.where(np.isnan(ay_model), ay_imu, ay_model)
    ay_usage = np.abs(ay) / (mu_for_friction * G)
    # Steering rate (cmd-side, captures controller chatter).
    u_steer_rate = _ddt(grid, np.where(np.isnan(delta), 0.0, delta))

    # Infeasible events: count of '1' samples in window (Phase 3 wires source).
    inf_t, _ = snapshot["infeasible"]
    infeasible_count = int(np.sum((inf_t >= grid[0]) & (inf_t <= grid[-1])))

    # ---- LAP-level aggregates ----
    lap = {
        **lap_event,
        "valid": True,
        "samples": int(grid.size),
        "duration_s": float(grid[-1] - grid[0]),
        "mean_abs_n": _nan_safe(np.nanmean, np.abs(n)),
        "max_abs_n": _nan_safe(np.nanmax, np.abs(n)),
        "p95_abs_n": _nan_safe(np.nanpercentile, np.abs(n), 95),
        "max_ay_usage": _nan_safe(np.nanmax, ay_usage),
        "mean_ay_usage": _nan_safe(np.nanmean, ay_usage),
        "max_abs_slip_f": _nan_safe(np.nanmax, np.abs(slip_f)),
        "max_abs_slip_r": _nan_safe(np.nanmax, np.abs(slip_r)),
        "u_steer_rate_rms": _nan_safe(_rms, u_steer_rate),
        "solve_p50_ms": _nan_safe(np.nanpercentile, solve_ms, 50),
        "solve_p99_ms": _nan_safe(np.nanpercentile, solve_ms, 99),
        "vx_mean": _nan_safe(np.nanmean, vx),
        "vx_max":  _nan_safe(np.nanmax,  vx),
        "infeasible_count": infeasible_count,
        "omega_oscillation_hz": _peak_freq(grid, np.where(np.isnan(omega), 0.0, omega)),
    }

    # ---- SECTOR-level ----
    sector_metrics: List[dict] = []
    for sec in sectors:
        mask = _sector_mask(s, sec, raceline_len=float(raceline[-1, C_S]))
        if not np.any(mask):
            continue
        ms = _sector_aggregate(sec, mask, n, ay_usage, slip_f, slip_r,
                               omega, delta, vx, u_steer_rate)
        sector_metrics.append(ms)

    corners = [m for m in sector_metrics if m["type"] == "corner"]
    return {"lap": lap, "sectors": sector_metrics, "corners": corners}


def _sector_mask(s: np.ndarray, sec: Sector, raceline_len: float) -> np.ndarray:
    """Boolean mask of samples whose s falls inside the sector.
    Handles s wrap-around (s_end < s_start crossing the start line).
    """
    s = np.where(np.isnan(s), -1.0, s)
    if sec.s_end >= sec.s_start:
        return (s >= sec.s_start) & (s <= sec.s_end)
    # wrap
    return (s >= sec.s_start) | (s <= sec.s_end)


def _sector_aggregate(sec: Sector, mask: np.ndarray,
                      n, ay_usage, slip_f, slip_r,
                      omega, delta, vx, u_steer_rate) -> dict:
    nm = n[mask]; ayu = ay_usage[mask]
    sf = slip_f[mask]; sr = slip_r[mask]
    om = omega[mask]; dl = delta[mask]; vxm = vx[mask]
    usr = u_steer_rate[mask]

    # For corners: split into entry / mid / exit thirds.
    third = max(1, len(nm) // 3)
    n_entry = nm[:third]; n_exit = nm[-third:]

    return {
        "name": sec.name,
        "type": sec.type,
        "s_start": sec.s_start,
        "s_end": sec.s_end,
        "kappa_peak": sec.kappa_peak,
        "samples": int(mask.sum()),
        "mean_abs_n": _nan_safe(np.nanmean, np.abs(nm)),
        "max_abs_n": _nan_safe(np.nanmax, np.abs(nm)),
        "n_entry_signed_mean": _nan_safe(np.nanmean, n_entry),
        "n_exit_signed_mean": _nan_safe(np.nanmean, n_exit),
        "max_n_exit_abs": _nan_safe(np.nanmax, np.abs(n_exit)),
        "max_ay_usage": _nan_safe(np.nanmax, ayu),
        "mean_ay_usage": _nan_safe(np.nanmean, ayu),
        "max_abs_slip_f": _nan_safe(np.nanmax, np.abs(sf)),
        "max_abs_slip_r": _nan_safe(np.nanmax, np.abs(sr)),
        "omega_rms": _nan_safe(_rms, om),
        "vx_mean": _nan_safe(np.nanmean, vxm),
        "u_steer_rate_rms": _nan_safe(_rms, usr),
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _nan_safe(fn, *args, **kwargs) -> float:
    try:
        v = fn(*args, **kwargs)
        v = float(v)
        return v if np.isfinite(v) else 0.0
    except (ValueError, IndexError):
        return 0.0


def _rms(arr: np.ndarray) -> float:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))


def _peak_freq(t: np.ndarray, v: np.ndarray) -> float:
    """Dominant frequency of v (Hz) via FFT. Returns 0 on degenerate input."""
    if t.size < 16:
        return 0.0
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return 0.0
    v = v - np.mean(v)
    spec = np.abs(np.fft.rfft(v))
    if spec.size < 2:
        return 0.0
    spec[0] = 0.0   # drop DC
    freqs = np.fft.rfftfreq(t.size, d=dt)
    return float(freqs[int(np.argmax(spec))])


# --------------------------------------------------------------------------
# CSV trail writer (per-lap one row)
# --------------------------------------------------------------------------
TRAIL_HEADER = [
    "ts", "lap_count", "lap_time", "valid", "infeasible_count",
    "mean_abs_n", "max_abs_n", "p95_abs_n",
    "max_ay_usage", "mean_ay_usage",
    "max_abs_slip_f", "max_abs_slip_r",
    "u_steer_rate_rms", "omega_oscillation_hz",
    "solve_p50_ms", "solve_p99_ms",
    "vx_mean", "vx_max",
    "applied_diff_json",
]


def trail_row(metrics: dict, applied_diff: dict, ts: float) -> List:
    lap = metrics.get("lap", {})
    import json
    return [
        f"{ts:.3f}",
        lap.get("lap_count", -1),
        f"{lap.get('lap_time', 0.0):.4f}",
        int(lap.get("valid", False)),
        lap.get("infeasible_count", 0),
        f"{lap.get('mean_abs_n', 0.0):.4f}",
        f"{lap.get('max_abs_n', 0.0):.4f}",
        f"{lap.get('p95_abs_n', 0.0):.4f}",
        f"{lap.get('max_ay_usage', 0.0):.3f}",
        f"{lap.get('mean_ay_usage', 0.0):.3f}",
        f"{lap.get('max_abs_slip_f', 0.0):.4f}",
        f"{lap.get('max_abs_slip_r', 0.0):.4f}",
        f"{lap.get('u_steer_rate_rms', 0.0):.4f}",
        f"{lap.get('omega_oscillation_hz', 0.0):.2f}",
        f"{lap.get('solve_p50_ms', 0.0):.2f}",
        f"{lap.get('solve_p99_ms', 0.0):.2f}",
        f"{lap.get('vx_mean', 0.0):.3f}",
        f"{lap.get('vx_max', 0.0):.3f}",
        json.dumps(applied_diff, separators=(",", ":")),
    ]
