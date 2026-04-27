#!/usr/bin/env python3
"""Smoke test for rules.py — synthesizes 4 lap scenarios and checks priority."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from daemon.rules import (
    load_ruleset, evaluate, updates_to_diff, PRIORITY_LEVELS,
)


RULES_YAML = os.path.join(os.path.dirname(_HERE), "rules", "upenn_mpc.yaml")


CURRENT_YAML = {
    # mirror upenn_mpc_srx1.yaml defaults relevant to rule actions
    "w_d": 1.5, "w_dpsi": 5.5, "w_vx": 1.5, "w_vy": 0.3, "w_omega": 2.0,
    "w_steer": 0.3, "w_u_steer_rate": 1.0, "w_u_accel": 0.3,
    "w_terminal_scale": 1.5,
    "friction_slack_penalty": 1000.0,
    "speed_boost": 1.0,
}

FORBIDDEN = {"friction_margin", "friction_circle", "N_horizon", "dt"}


def make_metrics(**overrides):
    base_lap = {
        "lap_count": 5, "lap_time": 18.0, "valid": True,
        "infeasible_count": 0,
        "mean_abs_n": 0.05, "max_abs_n": 0.15, "p95_abs_n": 0.10,
        "max_ay_usage": 0.6, "mean_ay_usage": 0.3,
        "max_abs_slip_f": 0.10, "max_abs_slip_r": 0.08,
        "u_steer_rate_rms": 0.5, "omega_oscillation_hz": 0.5,
        "solve_p50_ms": 4.0, "solve_p99_ms": 8.0,
        "vx_mean": 5.5, "vx_max": 8.5,
    }
    base_lap.update(overrides.get("lap", {}))
    corners = overrides.get("corners", [])
    return {"lap": base_lap, "sectors": corners, "corners": corners}


def case(name, metrics):
    print(f"\n=== {name} ===")
    policy, rules = load_ruleset(RULES_YAML)
    fired, updates, dbg = evaluate(metrics, policy, rules, CURRENT_YAML, FORBIDDEN)
    print(f"  top level: {dbg['top_level']}  fired: {dbg['fired_count']}  "
          f"suppressed: {dbg['suppressed_count']} {dbg['suppressed_rules']}")
    for f in fired:
        print(f"  + {f.rule.priority:<13} {f.rule.severity:<4} {f.rule.name}")
    diff = updates_to_diff(updates)
    for k, v in diff.items():
        print(f"      {k:>22}: {v['old']:.4f} -> {v['new']:.4f}  "
              f"x{v['mult']:.3f}  rules={v['rules']}  clamped={v['clamped']}")
    if dbg["rejected_actions"]:
        print(f"  rejected: {dbg['rejected_actions']}")
    if dbg["eval_errors"]:
        print(f"  EVAL ERRORS: {dbg['eval_errors']}")


def main():
    # Case 1: clean lap, nothing fires.
    case("clean lap (nothing should fire)",
         make_metrics())

    # Case 2: corner exit understeer + speed headroom (L1 should suppress L3).
    corners = [{
        "name": "C00", "type": "corner",
        "n_exit_signed_mean": 0.40, "n_entry_signed_mean": 0.05,
        "max_ay_usage": 0.65, "max_abs_slip_r": 0.10, "max_abs_slip_f": 0.10,
    }]
    case("L1 understeer + L3 headroom (L3 must be suppressed)",
         make_metrics(lap={"max_ay_usage": 0.65, "mean_abs_n": 0.10,
                           "infeasible_count": 0},
                      corners=corners))

    # Case 3: SAFETY infeasible — L0 should win, lower suppressed.
    case("L0 infeasible (must override everything)",
         make_metrics(lap={"infeasible_count": 12, "max_abs_slip_r": 0.30,
                           "max_ay_usage": 1.10, "mean_abs_n": 0.4,
                           "max_abs_n": 0.6},
                      corners=corners))

    # Case 4: pure speed headroom (only L3 fires).
    case("pure L3 headroom",
         make_metrics(lap={"max_ay_usage": 0.55, "mean_abs_n": 0.08,
                           "infeasible_count": 0}))

    # Case 5: yaw oscillation — L2 only.
    case("pure L2 yaw oscillation",
         make_metrics(lap={"omega_oscillation_hz": 3.5,
                           "infeasible_count": 0,
                           "max_ay_usage": 0.6, "mean_abs_n": 0.10,
                           "max_abs_slip_r": 0.05}))

    # Case 6: forbidden key in current_yaml -- shouldn't be matched (test rejection)
    case("steer chatter only (L4)",
         make_metrics(lap={"u_steer_rate_rms": 2.5,
                           "infeasible_count": 0,
                           "max_ay_usage": 0.6, "mean_abs_n": 0.10}))


if __name__ == "__main__":
    main()
