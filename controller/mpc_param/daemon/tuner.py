#!/usr/bin/env python3
"""tuner.py — Phase 4: daemon main.

Wires together: ChannelHub → metrics → rules → patcher.
State machine: WARMUP → ACTIVE → (FROZEN | PAUSED).

Run inside the icra2026 container:
    rosrun mpc_param tuner.py
or
    python3 controller/mpc_param/daemon/tuner.py \
        --target upenn_mpc \
        --yaml   controller/upenn_mpc/config/upenn_mpc_srx1.yaml \
        --rules  controller/mpc_param/rules/upenn_mpc.yaml

Ctrl-C → graceful shutdown. Last state stays applied.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import rospy
import yaml

# Allow `from daemon.X import Y` when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # mpc_param/

from daemon.channels import ChannelHub, LapEvent
from daemon.metrics  import (
    auto_sectors_by_curvature, load_sector_override,
    compute_lap_metrics, trail_row, TRAIL_HEADER,
)
from daemon.rules    import load_ruleset, evaluate, updates_to_diff
from daemon.patcher  import YamlPatcher, PatchSnapshot


# Default forbidden keys (mirror mpc_param.yaml). Defense in depth.
DEFAULT_FORBIDDEN = {
    "friction_margin", "friction_circle",
    "N_horizon", "dt", "loop_rate_hz",
    "window_size", "codegen_dir", "line_source", "mu_source",
    "startup_delay_s", "warmup_vx_min", "warmup_speed_cmd",
    "warmup_exit_vx", "crash_stuck_sec", "reset_jump_thres_m",
    "stuck_status_thres", "test_mode",
}


@dataclass
class TunerState:
    state: str = "WARMUP"        # WARMUP | ACTIVE | FROZEN | PAUSED
    laps_seen: int = 0
    laps_in_active: int = 0

    # Convergence tracking.
    consecutive_low_only: int = 0
    last_lap_time: Optional[float] = None
    last_primary_metric: Optional[float] = None

    # Rollback.
    last_snapshot: Optional[PatchSnapshot] = None
    last_applied_keys: List[str] = field(default_factory=list)
    just_rolled_back: bool = False

    # Cooldown: per-key consecutive-change streak.
    key_streak: Dict[str, int] = field(default_factory=dict)
    cooldown_keys: Dict[str, int] = field(default_factory=dict)   # key → laps remaining


class Tuner:
    def __init__(self, target: str, yaml_path: str, rules_path: str,
                 trail_csv: str,
                 lap_topic: str = "lap_data",
                 warmup_laps: int = 2,
                 rollback_threshold: float = 0.03,
                 rollback_lambda: float = 0.5,
                 cooldown_streak: int = 3,
                 cooldown_observe_laps: int = 1,
                 convergence_consecutive: int = 2,
                 convergence_lap_time_eps: float = 0.01,
                 crash_infeasible_max: int = 50,
                 sectors_yaml: Optional[str] = None,
                 mu_for_friction: float = 0.85,
                 forbidden_extra: Optional[set] = None):
        self.target = target
        self.yaml_path = yaml_path
        self.rules_path = rules_path
        self.trail_csv = trail_csv
        self.warmup_laps = warmup_laps
        self.rollback_threshold = rollback_threshold
        self.rollback_lambda = rollback_lambda
        self.cooldown_streak = cooldown_streak
        self.cooldown_observe_laps = cooldown_observe_laps
        self.conv_consecutive = convergence_consecutive
        self.conv_lap_time_eps = convergence_lap_time_eps
        self.crash_infeasible_max = crash_infeasible_max
        self.mu_for_friction = mu_for_friction
        self.forbidden = DEFAULT_FORBIDDEN | (forbidden_extra or set())

        # Components.
        self.hub = ChannelHub(target=target, lap_topic=lap_topic)
        self.policy, self.rules = load_ruleset(rules_path)
        self.patcher = YamlPatcher(
            yaml_path=yaml_path,
            namespace=f"/{target}",
            reload_service=f"/{target}/reload_params",
        )
        # Initial yaml values.
        self.current_yaml: Dict[str, float] = self._load_current_yaml()

        self.state = TunerState()
        self.sectors_override = (load_sector_override(yaml.safe_load(open(sectors_yaml)))
                                 if sectors_yaml and os.path.isfile(sectors_yaml) else None)
        self._lap_lock = threading.Lock()
        self.hub.set_lap_callback(self._on_lap)
        self._csv_init()
        rospy.loginfo(f"[mpc_param.tuner] init done. target={target} state={self.state.state}")

    # ----------------------------------------------------------------------
    # Public lifecycle
    # ----------------------------------------------------------------------
    def spin(self) -> None:
        # ROS spin handles callbacks. Sleep loop just keeps process alive.
        try:
            rospy.spin()
        except (rospy.ROSInterruptException, KeyboardInterrupt):
            pass
        finally:
            rospy.loginfo(f"[mpc_param.tuner] shutdown. final state={self.state.state} "
                          f"laps={self.state.laps_seen}")

    # ----------------------------------------------------------------------
    # Lap event entry point
    # ----------------------------------------------------------------------
    def _on_lap(self, evt: LapEvent) -> None:
        with self._lap_lock:
            self._handle_lap(evt)

    def _handle_lap(self, evt: LapEvent) -> None:
        self.state.laps_seen += 1
        rospy.loginfo(f"[mpc_param.tuner] lap #{evt.lap_count} t={evt.lap_time:.3f}s "
                      f"window=[{evt.t_start:.2f},{evt.t_end:.2f}] state={self.state.state}")

        # Cooldown countdown (decrement BEFORE this lap's evaluation).
        for k in list(self.state.cooldown_keys.keys()):
            self.state.cooldown_keys[k] -= 1
            if self.state.cooldown_keys[k] <= 0:
                del self.state.cooldown_keys[k]

        # Compute metrics.
        snap = self.hub.snapshot(evt.t_start, evt.t_end)
        raceline = self.hub.raceline()
        if raceline is None:
            rospy.logwarn("[mpc_param.tuner] no raceline cached yet; skip lap")
            return
        sectors = self.sectors_override or auto_sectors_by_curvature(raceline)
        vp = {"l_f": float(rospy.get_param("/vehicle/l_f", 0.162)),
              "l_r": float(rospy.get_param("/vehicle/l_r", 0.145))}
        lap_event_dict = {
            "lap_count": evt.lap_count,
            "lap_time":  evt.lap_time,
            "avg_lat_err": evt.avg_lat_err,
            "max_lat_err": evt.max_lat_err,
            "t_start": evt.t_start,
            "t_end":   evt.t_end,
        }
        metrics = compute_lap_metrics(snap, raceline, vp, sectors, lap_event_dict,
                                      mu_for_friction=self.mu_for_friction)
        lap = metrics["lap"]
        primary = float(lap.get("lap_time", 0.0)
                        + self.rollback_lambda * lap.get("infeasible_count", 0))

        applied_diff: Dict[str, dict] = {}

        # Crash guard FIRST.
        if lap.get("infeasible_count", 0) > self.crash_infeasible_max:
            self._do_rollback("crash_guard", lap)
            self.state.state = "PAUSED"
            self._write_csv(metrics, applied_diff)
            return

        # State transitions.
        if self.state.state == "WARMUP":
            if self.state.laps_seen >= self.warmup_laps:
                rospy.loginfo("[mpc_param.tuner] WARMUP done → ACTIVE")
                self.state.state = "ACTIVE"
            self._update_baselines(lap, primary)
            self._write_csv(metrics, applied_diff)
            return

        if self.state.state == "PAUSED":
            rospy.logwarn("[mpc_param.tuner] PAUSED — observation only")
            self._write_csv(metrics, applied_diff)
            self._update_baselines(lap, primary)
            return

        # FROZEN: only output recommendations, don't apply.
        if self.state.state == "FROZEN":
            fired, updates, dbg = evaluate(metrics, self.policy, self.rules,
                                           self.current_yaml, self.forbidden)
            if updates:
                rospy.loginfo(f"[mpc_param.tuner] FROZEN — would apply: "
                              f"{updates_to_diff(updates)}")
            self._write_csv(metrics, applied_diff)
            self._check_unfreeze(lap, primary)
            return

        # ACTIVE: rollback check first (worsening?).
        if (self.state.last_primary_metric is not None
                and self.state.last_snapshot is not None
                and primary > self.state.last_primary_metric * (1.0 + self.rollback_threshold)):
            rospy.logwarn(f"[mpc_param.tuner] worsened {self.state.last_primary_metric:.3f}"
                          f" → {primary:.3f} (>{self.rollback_threshold*100:.0f}%) — ROLLBACK")
            self._do_rollback(f"worsened lap#{evt.lap_count}", lap)
            self.state.state = "ACTIVE"   # stay active but with cooldown
            self._write_csv(metrics, applied_diff)
            return

        # Evaluate rules + apply.
        fired, updates, dbg = evaluate(metrics, self.policy, self.rules,
                                       self.current_yaml, self.forbidden)
        # Filter out cooldown'd keys.
        usable_updates = [u for u in updates if u.key not in self.state.cooldown_keys]
        cooled = [u.key for u in updates if u.key in self.state.cooldown_keys]
        if cooled:
            rospy.loginfo(f"[mpc_param.tuner] cooldown skipped: {cooled}")

        if usable_updates:
            kv = {u.key: u.new_value for u in usable_updates}
            commit = (f"mpc_param: lap#{evt.lap_count} apply "
                      + ",".join(u.key for u in usable_updates))
            result = self.patcher.apply(kv, commit_msg=commit)
            if result.success and result.applied:
                self.current_yaml.update({k: v[1] for k, v in result.applied.items()})
                self.state.last_snapshot = result.snapshot
                self.state.last_applied_keys = list(result.applied.keys())
                applied_diff = updates_to_diff(usable_updates)
                # Cooldown bookkeeping.
                for k in result.applied:
                    self.state.key_streak[k] = self.state.key_streak.get(k, 0) + 1
                    if self.state.key_streak[k] >= self.cooldown_streak:
                        self.state.cooldown_keys[k] = self.cooldown_observe_laps
                        self.state.key_streak[k] = 0
                rospy.loginfo(f"[mpc_param.tuner] applied: {applied_diff} "
                              f"git={result.git_sha}")
            else:
                rospy.logerr(f"[mpc_param.tuner] patch failed: {result.reload_message}")
                # Service failed → pause to be safe.
                self.state.state = "PAUSED"
        else:
            # No updates this lap → reset key streaks (stale).
            self.state.key_streak.clear()

        # Convergence tracking.
        max_sev = max((f.rule.severity for f in fired), default="LOW")
        if max_sev == "LOW":
            self.state.consecutive_low_only += 1
        else:
            self.state.consecutive_low_only = 0

        lap_time_delta_pct = (
            abs(lap["lap_time"] - self.state.last_lap_time) / self.state.last_lap_time
            if self.state.last_lap_time else 1.0
        )
        if (self.state.consecutive_low_only >= self.conv_consecutive
                and lap_time_delta_pct < self.conv_lap_time_eps):
            rospy.loginfo("[mpc_param.tuner] CONVERGED → FROZEN")
            self.state.state = "FROZEN"

        self._update_baselines(lap, primary)
        self._write_csv(metrics, applied_diff)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _update_baselines(self, lap: dict, primary: float) -> None:
        self.state.last_lap_time = lap["lap_time"]
        self.state.last_primary_metric = primary

    def _do_rollback(self, reason: str, lap: dict) -> None:
        if self.state.last_snapshot is None:
            rospy.logwarn(f"[mpc_param.tuner] rollback requested ({reason}) "
                          "but no snapshot available")
            return
        rospy.logwarn(f"[mpc_param.tuner] ROLLBACK ({reason})")
        result = self.patcher.restore(self.state.last_snapshot,
                                      commit_msg=f"mpc_param: rollback {reason}")
        # Roll current_yaml back too.
        for k in self.state.last_applied_keys:
            old = self.state.last_snapshot.applied_diff[k]["old"]
            self.current_yaml[k] = old
        self.state.just_rolled_back = True
        self.state.last_snapshot = None     # invalidate; need fresh apply for next rollback
        self.state.last_applied_keys = []
        # Clear streaks for safety.
        self.state.key_streak.clear()

    def _check_unfreeze(self, lap: dict, primary: float) -> None:
        # If lap time degrades > 5% in FROZEN, go back to ACTIVE.
        if (self.state.last_primary_metric
                and primary > self.state.last_primary_metric * 1.05):
            rospy.loginfo("[mpc_param.tuner] FROZEN broken (degradation) → ACTIVE")
            self.state.state = "ACTIVE"
            self.state.consecutive_low_only = 0
        self._update_baselines(lap, primary)

    def _load_current_yaml(self) -> Dict[str, float]:
        """Read current yaml keys (top-level numeric ones under target ns)."""
        with open(self.yaml_path) as f:
            doc = yaml.safe_load(f) or {}
        ns_doc = doc.get(self.target, doc)   # accept flat or nested
        out = {}
        for k, v in ns_doc.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = float(v)
        return out

    def _csv_init(self) -> None:
        os.makedirs(os.path.dirname(self.trail_csv), exist_ok=True)
        if not os.path.exists(self.trail_csv):
            with open(self.trail_csv, "w", newline="") as f:
                csv.writer(f).writerow(TRAIL_HEADER + ["state"])

    def _write_csv(self, metrics: dict, applied_diff: dict) -> None:
        row = trail_row(metrics, applied_diff, ts=time.time()) + [self.state.state]
        with open(self.trail_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="upenn_mpc")
    ap.add_argument("--yaml", required=True, help="path to controller config yaml")
    ap.add_argument("--rules", required=True, help="path to rules yaml")
    ap.add_argument("--trail-csv", default=os.path.join(_HERE, "..", "logs",
                                                        "tune_trail.csv"))
    ap.add_argument("--lap-topic", default="lap_data")
    ap.add_argument("--sectors", default=None)
    ap.add_argument("--warmup-laps", type=int, default=2)
    ap.add_argument("--rollback-threshold", type=float, default=0.03)
    ap.add_argument("--rollback-lambda", type=float, default=0.5)
    args = ap.parse_args(rospy.myargv()[1:] if "rospy" in sys.modules else None)

    rospy.init_node("mpc_param_tuner", anonymous=False)

    t = Tuner(
        target=args.target,
        yaml_path=os.path.abspath(args.yaml),
        rules_path=os.path.abspath(args.rules),
        trail_csv=os.path.abspath(args.trail_csv),
        lap_topic=args.lap_topic,
        sectors_yaml=args.sectors,
        warmup_laps=args.warmup_laps,
        rollback_threshold=args.rollback_threshold,
        rollback_lambda=args.rollback_lambda,
    )
    t.spin()


if __name__ == "__main__":
    main()
