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
### HJ : friction_margin 빼냄. crash_detected_* 룰들이 이 키를 만지는데
###      forbidden 이라 silent skip → crash 보수 효과 안 남. 룰의 clamp [0.3,1.0]
###      이 충분히 안전 범위 (mu·g·factor) 라 daemon tuning 허용.
###
### HJ Phase 2C : mpcc_dyna 의 codegen-baked 키 보호. 이 값들은 acados 의 cost
### expression 에 Python float 으로 baked 되어 runtime cost_set/constraints_set
### 으론 못 바꾼다. yaml 만 수정하면 노드 belief 와 solver 행동 mismatch.
### 변경하려면 노드 restart + /tmp/<target>_codegen 삭제 필요.
###   - friction_margin           : Pacejka peak 사용 비율 (h-constraint 식)
###   - Pacejka B/C/D/E (f/r)     : 타이어 stiffness/peak/curvature/shape
###   - 차량 기하 / 질량 (m, l_f, l_r, l_wb, I_z, h_cg, mu) : ODE 계수
DEFAULT_FORBIDDEN = {
    "friction_circle",
    "N_horizon", "dt", "loop_rate_hz",
    "window_size", "codegen_dir", "line_source", "mu_source",
    "startup_delay_s", "warmup_vx_min", "warmup_speed_cmd",
    "warmup_exit_vx", "crash_stuck_sec", "reset_jump_thres_m",
    "stuck_status_thres", "test_mode",
    # Phase 2C : mpcc_dyna codegen-baked
    "friction_margin",
    "Bf", "Cf", "Df", "Ef", "Br", "Cr", "Dr", "Er",
    "m", "l_f", "l_r", "l_wb", "I_z", "h_cg", "mu",
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
    ### HJ : best lap_time 추적 (crash 없는 lap 만 인정). lap_time_regression
    ###       rule 의 baseline.
    best_lap_time: Optional[float] = None
    ### HJ : Panic recovery — 연속 watchdog/spawn 누적 시 last_good yaml 복구
    consecutive_watchdog: int = 0          # 연속 watchdog timeout 카운트
    panic_cooldown_remaining: int = 0      # PANIC 후 rule 미적용 lap 수
    last_good_yaml_path: Optional[str] = None   # 마지막 best 갱신 시 yaml 백업 경로

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
        ### HJ : spawn (re-teleport) 즉각 인식 callback 등록.
        self.hub.set_spawn_callback(self._on_spawn)
        self._csv_init()
        ### HJ : spawn 즉각 escalation. 같은 cluster 묶음 + 다중 spawn 카운트.
        ###       spawn = 1.5s 정지 = 명확한 stuck 신호. 60s watchdog 안 기다림.
        self._spawn_cooldown_s: float = 2.0  # 같은 cluster 2초 내 1회만
        self._last_spawn_handled_t: float = 0.0
        self._spawn_event_history: list = []  # (t,) — 30초 window 내 모든 spawn
        self._spawn_window_s: float = 30.0
        self._spawn_panic_lv1_count: int = 2  # 2 spawn / 30s → PANIC LV1
        self._spawn_panic_lv2_count: int = 3  # 3 spawn / 30s → PANIC LV2

        ### HJ : Lap watchdog — 차가 lap 못 돌고 stuck 일 때 timeout 후 synthetic
        ###       failed-lap 이벤트 발생 (crash_count=999 로 force rollback).
        ###       MPCC 처럼 처음에 한 바퀴 못 도는 setup 도 데몬 작동시키기 위함.
        self.lap_watchdog_timeout_s = float(rospy.get_param(
            "~lap_watchdog_timeout_s", 60.0))
        self._last_lap_wallt: float = rospy.Time.now().to_sec()
        rospy.Timer(rospy.Duration(5.0), self._lap_watchdog_cb)
        ### HJ : Panic recovery 설정
        self.panic_watchdog_threshold = int(rospy.get_param(
            "~panic_watchdog_threshold", 3))    # 연속 watchdog N개 → PANIC
        self.panic_cooldown_laps = int(rospy.get_param(
            "~panic_cooldown_laps", 5))         # PANIC 후 N lap 동안 rule 미적용
        # 데몬 시작 시점 yaml 을 last_good 으로 저장 (첫 best 갱신까지 fallback).
        self._init_baseline_backup()
        ### HJ : artifact lap 필터. spawn_on_waypoint 가 차를 respawn 하면
        ###       lap_analyser 가 s wrap-around 로 fake "lap" 생성 (보통 5~10s).
        ###       이걸 진짜 lap 으로 학습하면 best_lap_time 이 5s 로 망가져
        ###       정상 lap 들이 worsen rollback 됨. min_real_lap_s 미만은 fake.
        self.min_real_lap_s = float(rospy.get_param("~min_real_lap_s", 15.0))
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
            ### HJ : watchdog reset 은 artifact 가 아닌 진짜 lap 일 때만.
            ###       artifact (fake spawn-wrap lap) 가 watchdog reset 시키면
            ###       차 stuck 중에도 영원히 timeout 안 발화 → 데몬 침묵.
            is_artifact = (evt.lap_count >= 0 and
                           evt.lap_time < self.min_real_lap_s)
            if not is_artifact:
                self._last_lap_wallt = rospy.Time.now().to_sec()
            self._handle_lap(evt)

    def _on_spawn(self, t_now: float) -> None:
        ### HJ : spawn 즉각 escalation. 30s window 내 spawn 카운트로 PANIC trigger.
        ###       cluster cooldown 2s (한 번 stuck 에 multi-publish 1회로).
        if (t_now - self._last_spawn_handled_t) < self._spawn_cooldown_s:
            return
        self._last_spawn_handled_t = t_now
        # 30s window 정리 + 추가
        self._spawn_event_history = [t for t in self._spawn_event_history
                                     if t_now - t < self._spawn_window_s]
        self._spawn_event_history.append(t_now)
        spawn_n = len(self._spawn_event_history)
        rospy.logwarn(f"[mpc_param.tuner] SPAWN #{spawn_n} (window {self._spawn_window_s:.0f}s)")
        # WARMUP/PAUSED/FROZEN — escalation 없이 카운트만 (정상 진단 path)
        if self.state.state != "ACTIVE":
            return
        with self._lap_lock:
            # spawn 카운트 escalation
            if spawn_n >= self._spawn_panic_lv2_count:
                rospy.logerr(f"[mpc_param.tuner] SPAWN PANIC LV2! "
                             f"{spawn_n} spawn in {self._spawn_window_s:.0f}s → EMERGENCY")
                self._emergency_safe_defaults()
                self._reset_panic_state()
                self._spawn_event_history.clear()
                return
            if spawn_n >= self._spawn_panic_lv1_count:
                if self.state.best_lap_time is not None:
                    rospy.logerr(f"[mpc_param.tuner] SPAWN PANIC LV1! "
                                 f"{spawn_n} spawn → restoring LAST_GOOD")
                    self._restore_last_good()
                else:
                    rospy.logerr(f"[mpc_param.tuner] SPAWN PANIC LV1 fallback (no best) → LV2")
                    self._emergency_safe_defaults()
                self._reset_panic_state()
                self._spawn_event_history.clear()
                return
            # 1번째 spawn — 진단 + rollback
            t0 = max(t_now - 5.0, 0.0)
            snap = self.hub.snapshot(t0, t_now)
            diagnosis, action_keys = self._diagnose_spawn(snap)
            rospy.logwarn(f"[mpc_param.tuner] spawn 진단: {diagnosis}")
            if self.state.last_snapshot is not None:
                self._do_rollback(f"spawn ({diagnosis})", lap={"lap_time": 0.0})
                for k in self.state.last_applied_keys:
                    self.state.cooldown_keys[k] = max(self.state.cooldown_keys.get(k, 0), 3)
            else:
                rospy.logwarn(f"[mpc_param.tuner] spawn 1회 — rollback snapshot 없음")

    def _reset_panic_state(self) -> None:
        ### HJ : PANIC 후 공통 reset. spawn-driven / watchdog-driven 둘 다 사용.
        self.state.consecutive_watchdog = 0
        self.state.panic_cooldown_remaining = self.panic_cooldown_laps
        self.state.last_snapshot = None
        self.state.last_applied_keys = []
        self.state.cooldown_keys.clear()
        self._last_lap_wallt = rospy.Time.now().to_sec()

    def _diagnose_spawn(self, snap: Dict) -> tuple:
        ### HJ : spawn 직전 metric 으로 실패 모드 분류.
        ###       - corner 진입에서 high vx + slip → progress 과대
        ###       - corner 안쪽 → premature inner cut
        ###       - 직선 stuck (vx≈0, kappa<0.05) → solver 발산 / w_d 과대
        try:
            vx_t, vx_v = snap.get("vx", (np.empty(0), np.empty(0)))
            n_t, n_v = snap.get("n", (np.empty(0), np.empty(0)))
            om_t, om_v = snap.get("omega", (np.empty(0), np.empty(0)))
            if vx_v.size == 0:
                return "no_data", []
            vx_max = float(np.nanmax(vx_v))
            vx_mean = float(np.nanmean(vx_v))
            n_abs_max = float(np.nanmax(np.abs(n_v))) if n_v.size else 0.0
            om_abs_max = float(np.nanmax(np.abs(om_v))) if om_v.size else 0.0
            if vx_max < 0.5:
                return "stuck_low_vx", ["w_d", "v_max"]
            if om_abs_max > 1.5 and vx_max > 2.0:
                return "high_yaw_in_corner", ["w_progress", "v_max"]
            if n_abs_max > 0.4:
                return "off_path_drift", ["w_d", "w_dpsi"]
            return "unknown_failure", ["v_max"]
        except Exception as e:
            rospy.logwarn(f"[mpc_param.tuner] _diagnose_spawn error: {e}")
            return "diag_error", []

    def _lap_watchdog_cb(self, _evt) -> None:
        ### HJ : 차가 lap 못 돌고 timeout 시 synthetic failed-lap.
        ###       Lock 으로 state 변경 보호. 3단 escalation: rollback → LV1 → LV2.
        elapsed = rospy.Time.now().to_sec() - self._last_lap_wallt
        if elapsed < self.lap_watchdog_timeout_s:
            return
        with self._lap_lock:
            self.state.consecutive_watchdog += 1
            rospy.logwarn(
                f"[mpc_param.tuner] lap WATCHDOG TIMEOUT ({elapsed:.1f}s, "
                f"consecutive={self.state.consecutive_watchdog})")
            if self.state.consecutive_watchdog >= self.panic_watchdog_threshold:
                self._trigger_panic()
                return
            # 단일 watchdog: synthetic crash event (rollback 시도)
            now = rospy.Time.now().to_sec()
            synth = LapEvent(
                lap_count=-1,
                lap_time=float(self.lap_watchdog_timeout_s),
                avg_lat_err=0.0, max_lat_err=0.0,
                t_start=self._last_lap_wallt, t_end=now,
                crash_count=999,
            )
            self._last_lap_wallt = now
            self._handle_lap(synth)

    def _trigger_panic(self) -> None:
        ### HJ : 3단 escalation:
        ###   - PANIC_COOLDOWN 중 또 발화 → LV2 (EMERGENCY) 강제. 무한 cycle 방지.
        ###   - best 갱신 있음 → LV1 (last_good)
        ###   - best 없음 → LV2 (EMERGENCY SAFE DEFAULTS)
        already_in_cooldown = self.state.panic_cooldown_remaining > 0
        has_good_baseline = self.state.best_lap_time is not None
        if already_in_cooldown:
            rospy.logerr(f"[mpc_param.tuner] PANIC LV2! cooldown 중 또 발화 → EMERGENCY ESCALATION")
            self._emergency_safe_defaults()
        elif has_good_baseline:
            rospy.logerr(f"[mpc_param.tuner] PANIC LV1! restoring LAST_GOOD "
                         f"(best={self.state.best_lap_time:.2f}s)")
            ok = self._restore_last_good()
            if not ok:
                rospy.logerr(f"[mpc_param.tuner] LV1 restore 실패 → LV2 fallback")
                self._emergency_safe_defaults()
        else:
            rospy.logerr(f"[mpc_param.tuner] PANIC LV2! best 미달 → EMERGENCY SAFE DEFAULTS")
            self._emergency_safe_defaults()
        self._reset_panic_state()
        self._spawn_event_history.clear()

    def _handle_lap(self, evt: LapEvent) -> None:
        ### HJ : artifact lap 필터. lap_time 이 너무 짧으면 (보통 5~10s) spawn-on-
        ###       waypoint 의 respawn 으로 인한 lap_analyser fake trigger. 진짜 lap
        ###       아님 → 무시 (laps_seen 카운트 안 함, 메트릭 처리 안 함).
        if evt.lap_count >= 0 and evt.lap_time < self.min_real_lap_s:
            rospy.logwarn(f"[mpc_param.tuner] ARTIFACT LAP 무시 (lap_time={evt.lap_time:.2f}s "
                          f"< {self.min_real_lap_s}s, lap#{evt.lap_count}). spawn 또는 sim reset 추정.")
            return
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
            "crash_count": evt.crash_count,   ### HJ : 충돌 카운트 전달
        }
        metrics = compute_lap_metrics(snap, raceline, vp, sectors, lap_event_dict,
                                      mu_for_friction=self.mu_for_friction)
        lap = metrics["lap"]
        primary = float(lap.get("lap_time", 0.0)
                        + self.rollback_lambda * lap.get("infeasible_count", 0))

        applied_diff: Dict[str, dict] = {}

        ### HJ : Wall-touch CRASH GUARD (P1).
        ###       lap 동안 wall_collision rising-edge ≥1 = 충돌.
        ###       이전: snapshot 있으면 rollback 만 하고 return → crash_detected 룰
        ###             영원히 발화 못 함 → 매 lap 같은 자리 부딪히기만 반복.
        ###       지금: rollback 한 뒤에도 rule eval 까지 fall-through. 같은 코너
        ###             반복 충돌 시 룰 (crash_detected_chronic 등) 이 path/v_max
        ###             등 추가 보수 적용.
        crash_n = int(lap.get("crash_count", 0))
        crash_handled = False
        if crash_n > 0 and self.state.state == "ACTIVE":
            if self.state.last_snapshot is not None:
                rospy.logerr(f"[mpc_param.tuner] CRASH 감지 (wall_touch={crash_n}) — ROLLBACK + rule eval")
                self._do_rollback(f"crash_count={crash_n} lap#{evt.lap_count}", lap)
                # cooldown 직전 변경 키만 1 lap 부여 (다른 키는 fall-through 룰 발화 가능)
                for k in self.state.last_applied_keys:
                    self.state.cooldown_keys[k] = max(self.state.cooldown_keys.get(k, 0), 2)
                crash_handled = True   # rollback 완료, 아래 rule eval 로 진입
            else:
                rospy.logwarn(f"[mpc_param.tuner] CRASH 감지 (snapshot 없음) — rule eval 만 진행")

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

        ### HJ : PANIC cooldown 중이면 rule 적용 안 함 — last_good 검증 시간.
        ###       단, crash 발생 중이면 cooldown 무시하고 룰 발화 (crash_detected 가 즉시
        ###       대응해야 같은 곳에서 계속 부딪히는 패턴 끊김).
        if self.state.panic_cooldown_remaining > 0 and int(lap.get("crash_count", 0)) == 0:
            rospy.loginfo(f"[mpc_param.tuner] PANIC cooldown ({self.state.panic_cooldown_remaining} lap 남음) — rule skip")
            self._update_baselines(lap, primary)
            self._write_csv(metrics, applied_diff)
            return
        if self.state.panic_cooldown_remaining > 0:
            rospy.logwarn(f"[mpc_param.tuner] PANIC cooldown 중이지만 crash 발생 → rule 평가 강제")

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
        ### HJ : best_lap_time 추적 — crash 없는 lap 만 인정. lap_time_regression
        ###       rule 의 reference. metrics 에 'best_lap_time' 으로 노출.
        ### HJ : outlier 보호. lap#1 = 25.18s 였다 lap#2~ = 26+s 인 패턴 관찰 →
        ###       1 lap outlier 가 best 잡으면 매 lap regression 발화 + push 누적
        ###       → unstable. 최소 3 lap 후 best 결정 (≥laps_seen).
        if (lap.get("crash_count", 0) == 0
                and lap.get("infeasible_count", 0) == 0
                and self.state.laps_seen >= 3):
            lt = float(lap.get("lap_time", 0.0))
            if lt > 0 and (self.state.best_lap_time is None or lt < self.state.best_lap_time):
                self.state.best_lap_time = lt
                rospy.loginfo(f"[mpc_param.tuner] new best lap_time={lt:.3f}s")
                ### HJ : best 갱신 = 현재 yaml 이 last_good. 백업.
                self._save_last_good()
        # rule 평가 시 lap dict 에 best 노출
        lap["best_lap_time"] = float(self.state.best_lap_time) if self.state.best_lap_time else float(lap.get("lap_time", 0.0))
        ### HJ : 정상 lap (lap_time > min_real_lap_s) 이면 watchdog count reset.
        if lap.get("lap_time", 0.0) >= self.min_real_lap_s and lap.get("crash_count", 0) == 0:
            self.state.consecutive_watchdog = 0
        ### HJ : panic cooldown 감소 — real lap 이면 무조건 감소 (crash 여부 무관).
        ###       crash 마다 cooldown 유지하면 룰 영원히 skip → crash 못 고침.
        if self.state.panic_cooldown_remaining > 0 \
                and lap.get("lap_time", 0.0) >= self.min_real_lap_s:
            self.state.panic_cooldown_remaining -= 1
            if self.state.panic_cooldown_remaining == 0:
                rospy.loginfo(
                    f"[mpc_param.tuner] PANIC cooldown 종료 (lap={lap.get('lap_time'):.2f}s) → ACTIVE 복귀")
            else:
                rospy.loginfo(
                    f"[mpc_param.tuner] PANIC cooldown 남음 {self.state.panic_cooldown_remaining}")

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

    def _init_baseline_backup(self) -> None:
        ### HJ : 데몬 시작 시 현재 yaml 을 LAST_GOOD 로 저장 (panic recovery 용).
        ###       이후 best lap 갱신 시 _save_last_good() 으로 업데이트.
        try:
            base_path = self.yaml_path + ".LAST_GOOD"
            import shutil
            shutil.copyfile(self.yaml_path, base_path)
            self.state.last_good_yaml_path = base_path
            rospy.loginfo(f"[mpc_param.tuner] baseline backed up → {base_path}")
        except Exception as e:
            rospy.logerr(f"[mpc_param.tuner] baseline backup fail: {e}")

    def _save_last_good(self) -> None:
        ### HJ : best lap 갱신 시 호출. 현재 yaml 을 LAST_GOOD 으로 덮어씀.
        try:
            if self.state.last_good_yaml_path:
                import shutil
                shutil.copyfile(self.yaml_path, self.state.last_good_yaml_path)
        except Exception as e:
            rospy.logwarn(f"[mpc_param.tuner] last_good save fail: {e}")

    def _restore_last_good(self) -> bool:
        ### HJ : PANIC 시 호출. last_good yaml 을 active yaml 로 복원.
        if not self.state.last_good_yaml_path or not os.path.isfile(self.state.last_good_yaml_path):
            rospy.logerr("[mpc_param.tuner] last_good 파일 없음 — restore 실패")
            return False
        try:
            import shutil
            shutil.copyfile(self.state.last_good_yaml_path, self.yaml_path)
            # rosparam push + reload
            self.current_yaml = self._load_current_yaml()
            for k, v in self.current_yaml.items():
                rospy.set_param(f"/{self.target}/{k}", float(v))
            ok, msg = self.patcher._call_reload()
            rospy.logwarn(f"[mpc_param.tuner] PANIC restore done. reload={ok} ({msg})")
            return True
        except Exception as e:
            rospy.logerr(f"[mpc_param.tuner] PANIC restore fail: {e}")
            return False

    def _emergency_safe_defaults(self) -> None:
        ### HJ : 모든 fallback 실패 시 발화. 안전 setup 강제.
        ###       config-driven (rosparam ~emergency_defaults). 기본값 fallback.
        ### HJ : 30s 안 또 EMERGENCY 시도 → 무한 cycle 진단. PAUSED 상태로 전환.
        now = rospy.Time.now().to_sec()
        if hasattr(self, '_last_emergency_t') and (now - self._last_emergency_t) < 30.0:
            rospy.logerr(
                f"[mpc_param.tuner] EMERGENCY 직후 30s 안 재발화 — 무한 cycle 진단. PAUSED 전환.")
            self.state.state = "PAUSED"
            return
        self._last_emergency_t = now
        SAFE = self._load_emergency_defaults()
        rospy.logerr(f"[mpc_param.tuner] EMERGENCY SAFE DEFAULTS 적용. "
                     f"강제 push 키: {list(SAFE.keys())}")
        try:
            for k, v in SAFE.items():
                rospy.set_param(f"/{self.target}/{k}", float(v))
                self.current_yaml[k] = float(v)
            self._patch_yaml_inplace(SAFE)
            ok, msg = self.patcher._call_reload()
            rospy.logwarn(f"[mpc_param.tuner] EMERGENCY reload={ok} ({msg})")
            # 새 baseline 으로 마킹 — 이게 진짜 last_good
            self._save_last_good()
        except Exception as e:
            rospy.logerr(f"[mpc_param.tuner] EMERGENCY safe defaults fail: {e}")

    def _load_emergency_defaults(self) -> Dict[str, float]:
        ### HJ : ~emergency_defaults rosparam (dict) 또는 hardcoded fallback.
        ###       MPCC 우선 covered, MPC 도 동일 키 다수 공유.
        defaults_param = rospy.get_param("~emergency_defaults", None)
        if isinstance(defaults_param, dict) and defaults_param:
            return {str(k): float(v) for k, v in defaults_param.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
        # hardcoded fallback — upenn_mpcc_kd.yaml verified baseline (사용자 검증판)
        ### HJ : 2026-04-28 — kd.yaml 의 baseline 과 정확히 일치시킴. 이전엔 내가
        ###       추측한 값 (w_progress 3.0, friction_slack 200 등) 이 들어가 있어서
        ###       PANIC 발화 시 실제로 baseline 보다 더 위험한 setup 강제됐었음.
        return {
            "n_ref_max_offset":  0.0,
            "n_ref_blend_stages": 5,
            "v_max":             3.0,    # baseline (KD iter1)
            "v_min":             0.0,
            "w_d":               5.0,
            "w_dpsi":            5.5,
            "w_progress":        2.0,    # baseline (iter20)
            "w_progress_e":      3.0,
            "w_terminal_scale":  1.0,
            "w_omega":           2.0,
            "w_vy":              0.3,
            "w_steer":           0.3,
            "w_u_steer_rate":    3.0,
            "w_u_accel":         0.3,
            "friction_margin":   0.95,   # baseline (KD iter1)
            "friction_slack_penalty": 1000.0,
            "max_steer":         0.4,
            "max_steer_rate":    3.0,
            "max_accel":         3.0,
            "max_decel":        -3.0,
        }

    def _patch_yaml_inplace(self, updates: Dict[str, float]) -> None:
        ### HJ : EMERGENCY 용. yaml 파일 직접 수정 (특정 key 들의 값 교체).
        with open(self.yaml_path) as f:
            doc = yaml.safe_load(f) or {}
        ns_doc = doc.get(self.target, doc)
        for k, v in updates.items():
            ns_doc[k] = float(v)
        if self.target in doc:
            doc[self.target] = ns_doc
        else:
            doc = ns_doc
        with open(self.yaml_path, "w") as f:
            yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)

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
