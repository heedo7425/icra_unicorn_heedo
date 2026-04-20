#!/usr/bin/env python3
"""Race-day tire ID orchestrator.

State machine:
    IDLE -> SAFETY_CHECK -> WARMUP -> MANEUVER_SELECT -> MANEUVER_RUN
         -> RECORD_FLUSH -> FIT -> QUALITY_GATE
         -> { YAML_WRITE -> MPC_RESPAWN -> DONE | REJECT -> NEXT_MANEUVER }
         -> ABORT

Phase 1 skeleton: full state transitions, real recorder + maneuver stubs +
fit_pipeline import. dry_run and synthetic_gt flags bypass car/MPC side effects.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

# local imports (catkin sets up PYTHONPATH via setup.py + catkin_python_setup)
try:
    from race_day_id import fit_pipeline, yaml_writer, mpc_respawn
    from race_day_id.recorder import Recorder
    from race_day_id.safety_monitor import SafetyMonitor
    from race_day_id.maneuvers import REGISTRY as MANEUVER_REGISTRY
except ImportError:
    # In-source fallback (ran without catkin install): add src/ to path.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))
    from race_day_id import fit_pipeline, yaml_writer, mpc_respawn  # noqa: E402
    from race_day_id.recorder import Recorder  # noqa: E402
    from race_day_id.safety_monitor import SafetyMonitor  # noqa: E402
    from race_day_id.maneuvers import REGISTRY as MANEUVER_REGISTRY  # noqa: E402


STATES = (
    "IDLE", "SAFETY_CHECK", "WARMUP", "MANEUVER_SELECT", "MANEUVER_RUN",
    "RECORD_FLUSH", "FIT", "QUALITY_GATE",
    "YAML_WRITE", "MPC_RESPAWN", "DONE",
    "REJECT", "ABORT",
)


class Orchestrator:
    def __init__(self):
        rospy.init_node("race_day_id_orchestrator")

        ns = "/race_day_id"
        self.params: Dict[str, Any] = rospy.get_param(ns, {}) or {}
        self.maneuver_profiles: Dict[str, Dict[str, Any]] = \
            rospy.get_param(f"{ns}/maneuvers", {}) or {}
        self.car = str(rospy.get_param(f"{ns}/racecar_version",
                                       rospy.get_param("/racecar_version", "SIM")))
        self.dry_run = bool(self.params.get("dry_run", False))
        self.synthetic_gt = bool(self.params.get("synthetic_gt", False))
        self.auto_start = bool(self.params.get("auto_start", False))

        self.queue = list(self.params.get("maneuver_queue", []))
        self.tick_hz = 50.0

        self.recorder = Recorder(rospy,
                                 buffer_s=float(self.params.get("recorder_buffer_s", 120.0)),
                                 hz=float(self.params.get("recorder_hz", 70.0)))
        self.safety = SafetyMonitor(rospy, self.params)

        self.state_pub = rospy.Publisher(f"{ns}/state", String,
                                         queue_size=10, latch=True)
        self.metrics_pub = rospy.Publisher(f"{ns}/metrics", String,
                                           queue_size=10, latch=True)

        rospy.Service(f"{ns}/start", Trigger, self._srv_start)

        self._state = "IDLE"
        self._set_state("IDLE")
        self._active_maneuver = None
        self._last_fit: Optional[Dict[str, Any]] = None
        self._kickoff = False

        rospy.loginfo(f"[race_day_id] ready. car={self.car} dry_run={self.dry_run} "
                      f"synthetic_gt={self.synthetic_gt} queue={self.queue}")

    # ---- state transitions ----
    def _set_state(self, s: str):
        assert s in STATES, s
        self._state = s
        self.state_pub.publish(String(data=s))
        rospy.loginfo(f"[race_day_id] -> {s}")

    def _publish_metrics(self, m: Dict[str, Any]):
        self.metrics_pub.publish(String(data=json.dumps(m)))

    # ---- service ----
    def _srv_start(self, _req):
        if self._state != "IDLE":
            return TriggerResponse(success=False, message=f"state={self._state}")
        self._kickoff = True
        return TriggerResponse(success=True, message="queued")

    # ---- helpers ----
    def _load_prior_yaml(self) -> Dict[str, Any]:
        path = yaml_writer._target_yaml(self.car)
        if not os.path.exists(path):
            return {}
        try:
            import yaml as _yaml
            with open(path, "r") as f:
                return _yaml.safe_load(f) or {}
        except Exception as e:
            rospy.logwarn(f"[race_day_id] prior yaml load failed: {e}")
            return {}

    def _model_for_fit(self, prior: Dict[str, Any]) -> Dict[str, Any]:
        # Build warm-start model consumed by solve_pacejka.
        def gp(k, default):
            return rospy.get_param(k, prior.get(k, default))
        return {
            "m": gp("/vehicle/m", prior.get("m", 3.54)),
            "l_f": gp("/vehicle/l_f", prior.get("l_f", 0.162)),
            "l_r": gp("/vehicle/l_r", prior.get("l_r", 0.145)),
            "l_wb": gp("/vehicle/l_wb", prior.get("l_wb", 0.307)),
            "h_cg": gp("/vehicle/h_cg", prior.get("h_cg", 0.014)),
            "C_Pf_model": list(prior.get("C_Pf", [4.80, 2.16, 0.65, 0.37])),
            "C_Pr_model": list(prior.get("C_Pr", [20.0, 1.50, 0.62, 0.0])),
        }

    def _synthetic_arrays(self, model):
        """Generate synthetic (vx, vy, omega, delta, ax) from GT Pacejka.
        Uses prior yaml's C_Pf/C_Pr as the "ground truth" so fit_pipeline can
        be verified against a known fixed point. synthetic_gt=true bypasses
        the recorder entirely."""
        from race_day_id import synthetic_gt
        gt_f = list(model["C_Pf_model"])
        gt_r = list(model["C_Pr_model"])
        return synthetic_gt.generate_dataset(model, gt_f, gt_r, noise_frac=0.01)

    # ---- main loop ----
    def run(self):
        rate = rospy.Rate(self.tick_hz)
        if self.auto_start:
            self._kickoff = True

        while not rospy.is_shutdown():
            try:
                self._tick()
            except Exception as e:
                rospy.logerr(f"[race_day_id] tick exception: {e}")
                self._set_state("ABORT")
            rate.sleep()
            if self._state in ("DONE", "ABORT"):
                # Stay latched so external observers can read final state.
                pass

    def _tick(self):
        # Safety latch has priority across every state except terminal ones.
        if self.safety.latched and self._state not in ("ABORT", "DONE", "IDLE"):
            self._publish_metrics({"event": "safety_latch", "reason": self.safety.reason})
            if self._active_maneuver is not None:
                self._active_maneuver.stop(self.safety.reason)
            self._set_state("ABORT")
            return

        s = self._state

        if s == "IDLE":
            if self._kickoff:
                self._kickoff = False
                self._set_state("SAFETY_CHECK")

        elif s == "SAFETY_CHECK":
            # Phase 1: just ensure we've seen at least one odom sample.
            if self.recorder._have_odom or self.synthetic_gt:
                self._set_state("WARMUP")
            # else wait

        elif s == "WARMUP":
            # Phase 5 will implement a real warm-up. Phase 1: skip.
            self._set_state("MANEUVER_SELECT")

        elif s == "MANEUVER_SELECT":
            if not self.queue:
                # No more maneuvers -> go fit whatever was captured.
                self._set_state("RECORD_FLUSH")
                return
            name = self.queue.pop(0)
            cls = MANEUVER_REGISTRY.get(name)
            profile = self.maneuver_profiles.get(name, {})
            if cls is None or not profile.get("enabled", True):
                rospy.logwarn(f"[race_day_id] skipping disabled/unknown maneuver: {name}")
                return
            self._active_maneuver = cls(profile, rospy)
            self._active_maneuver.start()
            if not self.synthetic_gt:
                self.recorder.start()
            self._publish_metrics({"event": "maneuver_start", "name": name})
            self._set_state("MANEUVER_RUN")

        elif s == "MANEUVER_RUN":
            m = self._active_maneuver
            if m is None:
                self._set_state("MANEUVER_SELECT")
                return
            m.step()
            self.recorder.sample()
            if m.is_done():
                self._publish_metrics({"event": "maneuver_done", "name": m.name,
                                       "status": m.status().value})
                self._active_maneuver = None
                # After each maneuver, optionally chain to next; Phase 1: always chain.
                self._set_state("MANEUVER_SELECT")

        elif s == "RECORD_FLUSH":
            self.recorder.stop()
            subdir = str(self.params.get("recording_subdir", "data/recordings"))
            # pkg-relative path
            pkg_root = os.path.abspath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            out = os.path.join(pkg_root, subdir, f"{self.car}_{ts}.csv")
            if not self.synthetic_gt:
                n = self.recorder.flush_to_csv(out)
                rospy.loginfo(f"[race_day_id] flushed {n} samples -> {out}")
            self._set_state("FIT")

        elif s == "FIT":
            prior = self._load_prior_yaml()
            model = self._model_for_fit(prior)
            arrays = self._synthetic_arrays(model) if self.synthetic_gt \
                else self.recorder.as_arrays()
            if arrays is None or len(arrays["vx"]) < 100:
                rospy.logwarn("[race_day_id] insufficient samples -> REJECT")
                self._last_fit = None
                self._set_state("REJECT")
                return
            try:
                C_Pf, C_Pr, diag = fit_pipeline.fit_pacejka(arrays, model)
            except Exception as e:
                rospy.logerr(f"[race_day_id] fit failed: {e}")
                self._set_state("REJECT")
                return
            self._last_fit = dict(C_Pf=C_Pf, C_Pr=C_Pr, diag=diag,
                                  model=model, prior=prior)
            self._set_state("QUALITY_GATE")

        elif s == "QUALITY_GATE":
            fit = self._last_fit or {}
            accept, metrics = fit_pipeline.evaluate_fit(
                fit["C_Pf"], fit["C_Pr"], fit["diag"], fit["model"], fit["prior"],
                self.params)
            metrics.update(dict(C_Pf=list(fit["C_Pf"]), C_Pr=list(fit["C_Pr"]),
                                accept=bool(accept)))
            self._publish_metrics({"event": "quality_gate", **metrics})
            rospy.loginfo(f"[race_day_id] gate metrics: {metrics}")
            if accept:
                self._set_state("YAML_WRITE")
            else:
                self._set_state("REJECT")

        elif s == "YAML_WRITE":
            if self.dry_run:
                rospy.loginfo("[race_day_id] dry_run=true -> skip yaml write")
                self._set_state("DONE")
                return
            fit = self._last_fit
            mu = fit["C_Pr"][2] / (fit["model"]["m"] * 9.81 *
                                   fit["model"]["l_f"] / fit["model"]["l_wb"])
            pkg_root = os.path.abspath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
            backup_dir = os.path.join(pkg_root,
                                      str(self.params.get("backup_subdir", "data/backups")))
            info = yaml_writer.write_pacejka_yaml(self.car, fit["C_Pf"], fit["C_Pr"],
                                                  mu, backup_dir)
            yaml_writer.sync_rosparam(rospy, fit["C_Pf"], fit["C_Pr"], mu)
            self._publish_metrics({"event": "yaml_written", **info, "mu": mu})
            self._set_state("MPC_RESPAWN")

        elif s == "MPC_RESPAWN":
            if self.dry_run:
                self._set_state("DONE")
                return
            ok = mpc_respawn.try_reload_service(rospy)
            if not ok:
                ok = mpc_respawn.kill_and_respawn(
                    rospy, self.car,
                    wait_s=float(self.params.get("respawn_wait_s", 5.0)),
                    retry=int(self.params.get("respawn_retry", 1)))
            self._publish_metrics({"event": "mpc_respawn", "ok": bool(ok)})
            self._set_state("DONE" if ok else "ABORT")

        elif s == "REJECT":
            # Try next maneuver if any remain; otherwise ABORT.
            if self.queue:
                self._set_state("MANEUVER_SELECT")
            else:
                self._set_state("ABORT")

        elif s in ("DONE", "ABORT"):
            return


def main():
    try:
        Orchestrator().run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
