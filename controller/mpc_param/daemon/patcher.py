"""patcher.py — Phase 3: yaml in-place edit + ROS param push + reload service + git commit.

Source-of-truth yaml = controller/<target>/config/<...>.yaml (real ROS yaml the
node was launched with). We edit it in place to keep git history meaningful.

Comment preservation: simple line-regex edit. The yaml format used in this
project is one-key-per-line (`key:    value   # comment`). No nested mapping
edits attempted (we never touch the `gp:` block etc.). For safety, the patcher
refuses to edit a key whose line doesn't match the simple pattern.

Rollback: snapshot the full file text + ROS param values BEFORE the patch.
restore() puts the file back, re-pushes params, calls reload.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Match `   key: <number>   # optional comment`
_KEY_RE = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
    r"(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?P<tail>\s*(?:#.*)?)?$"
)


@dataclass
class PatchSnapshot:
    """Saved state for rollback."""
    yaml_path: str
    yaml_text: str
    ros_param_values: Dict[str, float]
    applied_diff: Dict[str, dict]
    timestamp: float


@dataclass
class PatchResult:
    success: bool
    applied: Dict[str, Tuple[float, float]]   # key -> (old, new)
    skipped: Dict[str, str]                   # key -> reason
    snapshot: Optional[PatchSnapshot] = None
    git_sha: Optional[str] = None
    reload_message: str = ""


class YamlPatcher:
    """Edit yaml + push ROS params + call reload service + git commit.

    Usage:
        patcher = YamlPatcher(
            yaml_path='controller/upenn_mpc/config/upenn_mpc_srx1.yaml',
            namespace='/upenn_mpc',
            reload_service='/upenn_mpc/reload_params',
            git_repo='/home/unicorn/catkin_ws/src/race_stack',  # or local path
        )
        result = patcher.apply({'w_d': 1.275, 'w_dpsi': 6.325},
                               commit_msg='auto-tune lap#42 corner_exit_understeer')
        # later, if rollback:
        patcher.restore(result.snapshot, commit_msg='rollback lap#43 worsened')
    """

    def __init__(self,
                 yaml_path: str,
                 namespace: str = "/upenn_mpc",
                 reload_service: str = "/upenn_mpc/reload_params",
                 git_repo: Optional[str] = None,
                 dry_run: bool = False):
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(yaml_path)
        self.yaml_path = os.path.abspath(yaml_path)
        self.namespace = namespace.rstrip("/")
        self.reload_service = reload_service
        self.git_repo = git_repo or self._detect_repo(self.yaml_path)
        self.dry_run = dry_run

    # ----------------------------------------------------------------------
    # Apply
    # ----------------------------------------------------------------------
    def apply(self, updates: Dict[str, float],
              commit_msg: str = "mpc_param: auto-tune") -> PatchResult:
        """Patch yaml + push ROS params + reload + commit. Returns PatchResult."""
        with open(self.yaml_path, "r") as f:
            original_text = f.read()

        # Snapshot current ROS param values for rollback (best effort).
        snap_params = self._read_ros_params(updates.keys())

        new_text, applied, skipped = self._edit_text(original_text, updates)
        if not applied:
            return PatchResult(success=True, applied={}, skipped=skipped,
                               snapshot=None, reload_message="(no-op)")

        if self.dry_run:
            print("[patcher dry_run] would write:\n", new_text[:400])
            return PatchResult(success=True, applied=applied, skipped=skipped,
                               snapshot=None, reload_message="dry_run")

        with open(self.yaml_path, "w") as f:
            f.write(new_text)

        # Push to ROS param server.
        for k, (old, new) in applied.items():
            self._set_ros_param(k, new)

        # Call reload service.
        ok, reload_msg = self._call_reload()
        if not ok:
            # Best-effort: try to restore the file before bailing.
            with open(self.yaml_path, "w") as f:
                f.write(original_text)
            for k, v in snap_params.items():
                self._set_ros_param(k, v)
            return PatchResult(success=False, applied={}, skipped=skipped,
                               snapshot=None, reload_message=reload_msg)

        # Git commit (best effort — failure doesn't break the loop).
        diff_dict = {k: {"old": v[0], "new": v[1]} for k, v in applied.items()}
        sha = self._git_commit(self.yaml_path, commit_msg + "\n\n" + repr(diff_dict))

        snap = PatchSnapshot(
            yaml_path=self.yaml_path,
            yaml_text=original_text,
            ros_param_values=snap_params,
            applied_diff=diff_dict,
            timestamp=time.time(),
        )
        return PatchResult(success=True, applied=applied, skipped=skipped,
                           snapshot=snap, git_sha=sha,
                           reload_message=reload_msg)

    # ----------------------------------------------------------------------
    # Rollback
    # ----------------------------------------------------------------------
    def restore(self, snap: PatchSnapshot,
                commit_msg: str = "mpc_param: rollback") -> PatchResult:
        if snap.yaml_path != self.yaml_path:
            raise ValueError(f"snapshot path mismatch: {snap.yaml_path} vs {self.yaml_path}")
        with open(self.yaml_path, "w") as f:
            f.write(snap.yaml_text)
        for k, v in snap.ros_param_values.items():
            self._set_ros_param(k, v)
        ok, reload_msg = self._call_reload()
        sha = self._git_commit(self.yaml_path, commit_msg + "\n\n(rollback)")
        return PatchResult(
            success=ok, applied={k: (snap.applied_diff[k]["new"], snap.applied_diff[k]["old"])
                                  for k in snap.applied_diff},
            skipped={}, snapshot=None, git_sha=sha, reload_message=reload_msg,
        )

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------
    @staticmethod
    def _edit_text(text: str, updates: Dict[str, float]
                   ) -> Tuple[str, Dict[str, Tuple[float, float]], Dict[str, str]]:
        """Line-by-line regex edit. Preserves comments and indent."""
        applied: Dict[str, Tuple[float, float]] = {}
        skipped: Dict[str, str] = dict.fromkeys(updates.keys(), "key_not_found")
        new_lines = []
        for line in text.splitlines(keepends=True):
            stripped = line.rstrip("\n").rstrip("\r")
            m = _KEY_RE.match(stripped)
            if m:
                key = m.group("key")
                if key in updates:
                    old_val = float(m.group("value"))
                    new_val = float(updates[key])
                    tail = m.group("tail") or ""
                    new_line = f"{m.group('indent')}{key}: {new_val:g}{tail}\n"
                    new_lines.append(new_line)
                    applied[key] = (old_val, new_val)
                    skipped.pop(key, None)
                    continue
            new_lines.append(line)
        return "".join(new_lines), applied, skipped

    def _read_ros_params(self, keys) -> Dict[str, float]:
        try:
            import rospy
        except ImportError:
            return {}
        out = {}
        for k in keys:
            full = f"{self.namespace}/{k}"
            if rospy.has_param(full):
                try:
                    out[k] = float(rospy.get_param(full))
                except (TypeError, ValueError):
                    pass
        return out

    def _set_ros_param(self, key: str, value: float) -> None:
        try:
            import rospy
            rospy.set_param(f"{self.namespace}/{key}", float(value))
        except ImportError:
            pass

    def _call_reload(self) -> Tuple[bool, str]:
        try:
            import rospy
            from std_srvs.srv import Trigger
            rospy.wait_for_service(self.reload_service, timeout=2.0)
            proxy = rospy.ServiceProxy(self.reload_service, Trigger)
            resp = proxy()
            return bool(resp.success), str(resp.message)
        except Exception as e:
            return False, f"service call fail: {e}"

    # ---- git ----
    @staticmethod
    def _detect_repo(path: str) -> Optional[str]:
        d = os.path.dirname(path)
        while d and d != "/":
            if os.path.isdir(os.path.join(d, ".git")):
                return d
            d = os.path.dirname(d)
        return None

    def _git_commit(self, file_path: str, msg: str) -> Optional[str]:
        if not self.git_repo:
            return None
        try:
            rel = os.path.relpath(file_path, self.git_repo)
            subprocess.run(["git", "-C", self.git_repo, "add", rel],
                           check=True, capture_output=True, timeout=10)
            r = subprocess.run(["git", "-C", self.git_repo, "commit", "-m", msg,
                                "--author", "mpc_param-tuner <tuner@local>"],
                               capture_output=True, timeout=10)
            if r.returncode != 0:
                return None
            r2 = subprocess.run(["git", "-C", self.git_repo, "rev-parse", "HEAD"],
                                check=True, capture_output=True, timeout=5)
            return r2.stdout.decode().strip()[:8]
        except Exception:
            return None
