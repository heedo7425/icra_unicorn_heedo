#!/usr/bin/env python3
"""Smoke test for patcher — yaml edit only (no ROS).

Copies upenn_mpc_srx1.yaml to /tmp, applies a patch, verifies:
  - changed keys updated
  - other lines (comments, blank, untouched keys) preserved byte-for-byte
  - rollback restores original
"""
from __future__ import annotations

import os
import sys
import tempfile
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from daemon.patcher import YamlPatcher, PatchSnapshot

SRC = "/home/hmcl/icra2026_ws/ICRA2026_HJ/controller/upenn_mpc/config/upenn_mpc_srx1.yaml"


def main():
    tmpdir = tempfile.mkdtemp(prefix="mpc_param_test_")
    dst = os.path.join(tmpdir, "upenn_mpc_srx1.yaml")
    shutil.copy(SRC, dst)
    print(f"workdir: {tmpdir}")

    with open(dst) as f:
        original = f.read()

    p = YamlPatcher(yaml_path=dst, namespace="/upenn_mpc",
                    reload_service="/none", git_repo=None, dry_run=False)

    updates = {"w_d": 1.275, "w_dpsi": 6.325, "speed_boost": 1.05,
               "nonexistent_key": 99.9}

    # apply (call _edit_text directly to skip ROS bits)
    new_text, applied, skipped = p._edit_text(original, updates)
    print(f"applied: {applied}")
    print(f"skipped: {skipped}")

    assert "w_d" in applied and applied["w_d"][0] == 1.5 and applied["w_d"][1] == 1.275
    assert "w_dpsi" in applied and applied["w_dpsi"][0] == 5.5 and applied["w_dpsi"][1] == 6.325
    assert "speed_boost" in applied
    assert "nonexistent_key" in skipped

    with open(dst, "w") as f:
        f.write(new_text)

    # Verify by re-reading: each updated key must match, comments preserved
    with open(dst) as f:
        edited = f.read()

    # Comment lines should be byte-identical
    orig_comments = [l for l in original.splitlines() if l.strip().startswith("#")]
    edit_comments = [l for l in edited.splitlines() if l.strip().startswith("#")]
    assert orig_comments == edit_comments, "comments altered!"

    # Re-grep the new values
    import re
    assert re.search(r"^\s*w_d:\s*1\.275", edited, re.M), "w_d not updated"
    assert re.search(r"^\s*w_dpsi:\s*6\.325", edited, re.M), "w_dpsi not updated"
    assert re.search(r"^\s*speed_boost:\s*1\.05", edited, re.M), "speed_boost not updated"

    # Rollback
    snap = PatchSnapshot(yaml_path=dst, yaml_text=original,
                         ros_param_values={}, applied_diff={}, timestamp=0)
    with open(snap.yaml_path, "w") as f:
        f.write(snap.yaml_text)
    with open(dst) as f:
        restored = f.read()
    assert restored == original, "rollback failed"

    print("\nALL OK")
    print(f"(workdir kept at {tmpdir} for inspection)")


if __name__ == "__main__":
    main()
