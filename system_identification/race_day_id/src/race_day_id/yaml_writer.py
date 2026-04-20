"""Atomic writer for {car}_pacejka.yaml + rosparam sync.

Phase 1 skeleton: implements the writer; orchestrator won't call it unless
dry_run=false. Phase 4 wires respawn after write.
"""
from __future__ import annotations

import datetime
import os
import shutil
from typing import Any, Dict


def _target_yaml(racecar_version: str, stack_master_root: str = None) -> str:
    if stack_master_root is None:
        # system_identification/race_day_id/src/race_day_id/yaml_writer.py -> repo root
        here = os.path.dirname(os.path.abspath(__file__))
        stack_master_root = os.path.abspath(
            os.path.join(here, "..", "..", "..", "..", "stack_master"))
    return os.path.join(stack_master_root, "config", racecar_version,
                        f"{racecar_version}_pacejka.yaml")


def backup_existing(path: str, backup_dir: str) -> str:
    if not os.path.exists(path):
        return ""
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    dst = os.path.join(backup_dir, f"{os.path.basename(path)}.{ts}.bak")
    shutil.copy2(path, dst)
    return dst


def write_pacejka_yaml(racecar_version: str, C_Pf, C_Pr, mu: float,
                       backup_dir: str, stack_master_root: str = None) -> Dict[str, Any]:
    """Round-trip update of C_Pf/C_Pr/mu only. Other keys preserved."""
    path = _target_yaml(racecar_version, stack_master_root)
    backup = backup_existing(path, backup_dir)

    try:
        from ruamel.yaml import YAML
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        with open(path, "r") as f:
            data = yaml.load(f)
        data["C_Pf"] = [float(x) for x in C_Pf]
        data["C_Pr"] = [float(x) for x in C_Pr]
        data["mu"] = float(mu)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            yaml.dump(data, f)
        os.replace(tmp, path)
    except ImportError:
        # Fallback: naive yaml (loses comment ordering). Phase 4 requires ruamel.
        import yaml as _yaml
        with open(path, "r") as f:
            data = _yaml.safe_load(f) or {}
        data["C_Pf"] = [float(x) for x in C_Pf]
        data["C_Pr"] = [float(x) for x in C_Pr]
        data["mu"] = float(mu)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            _yaml.safe_dump(data, f)
        os.replace(tmp, path)

    return dict(path=path, backup=backup)


def sync_rosparam(rospy_mod, C_Pf, C_Pr, mu: float) -> None:
    rospy_mod.set_param("/tire_front/B", float(C_Pf[0]))
    rospy_mod.set_param("/tire_front/C", float(C_Pf[1]))
    rospy_mod.set_param("/tire_front/D", float(C_Pf[2]))
    rospy_mod.set_param("/tire_front/E", float(C_Pf[3]))
    rospy_mod.set_param("/tire_rear/B", float(C_Pr[0]))
    rospy_mod.set_param("/tire_rear/C", float(C_Pr[1]))
    rospy_mod.set_param("/tire_rear/D", float(C_Pr[2]))
    rospy_mod.set_param("/tire_rear/E", float(C_Pr[3]))
    rospy_mod.set_param("/mu", float(mu))
