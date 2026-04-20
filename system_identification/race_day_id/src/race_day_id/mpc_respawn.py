"""MPC respawn helpers.

Two-tier strategy (matches plan §5):
  1. Call /mpc/reload_params service (no kill).
  2. Fall back to killing live MPC nodes and spawning
     race_day_id/launch/mpc_respawn.launch as a detached subprocess.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import List, Optional


_MPC_NODE_NAMES = ("/mpc_controller", "/ekf_mpc", "/upenn_mpc", "/mpc_only")


def try_reload_service(rospy_mod, service: str = "/mpc/reload_params",
                       timeout_s: float = 2.0) -> bool:
    try:
        from std_srvs.srv import Trigger
        rospy_mod.wait_for_service(service, timeout=timeout_s)
        call = rospy_mod.ServiceProxy(service, Trigger)
        resp = call()
        return bool(resp.success)
    except Exception:
        return False


def list_mpc_nodes() -> List[str]:
    try:
        out = subprocess.check_output(["rosnode", "list"], text=True, timeout=3)
    except Exception:
        return []
    return [n.strip() for n in out.splitlines()
            if n.strip() in _MPC_NODE_NAMES]


def _spawn_launch(racecar_version: str, logfile: str) -> subprocess.Popen:
    """Launch race_day_id/mpc_respawn.launch in its own session so it survives
    orchestrator shutdown."""
    env = os.environ.copy()
    cmd = ["roslaunch", "race_day_id", "mpc_respawn.launch",
           f"racecar_version:={racecar_version}"]
    log_f = open(logfile, "ab")
    return subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f,
                            start_new_session=True)


def kill_and_respawn(rospy_mod, racecar_version: str,
                     wait_s: float = 5.0, retry: int = 1,
                     log_dir: Optional[str] = None) -> bool:
    alive = list_mpc_nodes()
    if alive:
        try:
            subprocess.call(["rosnode", "kill", *alive], timeout=5)
        except Exception:
            pass
        # Give master time to drop the nodes.
        time.sleep(0.5)

    log_dir = log_dir or os.path.expanduser("~/.ros/race_day_id")
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"mpc_respawn_{int(time.time())}.log")

    for attempt in range(retry + 1):
        proc = _spawn_launch(racecar_version, logfile)
        # Poll for any MPC node showing up within wait_s.
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if list_mpc_nodes():
                try:
                    rospy_mod.loginfo(f"[mpc_respawn] MPC alive (attempt {attempt + 1})")
                except Exception:
                    pass
                return True
            if proc.poll() is not None and proc.returncode != 0:
                break
            time.sleep(0.2)
        # Attempt failed. Kill spawned proc and retry.
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            pass
    return False
