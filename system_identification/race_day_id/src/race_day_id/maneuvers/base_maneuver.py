"""Maneuver plugin contract.

Each maneuver encapsulates: command publisher, termination logic, abort
condition. Orchestrator owns the data recorder; maneuver only actuates.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class ManeuverStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ABORTED = "aborted"


@dataclass
class ManeuverResult:
    status: ManeuverStatus
    reason: str = ""
    meta: Dict[str, Any] = None


class BaseManeuver:
    """Abstract base. Subclasses override start/step/stop/is_done."""

    name: str = "base"

    def __init__(self, profile: Dict[str, Any], rospy_mod):
        self.profile = profile
        self.rospy = rospy_mod
        self._status = ManeuverStatus.IDLE
        self._reason = ""

    # ---- lifecycle ----
    def start(self) -> None:
        self._status = ManeuverStatus.RUNNING

    def step(self) -> None:
        """Called at orchestrator tick rate; publish commands here."""
        raise NotImplementedError

    def stop(self, reason: str = "") -> None:
        if self._status == ManeuverStatus.RUNNING:
            self._status = ManeuverStatus.DONE if not reason else ManeuverStatus.ABORTED
            self._reason = reason

    # ---- queries ----
    def is_done(self) -> bool:
        return self._status in (ManeuverStatus.DONE, ManeuverStatus.ABORTED)

    def status(self) -> ManeuverStatus:
        return self._status

    def reason(self) -> str:
        return self._reason
