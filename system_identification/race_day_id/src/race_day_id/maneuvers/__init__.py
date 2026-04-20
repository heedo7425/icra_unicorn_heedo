from .base_maneuver import BaseManeuver, ManeuverStatus
from .ramp_steer import RampSteerManeuver
from .steady_circle import SteadyCircleManeuver
from .slalom import SlalomManeuver
from .free_lap import FreeLapManeuver

REGISTRY = {
    "ramp_steer": RampSteerManeuver,
    "steady_circle": SteadyCircleManeuver,
    "slalom": SlalomManeuver,
    "free_lap": FreeLapManeuver,
}
