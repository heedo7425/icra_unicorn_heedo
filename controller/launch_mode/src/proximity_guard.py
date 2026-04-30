"""ROI-based LiDAR proximity guard for launch_mode.

Checks two regions in the base_link frame from a single LaserScan:
  - FRONT box: collision-risk corridor straight ahead.
  - SIDE boxes (left + right): adjacent grid slot occupancy.

Coordinates assume LaserScan is already expressed in the vehicle's
base_link frame (typical for a single-LiDAR F1TENTH stack). x is forward,
y is left.

Two consumers:
  * decide_launch_intent(grid_slot) -> 'FULL' | 'REDUCED' | 'ABORT'
    Called once at ARMED -> LAUNCH transition.
  * runtime_abort(v_now) -> bool
    Called every tick while in LAUNCH; True means abort immediately.
"""

import math
import time


class ProximityGuard:
    def __init__(self,
                 front_x_max=1.5,
                 front_y_half=0.30,
                 side_x_min=-0.5,
                 side_x_max=1.5,
                 side_y_min=0.3,
                 side_y_max=1.5,
                 roi_min_points=5,
                 safe_lateral_gap_m=0.6,
                 front_ttc_th_s=0.4,
                 front_proximity_min_m=0.3,
                 scan_timeout_s=0.3):
        self.front_x_max = float(front_x_max)
        self.front_y_half = float(front_y_half)
        self.side_x_min = float(side_x_min)
        self.side_x_max = float(side_x_max)
        self.side_y_min = float(side_y_min)
        self.side_y_max = float(side_y_max)
        self.roi_min_points = int(roi_min_points)
        self.safe_lateral_gap_m = float(safe_lateral_gap_m)
        self.front_ttc_th_s = float(front_ttc_th_s)
        self.front_proximity_min_m = float(front_proximity_min_m)
        self.scan_timeout_s = float(scan_timeout_s)

        self._t_last_scan = -1.0   # sentinel: no scan yet
        self._result = {
            'fresh': False,
            'front_obstacle': False,
            'front_min_range': float('inf'),
            'front_count': 0,
            'lateral_gap': float('inf'),
            'side_l_count': 0,
            'side_r_count': 0,
        }

    # ------------------------------------------------------------------
    def update(self, scan_msg, t_now=None):
        """Process a sensor_msgs/LaserScan and cache the latest result."""
        if t_now is None:
            t_now = time.time()
        # use scan stamp if available so we know freshness wrt sensor
        try:
            stamp = scan_msg.header.stamp.to_sec()
            if stamp > 0:
                self._t_last_scan = stamp
            else:
                self._t_last_scan = t_now
        except Exception:
            self._t_last_scan = t_now

        ranges = scan_msg.ranges
        n = len(ranges)
        if n == 0:
            return self._result

        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        r_min = scan_msg.range_min
        r_max = scan_msg.range_max

        front_count = 0
        front_min_r = float('inf')
        side_l_count = 0
        side_l_min_r = float('inf')
        side_r_count = 0
        side_r_min_r = float('inf')

        for i in range(n):
            r = ranges[i]
            if not (r_min < r < r_max) or math.isinf(r) or math.isnan(r):
                continue
            a = angle_min + i * angle_inc
            x = r * math.cos(a)
            y = r * math.sin(a)

            # FRONT: x in (0, front_x_max], |y| <= front_y_half
            if 0.0 < x <= self.front_x_max and abs(y) <= self.front_y_half:
                front_count += 1
                if r < front_min_r:
                    front_min_r = r

            # SIDE LEFT: x in [side_x_min, side_x_max], y in [side_y_min, side_y_max]
            if (self.side_x_min <= x <= self.side_x_max
                    and self.side_y_min <= y <= self.side_y_max):
                side_l_count += 1
                if r < side_l_min_r:
                    side_l_min_r = r

            # SIDE RIGHT: same x range, y in [-side_y_max, -side_y_min]
            if (self.side_x_min <= x <= self.side_x_max
                    and -self.side_y_max <= y <= -self.side_y_min):
                side_r_count += 1
                if r < side_r_min_r:
                    side_r_min_r = r

        front_obstacle = (front_count >= self.roi_min_points)
        # lateral_gap: closest side distance among occupied sides
        candidates = []
        if side_l_count >= self.roi_min_points:
            candidates.append(side_l_min_r)
        if side_r_count >= self.roi_min_points:
            candidates.append(side_r_min_r)
        lateral_gap = min(candidates) if candidates else float('inf')

        fresh = (t_now - self._t_last_scan) < self.scan_timeout_s

        self._result = {
            'fresh': fresh,
            'front_obstacle': front_obstacle,
            'front_min_range': front_min_r,
            'front_count': front_count,
            'lateral_gap': lateral_gap,
            'side_l_count': side_l_count,
            'side_r_count': side_r_count,
        }
        return self._result

    # ------------------------------------------------------------------
    def is_fresh(self, t_now=None):
        if self._t_last_scan < 0.0:
            return False
        if t_now is None:
            t_now = time.time()
        return (t_now - self._t_last_scan) < self.scan_timeout_s

    @property
    def last(self):
        return self._result

    # ------------------------------------------------------------------
    def feasibility(self, grid_slot, t_now=None):
        """
        Continuously-evaluable advisory about launch viability.

        Returns one of:
          'GREEN'   : full launch is safe.
          'YELLOW'  : reduced launch is safer (front obstacle, lateral ok).
          'RED'     : front obstacle AND lateral tight; the human is
                      explicitly choosing risk if they arm.
          'UNKNOWN' : scan is stale / no scan yet — system cannot advise.

        The system uses this only as advisory output; it does NOT veto
        the human's launch_arm. The only auto-veto is sensor failure
        (UNKNOWN), handled separately in decide_launch_intent.
        """
        slot = (grid_slot or 'rear').lower()
        if slot == 'front':
            return 'GREEN'

        if not self.is_fresh(t_now):
            return 'UNKNOWN'

        if not self._result['front_obstacle']:
            return 'GREEN'

        if self._result['lateral_gap'] > self.safe_lateral_gap_m:
            return 'YELLOW'

        return 'RED'

    # ------------------------------------------------------------------
    def decide_launch_intent(self, grid_slot, t_now=None):
        """
        ARMED -> LAUNCH: pick FULL / REDUCED / ABORT from current scan.

        Policy:
          * front grid                 -> FULL
          * rear, scan stale           -> ABORT  (sensor failure veto)
          * rear, front clear          -> FULL
          * rear, front blocked        -> REDUCED  (regardless of lateral
                                          gap — human armed knowing risk)
        Lateral tightness is only used by feasibility() for advisory
        output; we do not veto the human here.
        """
        slot = (grid_slot or 'rear').lower()
        if slot == 'front':
            return 'FULL'

        if not self.is_fresh(t_now):
            return 'ABORT'

        if not self._result['front_obstacle']:
            return 'FULL'

        return 'REDUCED'

    # ------------------------------------------------------------------
    def runtime_abort(self, v_now, t_now=None):
        """While in LAUNCH: True if we should abort due to proximity."""
        if not self.is_fresh(t_now):
            # scan died mid-launch: do NOT abort on stale data (false-alarm
            # risk). The FSM has its own time/distance/speed exit gates.
            return False

        r = self._result['front_min_range']
        if r < self.front_proximity_min_m:
            return True
        if v_now > 0.1:
            ttc = r / max(v_now, 0.1)
            if ttc < self.front_ttc_th_s:
                return True
        return False
