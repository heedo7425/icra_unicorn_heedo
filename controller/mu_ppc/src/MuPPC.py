"""Friction-aware Pure Pursuit controller.

Plug-compatible with `combined.src.Controller.Controller.main_loop`:
  returns (speed, accel, jerk, steer, L1_point, L1_dist, idx_nearest,
           curvature, future_position)

Differences vs. baseline PPC:
  * lookahead distance is scaled by a friction-aware GainScheduler
  * steering rate limited
  * longitudinal accel / brake commands clipped per current grip belief
  * online slip estimator + Frenet-s indexed mu prior fused

Speed reference is taken straight from local waypoints — we never override
the offline raceline profile (project rule). All adaptation happens on
controller-side signals: steer, jerk, accel limits.
"""

import math
import time
import numpy as np

from .mu_zone_map import MuZoneMap
from .slip_estimator import SlipEstimator
from .gain_scheduler import GainScheduler


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class MuPPC:
    def __init__(self,
                 wheelbase=0.33,
                 loop_rate=50,
                 ld_base_min=0.6,
                 ld_base_max=3.0,
                 ld_speed_slope=0.30,
                 ld_speed_intercept=0.20,
                 max_steer=0.42,
                 logger_info=print,
                 logger_warn=print,
                 converter=None):
        self.L = float(wheelbase)
        self.dt = 1.0 / float(loop_rate)
        self.ld_base_min = ld_base_min
        self.ld_base_max = ld_base_max
        self.ld_m = ld_speed_slope
        self.ld_q = ld_speed_intercept
        self.max_steer = max_steer
        self.log_i = logger_info
        self.log_w = logger_warn
        self.converter = converter

        self.zone_map = MuZoneMap(track_length=0.0)
        self.slip = SlipEstimator(wheelbase=self.L)
        self.sched = GainScheduler()

        self.last_steer_cmd = 0.0
        self.last_speed_cmd = 0.0
        self.yaw_rate = 0.0
        self.wheel_speed = 0.0
        self.future_position_z = 0.0

    # external setters used by node / manager
    def set_zone_map_from_rosparam(self, rospy):
        self.zone_map.load_from_rosparam(rospy)

    def set_track_length(self, L):
        self.zone_map.track_length = float(L)

    def set_wheel_speed(self, v):
        self.wheel_speed = float(v)

    def set_yaw_rate(self, r):
        self.yaw_rate = float(r)

    # ------------------------------------------------------------------
    def _baseline_lookahead(self, v):
        ld = self.ld_q + self.ld_m * max(v, 0.0)
        return _clip(ld, self.ld_base_min, self.ld_base_max)

    def _nearest_idx(self, pos_xy, wpts_xy):
        d = wpts_xy - pos_xy
        return int(np.argmin(np.einsum('ij,ij->i', d, d)))

    def _find_lookahead_point(self, pos_xy, wpts_xy, ld, idx_near):
        N = wpts_xy.shape[0]
        for k in range(N):
            i = (idx_near + k) % N
            if np.linalg.norm(wpts_xy[i] - pos_xy) >= ld:
                return i, wpts_xy[i]
        return idx_near, wpts_xy[idx_near]

    def _pp_steer(self, pos_xy, heading, target_xy, ld_eff):
        dx = target_xy[0] - pos_xy[0]
        dy = target_xy[1] - pos_xy[1]
        c, s = math.cos(-heading), math.sin(-heading)
        x_b = c * dx - s * dy
        y_b = s * dx + c * dy
        if ld_eff < 1e-3:
            return 0.0
        kappa = 2.0 * y_b / (ld_eff * ld_eff)
        return math.atan(self.L * kappa)

    # ------------------------------------------------------------------
    def main_loop(self, state, position_in_map, waypoint_array,
                  speed_now, opponent, position_in_map_frenet,
                  acc_now, track_length):
        """Drop-in replacement for combined.Controller.main_loop."""
        t_now = time.time()
        if track_length:
            self.zone_map.track_length = float(track_length)

        if waypoint_array is None or len(waypoint_array) == 0 \
                or position_in_map is None or len(position_in_map) == 0:
            return (0.0, 0.0, 0.0, 0.0,
                    np.zeros(3), 0.0, 0, 0.0,
                    np.zeros((1, 3)))

        pos = position_in_map[0]  # (x, y, theta)
        pos_xy = pos[:2]
        heading = pos[2]

        wpts = np.asarray(waypoint_array)
        wpts_xy = wpts[:, :2]
        v_ref_arr = wpts[:, 3]
        ax_ref_arr = wpts[:, 8] if wpts.shape[1] > 8 else np.zeros(len(wpts))

        idx_near = self._nearest_idx(pos_xy, wpts_xy)

        # --- friction belief ---------------------------------------------------
        s_now = position_in_map_frenet[0] if (
            position_in_map_frenet is not None and len(position_in_map_frenet) > 0
        ) else 0.0
        v = max(speed_now, 0.0)
        ld_base = self._baseline_lookahead(v)
        mu_prior = self.zone_map.mu_ahead(s_now, lookahead=max(ld_base, 1.0) * 1.5)

        ay_meas = float(acc_now[0]) if (acc_now is not None and len(acc_now) > 0) else 0.0
        slip = self.slip.update(
            t_now=t_now,
            v_body=v,
            v_wheel=self.wheel_speed if self.wheel_speed > 0.0 else v,
            yaw_rate_meas=self.yaw_rate,
            ay_meas=ay_meas,
            steer_cmd=self.last_steer_cmd,
        )

        gains = self.sched.update(t_now, mu_prior, slip)
        ld = ld_base * gains['ld_scale']
        ld = _clip(ld, self.ld_base_min, self.ld_base_max * 1.5)

        # --- pure pursuit ------------------------------------------------------
        idx_la, target = self._find_lookahead_point(pos_xy, wpts_xy, ld, idx_near)
        steer_raw = self._pp_steer(pos_xy, heading, target, ld)
        steer_raw = _clip(steer_raw, -self.max_steer, self.max_steer)

        # steering rate limit
        max_d = gains['steer_rate_lim'] * self.dt
        steer = _clip(steer_raw,
                      self.last_steer_cmd - max_d,
                      self.last_steer_cmd + max_d)
        self.last_steer_cmd = steer

        # --- speed / accel -----------------------------------------------------
        v_ref = float(v_ref_arr[idx_la])
        ax_ref = float(ax_ref_arr[idx_la])
        # respect grip-aware accel/brake caps without raising the profile
        ax_cmd = _clip(ax_ref, gains['ax_min'], gains['ax_max'])
        speed_cmd = v_ref
        self.last_speed_cmd = speed_cmd

        L1_point = np.array([target[0], target[1],
                             wpts[idx_la, 2] if wpts.shape[1] > 2 else 0.0])
        L1_distance = float(np.linalg.norm(target - pos_xy))
        curvature = float(wpts[idx_near, 6]) if wpts.shape[1] > 6 else 0.0
        future_position = np.array([[pos_xy[0], pos_xy[1], heading]])
        self.future_position_z = float(L1_point[2])

        return (speed_cmd, ax_cmd, 0.0, steer,
                L1_point, L1_distance, idx_near, curvature, future_position)
