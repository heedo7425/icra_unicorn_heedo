"""Online slip estimator.

Three independent residuals — fused into a single normalized slip level in
[-1, +1] where positive means 'measured grip below model expectation' and
negative means 'measured grip exceeds model expectation' (i.e. the road is
better than we assumed).

Signals:
    R_yaw  = (yaw_rate_meas - yaw_rate_cmd) / max(|yaw_rate_cmd|, eps)
              expected sign mostly negative on understeer; we use |.|
    R_long = (v_wheel - v_body) / max(v_wheel, eps)
              positive when drive wheel spins faster than chassis
    R_ay   = (ay_expected - ay_meas) / max(|ay_expected|, eps)
              positive when measured lateral g falls short of bicycle model

Each residual is passed through a soft saturation, low-pass filtered, and
combined as a max (slip-side) / min (grip-side).
"""

import math


def _sat(x, lim):
    return max(-lim, min(lim, x))


def _lp(prev, new, dt, tau):
    if tau <= 0.0:
        return new
    a = dt / (dt + tau)
    return prev + a * (new - prev)


class SlipEstimator:
    def __init__(self,
                 wheelbase=0.33,
                 yaw_thr=0.15,
                 long_thr=0.10,
                 ay_thr=0.20,
                 tau_up=0.05,
                 tau_down=0.30,
                 v_min=0.8):
        self.L = float(wheelbase)
        self.yaw_thr = float(yaw_thr)
        self.long_thr = float(long_thr)
        self.ay_thr = float(ay_thr)
        self.tau_up = float(tau_up)      # fast attack when slip increases
        self.tau_down = float(tau_down)  # slow release when slip recovers
        self.v_min = float(v_min)
        self._level = 0.0
        self._t_prev = None

    def reset(self):
        self._level = 0.0
        self._t_prev = None

    def update(self, t_now, v_body, v_wheel, yaw_rate_meas, ay_meas,
               steer_cmd):
        """Returns slip level in [-1, +1]. Positive = slipping; negative = grip excess."""
        if self._t_prev is None:
            self._t_prev = t_now
            return self._level
        dt = max(1e-3, t_now - self._t_prev)
        self._t_prev = t_now

        if v_body < self.v_min:
            level = 0.0
        else:
            yaw_cmd = v_body * math.tan(steer_cmd) / self.L
            ay_exp = v_body * yaw_cmd

            r_yaw = (abs(yaw_rate_meas) - abs(yaw_cmd)) / max(abs(yaw_cmd), 0.2)
            r_yaw = -r_yaw  # understeer => measured < commanded => positive slip

            r_long = (v_wheel - v_body) / max(abs(v_wheel), 0.5)

            r_ay = (abs(ay_exp) - abs(ay_meas)) / max(abs(ay_exp), 0.5)

            slip_side = max(
                _sat(r_yaw / max(self.yaw_thr, 1e-3), 1.5),
                _sat(r_long / max(self.long_thr, 1e-3), 1.5),
                _sat(r_ay / max(self.ay_thr, 1e-3), 1.5),
            )
            grip_side = min(
                _sat(r_yaw / max(self.yaw_thr, 1e-3), 1.5),
                _sat(r_long / max(self.long_thr, 1e-3), 1.5),
                _sat(r_ay / max(self.ay_thr, 1e-3), 1.5),
            )
            level = slip_side if slip_side > 0.0 else grip_side
            level = max(-1.0, min(1.0, level))

        tau = self.tau_up if level > self._level else self.tau_down
        self._level = _lp(self._level, level, dt, tau)
        return self._level

    @property
    def level(self):
        return self._level
