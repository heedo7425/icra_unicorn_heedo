"""Gain scheduler for friction-aware Pure Pursuit.

Inputs:
    mu_prior  : feedforward friction prior from MuZoneMap (~ 0.4..1.2)
    slip_lvl  : online slip estimate in [-1, +1]
                  +1 = slipping, -1 = excess grip, 0 = nominal

Outputs (all multiplicative scales applied to PPC baseline):
    L_d_scale       : lookahead distance scale  (>1 = longer / softer)
    steer_rate_lim  : max |delta_dot| [rad/s]
    ax_max          : max long. accel command [m/s^2] (positive)
    ax_min          : min long. accel command [m/s^2] (negative, brake)

Asymmetric time constants: tighten quickly toward slip side (safety), relax
slowly toward grip side (avoid premature aggressiveness). Prior gives the
*pre-corner* setting; measurement corrects after entry.
"""

import math


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _lp(prev, new, dt, tau):
    if tau <= 0.0:
        return new
    a = dt / (dt + tau)
    return prev + a * (new - prev)


class GainScheduler:
    def __init__(self,
                 ld_low=1.6, ld_high=0.8,
                 steer_rate_low=2.5, steer_rate_high=8.0,
                 ax_low=2.0, ax_high=6.0,
                 brake_low=2.5, brake_high=7.0,
                 tau_tighten=0.08, tau_relax=0.6,
                 prior_weight=0.5):
        self.ld_low = ld_low
        self.ld_high = ld_high
        self.sr_low = steer_rate_low
        self.sr_high = steer_rate_high
        self.ax_low = ax_low
        self.ax_high = ax_high
        self.br_low = brake_low
        self.br_high = brake_high
        self.tau_tighten = tau_tighten
        self.tau_relax = tau_relax
        self.prior_weight = prior_weight

        self._t_prev = None
        self._state = {
            'ld_scale': 1.0,
            'steer_rate': (steer_rate_low + steer_rate_high) * 0.5,
            'ax_max': (ax_low + ax_high) * 0.5,
            'ax_min': -((brake_low + brake_high) * 0.5),
        }

    def _mu_to_alpha(self, mu):
        """Map friction prior (0.4..1.2) -> alpha in [0,1] (1=high grip)."""
        return _clip((mu - 0.4) / (1.2 - 0.4), 0.0, 1.0)

    def _slip_to_alpha(self, slip):
        """Map slip level [-1,+1] -> alpha in [0,1] (1=excess grip, 0=slipping)."""
        return _clip(0.5 - 0.5 * slip, 0.0, 1.0)

    def update(self, t_now, mu_prior, slip_level):
        if self._t_prev is None:
            self._t_prev = t_now
            dt = 0.02
        else:
            dt = max(1e-3, t_now - self._t_prev)
            self._t_prev = t_now

        a_prior = self._mu_to_alpha(mu_prior)
        a_meas = self._slip_to_alpha(slip_level)
        # Prior is anticipatory; measurement governs once we are inside.
        # Take min so we never trust grip we haven't measured.
        alpha = min(self.prior_weight * a_prior + (1.0 - self.prior_weight) * a_meas,
                    a_meas if slip_level > 0.0 else a_prior)
        alpha = _clip(alpha, 0.0, 1.0)

        ld_target = self.ld_low + (self.ld_high - self.ld_low) * alpha
        sr_target = self.sr_low + (self.sr_high - self.sr_low) * alpha
        ax_target = self.ax_low + (self.ax_high - self.ax_low) * alpha
        br_target = self.br_low + (self.br_high - self.br_low) * alpha

        def step(name, target):
            cur = self._state[name]
            tightening = (
                (name == 'ld_scale' and target > cur) or
                (name == 'steer_rate' and target < cur) or
                (name == 'ax_max' and target < cur) or
                (name == 'ax_min' and target > cur)  # ax_min stored negative; tighter = larger
            )
            tau = self.tau_tighten if tightening else self.tau_relax
            self._state[name] = _lp(cur, target, dt, tau)

        step('ld_scale', ld_target)
        step('steer_rate', sr_target)
        step('ax_max', ax_target)
        step('ax_min', -br_target)

        return {
            'ld_scale': self._state['ld_scale'],
            'steer_rate_lim': self._state['steer_rate'],
            'ax_max': self._state['ax_max'],
            'ax_min': self._state['ax_min'],
            'alpha': alpha,
        }
