"""Wheel-slip guard for current-based launch.

We only have wheel speed (from VESC ERPM) and body speed (from odom). With
those two we estimate longitudinal slip ratio:

    s_long = (v_wheel - v_body) / max(v_wheel, eps)

If s_long exceeds a threshold for `confirm_n` consecutive ticks, we step the
current command down by a multiplicative factor. We never raise it back; the
launch window is short and re-spinning would waste more time.
"""


class SlipGuard:
    def __init__(self,
                 slip_threshold=0.20,
                 confirm_n=2,
                 cut_factor=0.8,
                 i_floor_ratio=0.4):
        self.slip_threshold = float(slip_threshold)
        self.confirm_n = int(confirm_n)
        self.cut_factor = float(cut_factor)
        self.i_floor_ratio = float(i_floor_ratio)
        self.reset()

    def reset(self):
        self._slip_count = 0
        self._scale = 1.0
        self._last_slip = 0.0

    def update(self, v_wheel, v_body):
        eps = 0.05
        denom = max(abs(v_wheel), eps)
        slip = (v_wheel - v_body) / denom
        self._last_slip = slip
        if slip > self.slip_threshold:
            self._slip_count += 1
            if self._slip_count >= self.confirm_n:
                self._scale = max(self._scale * self.cut_factor,
                                  self.i_floor_ratio)
                self._slip_count = 0
        else:
            self._slip_count = 0
        return self._scale

    def apply(self, i_cmd):
        return i_cmd * self._scale

    @property
    def slip(self):
        return self._last_slip

    @property
    def scale(self):
        return self._scale
