"""Launch-mode finite-state machine.

States
------
IDLE      : autonomous arm OFF. nothing to do.
DISARMED  : autonomous ON, but launch_arm OFF or vehicle not stationary.
ARMED     : autonomous ON, launch_arm ON, vehicle stationary. waiting for GO.
LAUNCH    : current-based throttle override active. one-shot.
NORMAL    : exit complete; controller takes over again.

The FSM only emits transitions/intents. The ROS node decides what to publish.
launch_arm is a one-shot: once LAUNCH fires, it must be re-toggled by the
human (OFF -> ON) before another launch is allowed.
"""

import enum
import time


class LaunchState(enum.IntEnum):
    IDLE = 0
    DISARMED = 1
    ARMED = 2
    LAUNCH = 3
    NORMAL = 4


class LaunchFSM:
    def __init__(self,
                 v_stationary=0.1,
                 s_launch_m=5.0,
                 v_ratio_exit=0.7,
                 t_max_s=1.5,
                 steer_exit_rad=0.2,
                 kappa_exit=0.5):
        self.s_launch_m_full = float(s_launch_m)
        self.v_stationary = float(v_stationary)
        self.s_launch_m = float(s_launch_m)
        self.v_ratio_exit = float(v_ratio_exit)
        self.t_max_s = float(t_max_s)
        self.steer_exit_rad = float(steer_exit_rad)
        self.kappa_exit = float(kappa_exit)

        self.state = LaunchState.IDLE
        self.s_start = None
        self.t_launch = None
        self.launch_arm_consumed = False
        self._prev_launch_arm = False
        self.intent = 'FULL'        # FULL | REDUCED (set on LAUNCH entry)

    def reset(self):
        self.state = LaunchState.IDLE
        self.s_start = None
        self.t_launch = None
        self.launch_arm_consumed = False
        self._prev_launch_arm = False

    def update(self, *,
               t_now,
               autonomous_armed,
               launch_arm,
               go_signal,
               v_now,
               s_now,
               v_ref,
               steer_now,
               kappa_ahead,
               joy_alive,
               proximity_intent='FULL',
               proximity_runtime_abort=False,
               reduced_s_launch_m=None):
        """Step the FSM. Returns the new LaunchState."""

        # rising edge of launch_arm clears the one-shot consumed flag
        if launch_arm and not self._prev_launch_arm:
            self.launch_arm_consumed = False
        # falling edge: also clear so next ON is fresh
        if not launch_arm:
            self.launch_arm_consumed = False
        self._prev_launch_arm = launch_arm

        # global safety gates: any of these collapse to IDLE/NORMAL
        if not joy_alive:
            self._collapse_to_safe()
            return self.state
        if not autonomous_armed:
            # human is in manual or autonomous OFF: forget everything
            self.reset()
            return self.state

        # state-specific logic
        if self.state == LaunchState.IDLE:
            # autonomous is on => move forward
            self.state = LaunchState.DISARMED

        if self.state == LaunchState.DISARMED:
            if (launch_arm
                    and not self.launch_arm_consumed
                    and abs(v_now) < self.v_stationary):
                self.state = LaunchState.ARMED

        elif self.state == LaunchState.ARMED:
            if not launch_arm:
                self.state = LaunchState.DISARMED
            elif abs(v_now) >= self.v_stationary:
                # rolled while armed -> drop back; human re-toggles
                self.state = LaunchState.DISARMED
            elif go_signal:
                # consult proximity guard for intent
                self.launch_arm_consumed = True
                if proximity_intent == 'ABORT':
                    # skip LAUNCH, hand directly to NORMAL
                    self.state = LaunchState.NORMAL
                    self.intent = 'ABORTED'
                else:
                    self.s_start = float(s_now)
                    self.t_launch = float(t_now)
                    self.intent = proximity_intent  # FULL or REDUCED
                    if proximity_intent == 'REDUCED' and reduced_s_launch_m is not None:
                        self.s_launch_m = float(reduced_s_launch_m)
                    else:
                        self.s_launch_m = self.s_launch_m_full
                    self.state = LaunchState.LAUNCH

        elif self.state == LaunchState.LAUNCH:
            if proximity_runtime_abort:
                self.state = LaunchState.NORMAL
            elif self._should_exit_launch(t_now, v_now, s_now, v_ref,
                                          steer_now, kappa_ahead):
                self.state = LaunchState.NORMAL

        elif self.state == LaunchState.NORMAL:
            # stay in NORMAL until autonomous goes off (handled at top) or
            # the user manually re-arms (DISARMED) by toggling launch_arm OFF
            # -> we move back to DISARMED to allow another launch.
            if not launch_arm:
                self.state = LaunchState.DISARMED

        return self.state

    def _collapse_to_safe(self):
        if self.state == LaunchState.LAUNCH:
            self.state = LaunchState.NORMAL  # graceful hand-off, not abrupt cut
        else:
            self.state = LaunchState.IDLE
        self.s_start = None
        self.t_launch = None

    def _should_exit_launch(self, t_now, v_now, s_now, v_ref,
                            steer_now, kappa_ahead):
        if self.t_launch is None or self.s_start is None:
            return True
        if (t_now - self.t_launch) > self.t_max_s:
            return True
        ds = s_now - self.s_start  # caller wraps if needed
        if ds is not None and ds > self.s_launch_m:
            return True
        if v_ref > 1e-3 and v_now > self.v_ratio_exit * v_ref:
            return True
        if abs(steer_now) > self.steer_exit_rad:
            return True
        if abs(kappa_ahead) > self.kappa_exit:
            return True
        return False

    # convenience for logging
    def in_launch(self):
        return self.state == LaunchState.LAUNCH

    def info(self, t_now):
        if self.t_launch is None:
            return {'state': self.state.name, 't_in_launch': 0.0}
        return {'state': self.state.name,
                't_in_launch': max(0.0, t_now - self.t_launch)}
