from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, List

from states_types import StateType
import rospy
from f110_msgs.msg import Wpnt
import states


if TYPE_CHECKING:
    from state_machine_node import StateMachine
    from state_helper_for_smart import SmartStaticChecker  # ===== HJ ADDED =====
else:
    # ===== HJ ADDED: Runtime import =====
    from state_helper_for_smart import SmartStaticChecker
    # ===== HJ ADDED END =====

# ===== HJ ADDED: Global Smart Static checker instance =====
_smart_static_checker = None
# ===== HJ ADDED END =====

close_threshold_smart = 0.05
close_threshold_gb = 0.05

"""
Transitions should loosely follow the following template (basically a match-case)

if (logic sum of bools obtained by methods of state_machine):
    return StateType.<DESIRED STATE>
elif (e.g. state_machine.obstacles are near):
    return StateType.<ANOTHER DESIRED STATE>
...

NOTE: ideally put the most common cases on top of the match-case

NOTE 2: notice that, when implementing new states, if an attribute/condition in the
    StateMachine is not available, your IDE will tell you, but only if you have a smart
    enough IDE. So use vscode, pycharm, fleet or whatever has specific python syntax highlights.

NOTE 3: transistions must not have side effects on the state machine!
    i.e. any attribute of the state machine should not be modified in the transitions.
"""

# ===== HJ MODIFIED: Complete mode separation - each mode has its own closed loop =====
def GlobalTrackingTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.GB_TRACK`

    Routes to completely separate Smart or GB mode transitions.
    """
    # rospy.logwarn("========== GlobalTrackingTransition ENTERED ==========")

    smart_active = state_machine.smart_static_active

    # Complete mode switching - call separate function sets
    if smart_active:
        close_to_smart = state_machine._check_close_to_smart_static_path(close_threshold_smart) * state_machine._check_close_to_smart_static_path_heading(20)
        # rospy.logwarn(f"[GlobalTracking] SMART MODE: close_to_smart={close_to_smart}, num_obs={len(state_machine.cur_obstacles_in_interest)}")

        if len(state_machine.cur_obstacles_in_interest) == 0:
            return NonObstacleTransition_SmartMode(state_machine, close_to_smart)
        else:
            return ObstacleTransition_SmartMode(state_machine, close_to_smart)
    else:
        close_to_gb = state_machine._check_close_to_raceline(close_threshold_gb) * state_machine._check_close_to_raceline_heading(20)
        # rospy.logwarn(f"[GlobalTracking] GB MODE: close_to_gb={close_to_gb}, num_obs={len(state_machine.cur_obstacles_in_interest)}")

        if len(state_machine.cur_obstacles_in_interest) == 0:
            return NonObstacleTransition_GBMode(state_machine, close_to_gb)
        else:
            return ObstacleTransition_GBMode(state_machine, close_to_gb)


def RecoveryTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.RECOVERY`

    Recovery operates within the mode's closed loop.
    """
    # rospy.logwarn("========== RecoveryTransition ENTERED ==========")
    recovery_sustainability = state_machine._check_sustainability(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)

    smart_active = state_machine.smart_static_active

    # Check proximity based on current mode only
    if smart_active:
        close_to_smart = state_machine._check_close_to_smart_static_path(close_threshold_smart) * state_machine._check_close_to_smart_static_path_heading(20)
        # rospy.logwarn(f"[Recovery] SMART MODE: close_to_smart={close_to_smart}, sustainable={recovery_sustainability}")

        if recovery_sustainability and not close_to_smart:
            return StateType.RECOVERY, StateType.RECOVERY
        # Recovery ended - return to Smart mode closed loop
        return SmartStaticTransition(state_machine)
    else:
        close_to_gb = state_machine._check_close_to_raceline(close_threshold_gb) * state_machine._check_close_to_raceline_heading(20)
        # rospy.logwarn(f"[Recovery] GB MODE: close_to_gb={close_to_gb}, sustainable={recovery_sustainability}")

        if recovery_sustainability and not close_to_gb:
            return StateType.RECOVERY, StateType.RECOVERY
        # Recovery ended - return to GB mode closed loop
        return GlobalTrackingTransition(state_machine)


def TrailingTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.TRAILING`"""
    # rospy.logwarn("========== TrailingTransition ENTERED ==========")

    smart_active = state_machine.smart_static_active

    if smart_active:
        close_to_smart = state_machine._check_close_to_smart_static_path(close_threshold_smart) * state_machine._check_close_to_smart_static_path_heading(20)
        # rospy.logwarn(f"[Trailing] SMART MODE: close_to_smart={close_to_smart}")

        if len(state_machine.cur_obstacles_in_interest) == 0:
            return NonObstacleTransition_SmartMode(state_machine, close_to_smart)
        else:
            if state_machine._check_ftg():
                return StateType.FTGONLY, StateType.FTGONLY
            return ObstacleTransition_SmartMode(state_machine, close_to_smart)
    else:
        close_to_gb = state_machine._check_close_to_raceline(close_threshold_gb) * state_machine._check_close_to_raceline_heading(20)
        # rospy.logwarn(f"[Trailing] GB MODE: close_to_gb={close_to_gb}")

        if len(state_machine.cur_obstacles_in_interest) == 0:
            return NonObstacleTransition_GBMode(state_machine, close_to_gb)
        else:
            if state_machine._check_ftg():
                return StateType.FTGONLY, StateType.FTGONLY
            return ObstacleTransition_GBMode(state_machine, close_to_gb)


def OvertakingTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.OVERTAKE`"""
    # rospy.logwarn("========== OvertakingTransition ENTERED ==========")
    ot_sustainability = state_machine._check_overtaking_mode_sustainability()
    enemy_in_front = state_machine._check_enemy_in_front()

    if ot_sustainability and enemy_in_front:
        state_machine.overtaking_ttl_count = 0
        return StateType.OVERTAKE, StateType.OVERTAKE
    if ot_sustainability and state_machine.overtaking_ttl_count < state_machine.overtaking_ttl_count_threshold:
        state_machine.overtaking_ttl_count += 1
        return StateType.OVERTAKE, StateType.OVERTAKE
    state_machine.overtaking_ttl_count = 0

    # Overtaking ended - return to appropriate mode's closed loop
    smart_active = state_machine.smart_static_active
    if smart_active:
        # rospy.logwarn(f"[Overtaking→SMART MODE]")
        return SmartStaticTransition(state_machine)
    else:
        # rospy.logwarn(f"[Overtaking→GB MODE]")
        return GlobalTrackingTransition(state_machine)


def StartTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.START`"""
    start_free = state_machine._check_free_cartesian(state_machine.cur_start_wpnts)
    on_spline = state_machine._check_on_spline(state_machine.cur_start_wpnts)

    if start_free and on_spline:
        return StateType.START, StateType.START
    else:
        state_machine.cur_start_wpnts.is_init = False
        return GlobalTrackingTransition(state_machine)


def FTGOnlyTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.FTGONLY`"""
    close_to_raceline = state_machine._check_close_to_raceline(close_threshold_smart) * state_machine._check_close_to_raceline_heading(20)

    if len(state_machine.cur_obstacles_in_interest) == 0:
        # FTGOnly always uses GB mode logic
        return NonObstacleTransition_GBMode(state_machine, close_to_raceline)
    else:
        if close_to_raceline and state_machine._check_free_frenet(state_machine.cur_gb_wpnts):
            return StateType.GB_TRACK, StateType.GB_TRACK

        recovery_availability = state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
        if (recovery_availability and state_machine._check_free_frenet(state_machine.cur_recovery_wpnts)):
            return StateType.RECOVERY, StateType.RECOVERY

        if state_machine._check_overtaking_mode() or state_machine._check_static_overtaking_mode():
            return StateType.OVERTAKE, StateType.OVERTAKE
        else:
            return StateType.FTGONLY, StateType.FTGONLY


def SmartStaticTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.SMART_STATIC`

    Entry point for Smart mode's closed loop.
    Only considers Smart Static path - GB raceline is completely ignored.
    """
    # rospy.logwarn("========== SmartStaticTransition ENTERED ==========")

    close_to_smart = state_machine._check_close_to_smart_static_path(close_threshold_smart) * state_machine._check_close_to_smart_static_path_heading(20)
    num_obstacles = len(state_machine.cur_obstacles_in_interest)

    # rospy.logwarn(f"[SMART_STATIC] close_to_smart={close_to_smart}, num_obstacles={num_obstacles}")

    # Delegate to Smart mode transitions only
    if num_obstacles == 0:
        return NonObstacleTransition_SmartMode(state_machine, close_to_smart)
    else:
        return ObstacleTransition_SmartMode(state_machine, close_to_smart)


##################################################################################################################
##################################################################################################################
# ===== SMART MODE CLOSED LOOP - Only considers Smart Static path =====

def NonObstacleTransition_SmartMode(state_machine: StateMachine, close_to_smart: bool) -> Tuple[StateType, StateType]:
    """Handle no obstacles case in Smart Static mode

    CLOSED LOOP: Only considers Smart Static path.
    GB raceline is completely ignored.

    Args:
        close_to_smart: True if close to Smart Static path
    """
    # rospy.logwarn(f">>> NonObstacleTransition_SmartMode: close_to_smart={close_to_smart}")

    wpnts_valid = state_machine._check_latest_wpnts(
        state_machine.smart_static_wpnts,
        state_machine.cur_smart_static_avoidance_wpnts)

    # rospy.logwarn(f"[NonObstacle_Smart] wpnts_valid={wpnts_valid}, close_to_smart={close_to_smart}")

    # Priority 1: Smart path available and close - use it
    if wpnts_valid and close_to_smart:
        # rospy.logwarn(f"[NonObstacle_Smart→SMART_STATIC] ✓ Valid & close")
        return StateType.SMART_STATIC, StateType.SMART_STATIC

    # Priority 2: Smart path valid but not close - stay in Smart, use recovery to return
    if wpnts_valid:
        # rospy.logwarn(f"[NonObstacle_Smart→SMART_STATIC] ✓ Valid (not close)")
        return StateType.SMART_STATIC, StateType.SMART_STATIC

    # Priority 3: Smart path invalid - use recovery to get back
    if state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts):
        if state_machine._check_on_spline(state_machine.cur_recovery_wpnts):
            # rospy.logwarn(f"[NonObstacle_Smart→RECOVERY] Smart invalid, recovering")
            return StateType.RECOVERY, StateType.RECOVERY

    # Priority 4: No valid path - lost line (still return SMART_STATIC trajectory to stay in loop)
    # rospy.logwarn(f"[NonObstacle_Smart→LOSTLINE] Lost line")
    return StateType.LOSTLINE, StateType.SMART_STATIC


def ObstacleTransition_SmartMode(state_machine: StateMachine, close_to_smart: bool) -> Tuple[StateType, StateType]:
    """Handle obstacles present case in Smart Static mode

    CLOSED LOOP: Only considers Smart Static path.
    GB raceline and GB path free status are completely ignored.

    Args:
        close_to_smart: True if close to Smart Static path
    """
    # rospy.logwarn(f">>> ObstacleTransition_SmartMode: close_to_smart={close_to_smart}, num_obs={len(state_machine.cur_obstacles_in_interest)}")

    wpnts_valid = state_machine._check_latest_wpnts(
        state_machine.smart_static_wpnts,
        state_machine.cur_smart_static_avoidance_wpnts)
    smart_path_free = state_machine._check_free_frenet(state_machine.cur_smart_static_avoidance_wpnts)

    # rospy.logwarn(f"[Obstacle_Smart] wpnts_valid={wpnts_valid}, close={close_to_smart}, path_free={smart_path_free}")

    # Priority 1: Smart path available, close, and free - use it
    if wpnts_valid and close_to_smart and smart_path_free:
        # rospy.logwarn(f"[Obstacle_Smart→SMART_STATIC] ✓ Valid, close & free")
        return StateType.SMART_STATIC, StateType.SMART_STATIC

    # Priority 2: Check recovery availability (only if not close to Smart path) - SAME AS GB MODE
    recovery_availability = False
    if not close_to_smart:
        recovery_availability = state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
        if (recovery_availability and state_machine._check_free_frenet(state_machine.cur_recovery_wpnts)):
            # rospy.logwarn(f"[Obstacle_Smart→RECOVERY] Not close, recovery available")
            return StateType.RECOVERY, StateType.RECOVERY

    # Priority 3: Overtaking check
    if state_machine._check_overtaking_mode() or state_machine._check_static_overtaking_mode():
        # rospy.logwarn(f"[Obstacle_Smart→OVERTAKE] Overtaking triggered")
        return StateType.OVERTAKE, StateType.OVERTAKE

    # Priority 4: TRAILING state - Smart mode always uses Smart path
    if wpnts_valid and close_to_smart:
        # rospy.logwarn(f"[Obstacle_Smart→TRAILING+SMART] Valid & close")
        return StateType.TRAILING, StateType.SMART_STATIC
    elif wpnts_valid:
        # Smart path valid but not close - STILL use Smart (don't fallback!)
        # rospy.logwarn(f"[Obstacle_Smart→TRAILING+SMART] Valid (not close) - staying in Smart")
        return StateType.TRAILING, StateType.SMART_STATIC
    elif recovery_availability:
        # rospy.logwarn(f"[Obstacle_Smart→TRAILING+RECOVERY] Smart invalid, using recovery")
        return StateType.TRAILING, StateType.RECOVERY
    else:
        # Last resort - Smart invalid, no recovery, still stay in Smart mode
        # rospy.logwarn(f"[Obstacle_Smart→TRAILING+SMART] Fallback to Smart (no alternatives)")
        return StateType.TRAILING, StateType.SMART_STATIC


##################################################################################################################
# ===== GB MODE CLOSED LOOP - Only considers GB raceline =====

def NonObstacleTransition_GBMode(state_machine: StateMachine, close_to_gb: bool) -> Tuple[StateType, StateType]:
    """Handle no obstacles case in GB tracking mode

    CLOSED LOOP: Only considers GB raceline.
    Smart Static path is completely ignored.

    Args:
        close_to_gb: True if close to GB raceline
    """
    # rospy.logwarn(f">>> NonObstacleTransition_GBMode: close_to_gb={close_to_gb}")

    # Priority 1: Close to GB raceline - use it
    if close_to_gb:
        # rospy.logwarn(f"[NonObstacle_GB→GB_TRACK] ✓ Close to GB")
        return StateType.GB_TRACK, StateType.GB_TRACK

    # Priority 2: Not close to GB - use recovery to get back
    if state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts):
        if state_machine._check_on_spline(state_machine.cur_recovery_wpnts):
            # rospy.logwarn(f"[NonObstacle_GB→RECOVERY] Not close, recovering")
            return StateType.RECOVERY, StateType.RECOVERY

    # Priority 3: No valid path - lost line
    # rospy.logwarn(f"[NonObstacle_GB→LOSTLINE] Lost line")
    return StateType.LOSTLINE, StateType.GB_TRACK


def ObstacleTransition_GBMode(state_machine: StateMachine, close_to_gb: bool) -> Tuple[StateType, StateType]:
    """Handle obstacles present case in GB tracking mode

    CLOSED LOOP: Only considers GB raceline and GB path.
    Smart Static path is completely ignored.

    Args:
        close_to_gb: True if close to GB raceline
    """
    # rospy.logwarn(f">>> ObstacleTransition_GBMode: close_to_gb={close_to_gb}, num_obs={len(state_machine.cur_obstacles_in_interest)}")

    gb_path_free = state_machine._check_free_frenet(state_machine.cur_gb_wpnts)

    # rospy.logwarn(f"[Obstacle_GB] close_to_gb={close_to_gb}, gb_path_free={gb_path_free}")

    # Priority 1: GB path close and free - use it
    if close_to_gb and gb_path_free:
        # rospy.logwarn(f"[Obstacle_GB→GB_TRACK] ✓ Close & free")
        return StateType.GB_TRACK, StateType.GB_TRACK

    # Check recovery availability (only if not close to GB)
    recovery_availability = False
    if not close_to_gb:
        recovery_availability = state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
        if (recovery_availability and state_machine._check_free_frenet(state_machine.cur_recovery_wpnts)):
            # rospy.logwarn(f"[Obstacle_GB→RECOVERY] Not close, recovery available")
            return StateType.RECOVERY, StateType.RECOVERY

    # Priority 2: Overtaking check
    if state_machine._check_overtaking_mode() or state_machine._check_static_overtaking_mode():
        # rospy.logwarn(f"[Obstacle_GB→OVERTAKE] Overtaking triggered")
        return StateType.OVERTAKE, StateType.OVERTAKE

    # Priority 3: TRAILING state - GB mode logic
    if close_to_gb:
        # rospy.logwarn(f"[Obstacle_GB→TRAILING+GB] Close to GB")
        return StateType.TRAILING, StateType.GB_TRACK
    elif recovery_availability:
        # rospy.logwarn(f"[Obstacle_GB→TRAILING+RECOVERY] Not close, using recovery")
        return StateType.TRAILING, StateType.RECOVERY
    elif gb_path_free:
        # rospy.logwarn(f"[Obstacle_GB→TRAILING+GB] GB path free")
        return StateType.TRAILING, StateType.GB_TRACK
    else:
        # Default fallback
        # rospy.logwarn(f"[Obstacle_GB→TRAILING+GB] Fallback to GB")
        return StateType.TRAILING, StateType.GB_TRACK
