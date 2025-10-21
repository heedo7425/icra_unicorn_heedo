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
def GlobalTrackingTransition(state_machine: StateMachine, close_to_raceline = None) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.GB_TRACK`"""
    if close_to_raceline is None:
        close_to_raceline = state_machine._check_close_to_raceline()

    # ===== HJ ADDED: Detailed debug logging for Smart Static transition =====
    smart_active = state_machine.smart_static_active
    smart_wpnts_none = state_machine.smart_static_wpnts is None
    smart_wpnts_len = len(state_machine.smart_static_wpnts.wpnts) if state_machine.smart_static_wpnts is not None else 0

    wpnts_valid = state_machine._check_latest_wpnts(
        state_machine.smart_static_wpnts,
        state_machine.cur_smart_static_avoidance_wpnts)

    path_free = state_machine._check_free_frenet(state_machine.cur_smart_static_avoidance_wpnts)

    num_obstacles = len(state_machine.cur_obstacles_in_interest)

    # ===== HJ DEBUG: Test if logging works =====
    rospy.logwarn_throttle(0.5, f"[GB_TRACK Transition] CALLED!")
    rospy.logwarn_throttle(1.0,
        f"[GB_TRACK→?] Smart Static Check:\n"
        f"  active={smart_active}, wpnts_valid={wpnts_valid}, path_free={path_free}\n"
        f"  wpnts={'None' if smart_wpnts_none else f'{smart_wpnts_len}pts'}, "
        f"  close_raceline={close_to_raceline}, obs={num_obstacles}")

    # Check if should transition to Smart Static Fixed Path
    if smart_active and wpnts_valid and path_free:
        rospy.logwarn(f"[GB_TRACK→SMART_STATIC] ✓ All conditions met!")
        return StateType.SMART_STATIC, StateType.SMART_STATIC
    # ===== HJ ADDED END =====

    if len(state_machine.cur_obstacles_in_interest) == 0:
        return NonObstacleTransition(state_machine, close_to_raceline)

    else:
        return ObstacleTransition(state_machine, close_to_raceline)
   
def RecoveryTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.RECOVERY`"""
    recovery_sustainability = state_machine._check_sustainability(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
    close_to_raceline = state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)

    if recovery_sustainability and not close_to_raceline:
        return StateType.RECOVERY, StateType.RECOVERY

    # Recovery ended - return to GB tracking
    return GlobalTrackingTransition(state_machine, close_to_raceline)

def TrailingTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.TRAILING`"""
    close_to_raceline = state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)

    if len(state_machine.cur_obstacles_in_interest) == 0:
        return NonObstacleTransition(state_machine, close_to_raceline)
    else:
        if state_machine._check_ftg():
            return StateType.FTGONLY, StateType.FTGONLY
        return ObstacleTransition(state_machine, close_to_raceline)
            
def OvertakingTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.OVERTAKE`"""
    ot_sustainability = state_machine._check_overtaking_mode_sustainability()
    enemy_in_front = state_machine._check_enemy_in_front()

    if ot_sustainability and enemy_in_front:
        state_machine.overtaking_ttl_count = 0
        return StateType.OVERTAKE, StateType.OVERTAKE
    if ot_sustainability and state_machine.overtaking_ttl_count < state_machine.overtaking_ttl_count_threshold:
        state_machine.overtaking_ttl_count += 1
        return StateType.OVERTAKE, StateType.OVERTAKE
    state_machine.overtaking_ttl_count = 0

    # Overtaking ended - return to GB tracking
    close_to_raceline = state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)
    return GlobalTrackingTransition(state_machine, close_to_raceline)

def StartTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.START`"""
    start_free = state_machine._check_free_cartesian(state_machine.cur_start_wpnts)
    on_spline = state_machine._check_on_spline(state_machine.cur_start_wpnts)

    if start_free and on_spline:
        return StateType.START, StateType.START
    else:
        close_to_raceline = state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)
        state_machine.cur_start_wpnts.is_init = False
        return GlobalTrackingTransition(state_machine, close_to_raceline)
            
def FTGOnlyTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.FTGONLY`"""
    close_to_raceline = state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)

    if len(state_machine.cur_obstacles_in_interest) == 0:
        return NonObstacleTransition(state_machine, close_to_raceline)

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

# ===== HJ ADDED: Smart Static Fixed Path Transition =====
def SmartStaticTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.SMART_STATIC`

    Follows Smart Static Fixed Path (GB Optimizer output).
    Uses GB Frenet coordinates (from topics).
    Checks proximity to Smart Static path (not GB raceline).
    """
    # ===== HJ ADDED: Detailed debug logging =====
    smart_active = state_machine.smart_static_active
    smart_wpnts_len = len(state_machine.cur_smart_static_avoidance_wpnts.list)

    # Check if close to Smart Static path (both position and heading)
    close_to_pos = state_machine._check_close_to_smart_static_path(0.5)
    close_to_heading = state_machine._check_close_to_smart_static_path_heading(20)
    close_to_smart_path = close_to_pos * close_to_heading

    num_obstacles = len(state_machine.cur_obstacles_in_interest)

    rospy.loginfo_throttle(1.0,
        f"[SMART_STATIC→?] Transition Check:\n"
        f"  smart_active: {smart_active}\n"
        f"  smart_wpnts_len: {smart_wpnts_len}\n"
        f"  close_to_pos: {close_to_pos}\n"
        f"  close_to_heading: {close_to_heading}\n"
        f"  close_to_smart_path: {close_to_smart_path}\n"
        f"  num_obstacles: {num_obstacles}")

    # Delegate to standard transitions - they will check smart_static_active
    if num_obstacles == 0:
        result = NonObstacleTransition(state_machine, close_to_smart_path)
        rospy.loginfo_throttle(1.0, f"[SMART_STATIC→?] No obstacles, result: {result}")
        return result
    else:
        result = ObstacleTransition(state_machine, close_to_smart_path)
        rospy.loginfo_throttle(1.0, f"[SMART_STATIC→?] With obstacles, result: {result}")
        return result
    # ===== HJ ADDED END =====
# ===== HJ ADDED END =====

##################################################################################################################
##################################################################################################################

def NonObstacleTransition(state_machine: StateMachine, close_to_raceline) -> Tuple[StateType, StateType]:
    """Handle no obstacles case"""
    # ===== HJ ADDED: Debug logging =====
    smart_active = state_machine.smart_static_active
    wpnts_valid = state_machine._check_latest_wpnts(
        state_machine.smart_static_wpnts,
        state_machine.cur_smart_static_avoidance_wpnts)

    rospy.loginfo_throttle(1.0,
        f"[NonObstacleTransition] Check:\n"
        f"  smart_active: {smart_active}\n"
        f"  wpnts_valid: {wpnts_valid}\n"
        f"  close_to_raceline: {close_to_raceline}")

    # Check if Smart Static path is available and close enough to use
    if smart_active and wpnts_valid and close_to_raceline:
        # Smart Static available and close to it - follow it
        rospy.logwarn(f"[NonObstacleTransition→SMART_STATIC] ✓ All conditions met!")
        return StateType.SMART_STATIC, StateType.SMART_STATIC
    else:
        rospy.logwarn_throttle(1.0,
            f"[NonObstacleTransition] Smart Static NOT chosen: "
            f"active={smart_active}, valid={wpnts_valid}, close={close_to_raceline}")
    # ===== HJ ADDED END =====

    # No Smart Static or not close - use GB
    if close_to_raceline:
        rospy.logwarn_throttle(1.0, f"[NonObstacleTransition→GB_TRACK] Using GB track (close_to_raceline={close_to_raceline})")
        return StateType.GB_TRACK, StateType.GB_TRACK

    if state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts):
        if state_machine._check_on_spline(state_machine.cur_recovery_wpnts):
            rospy.loginfo_throttle(1.0, f"[NonObstacleTransition→RECOVERY] Using recovery")
            return StateType.RECOVERY, StateType.RECOVERY

    rospy.loginfo_throttle(1.0, f"[NonObstacleTransition→LOSTLINE] Lost line")
    return StateType.LOSTLINE, StateType.GB_TRACK
    
def ObstacleTransition(state_machine: StateMachine, close_to_raceline) -> Tuple[StateType, StateType]:
    """Handle obstacles present case"""
    # ===== HJ ADDED: Debug logging =====
    smart_active = state_machine.smart_static_active
    wpnts_valid = state_machine._check_latest_wpnts(
        state_machine.smart_static_wpnts,
        state_machine.cur_smart_static_avoidance_wpnts)
    smart_path_free = state_machine._check_free_frenet(state_machine.cur_smart_static_avoidance_wpnts)
    gb_path_free = state_machine._check_free_frenet(state_machine.cur_gb_wpnts)

    rospy.loginfo_throttle(1.0,
        f"[ObstacleTransition] Check:\n"
        f"  smart_active: {smart_active}\n"
        f"  wpnts_valid: {wpnts_valid}\n"
        f"  close_to_raceline: {close_to_raceline}\n"
        f"  smart_path_free: {smart_path_free}\n"
        f"  gb_path_free: {gb_path_free}")

    # Check if Smart Static path is available, close, and free
    if smart_active and wpnts_valid and close_to_raceline and smart_path_free:
        # Smart Static path available, close, and free - use it
        rospy.logwarn(f"[ObstacleTransition→SMART_STATIC] ✓ All conditions met!")
        return StateType.SMART_STATIC, StateType.SMART_STATIC
    # ===== HJ ADDED END =====

    # Smart Static not available/close/free - check GB path
    if close_to_raceline and gb_path_free:
        rospy.loginfo_throttle(1.0, f"[ObstacleTransition→GB_TRACK] GB path free")
        return StateType.GB_TRACK, StateType.GB_TRACK

    recovery_availability = False
    if not close_to_raceline:
        recovery_availability = state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
        if (recovery_availability and state_machine._check_free_frenet(state_machine.cur_recovery_wpnts)):
            rospy.loginfo_throttle(1.0, f"[ObstacleTransition→RECOVERY] Recovery path available")
            return StateType.RECOVERY, StateType.RECOVERY

    if state_machine._check_overtaking_mode() or state_machine._check_static_overtaking_mode():
        rospy.loginfo_throttle(1.0, f"[ObstacleTransition→OVERTAKE] Overtaking mode")
        return StateType.OVERTAKE, StateType.OVERTAKE

    else:
        if close_to_raceline:
            rospy.loginfo_throttle(1.0, f"[ObstacleTransition→TRAILING] Close to raceline, trailing")
            return StateType.TRAILING, StateType.GB_TRACK
        elif recovery_availability:
            rospy.loginfo_throttle(1.0, f"[ObstacleTransition→TRAILING] Recovery available, trailing")
            return StateType.TRAILING, StateType.RECOVERY
        elif gb_path_free:
            rospy.loginfo_throttle(1.0, f"[ObstacleTransition→TRAILING] GB path free, trailing")
            return StateType.TRAILING, StateType.GB_TRACK
        else:
            rospy.loginfo_throttle(1.0, f"[ObstacleTransition→TRAILING] Default trailing")
            return StateType.TRAILING, StateType.GB_TRACK
        