#!/usr/bin/env python3
"""
Visualize do_spline inputs in RViz as markers and debug do_spline
"""
import rospy
import pickle
import sys
import numpy as np
import copy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.interpolate import BPoly

# Add paths for imports
sys.path.append('/home/hj/unicorn_ws/UNICORN/f110_utils/libs/frenet_conversion/src')
sys.path.append('/home/hj/unicorn_ws/UNICORN/f110_utils/libs/trajectory_planning_helpers')

from frenet_converter.frenet_converter import FrenetConverter
import trajectory_planning_helpers as tph
from f110_msgs.msg import WpntArray

# Global variable to store GB global waypoints
gb_global_waypoints = None

def gb_waypoints_callback(msg):
    global gb_global_waypoints
    gb_global_waypoints = msg.wpnts

# ===== HELPER FUNCTIONS FROM smart_static_avoidance_node.py =====
def _more_space(obstacle, gb_wpnts, obs_s_idx, evasion_dist=0.5, spline_bound_mindist=0.5):
    """Determine which side has more space to evade obstacle"""
    wpnt_d_left = gb_wpnts[obs_s_idx].d_left
    wpnt_d_right = gb_wpnts[obs_s_idx].d_right

    print(f"_more_space: wpnt[{obs_s_idx}] d_left={wpnt_d_left:.2f}, d_right={wpnt_d_right:.2f}, "
          f"obs d_left={obstacle.d_left:.2f}, d_right={obstacle.d_right:.2f}")

    left_gap = abs(wpnt_d_left - obstacle.d_left)
    right_gap = abs(wpnt_d_right + obstacle.d_right)
    min_space = evasion_dist + spline_bound_mindist

    if right_gap > min_space and left_gap < min_space:
        d_apex_right = obstacle.d_right - evasion_dist
        if d_apex_right > 0:
            d_apex_right = 0
        return "right", d_apex_right
    elif left_gap > min_space and right_gap < min_space:
        d_apex_left = obstacle.d_left + evasion_dist
        if d_apex_left < 0:
            d_apex_left = 0
        return "left", d_apex_left
    else:
        candidate_d_apex_left = obstacle.d_left + evasion_dist
        candidate_d_apex_right = obstacle.d_right - evasion_dist

        if abs(candidate_d_apex_left) <= abs(candidate_d_apex_right):
            if candidate_d_apex_left < 0:
                candidate_d_apex_left = 0
            return "left", candidate_d_apex_left
        else:
            if candidate_d_apex_right > 0:
                candidate_d_apex_right = 0
            return "right", candidate_d_apex_right

def do_spline_debug(obs, gb_wpnts, fixed_converter, cur_x, cur_y, cur_yaw,
                    spline_scale=1.5, post_min_dist=2.0, post_max_dist=10.0,
                    sampling_dist=0.1, evasion_dist=0.5, spline_bound_mindist=0.5,
                    map_filter=None):
    """
    Debug version of do_spline with detailed logging
    """
    print("="*80)
    print("DO_SPLINE DEBUG START")
    print("="*80)

    # Calculate parameters
    ref_max_idx = len(gb_wpnts)

    # Fixed mode: Convert waypoints to Fixed Frenet
    wp0_frenet = fixed_converter.get_frenet(
        np.array([gb_wpnts[0].x_m]), np.array([gb_wpnts[0].y_m]))
    wp1_frenet = fixed_converter.get_frenet(
        np.array([gb_wpnts[1].x_m]), np.array([gb_wpnts[1].y_m]))
    wpnt_dist = float(wp1_frenet[0] - wp0_frenet[0])

    # Calculate ref_max_s as cumulative path length
    total_dist = 0.0
    for i in range(len(gb_wpnts)):
        dx = gb_wpnts[i].x_m - gb_wpnts[i-1].x_m
        dy = gb_wpnts[i].y_m - gb_wpnts[i-1].y_m
        total_dist += np.sqrt(dx**2 + dy**2)
    ref_max_s = total_dist

    print(f"wpnt_dist={wpnt_dist:.4f}m, ref_max_s={ref_max_s:.2f}m, ref_max_idx={ref_max_idx}")

    # Convert current position to Fixed Frenet
    result_ego = fixed_converter.get_frenet(np.array([cur_x]), np.array([cur_y]))
    cur_s_ref = float(result_ego[0]) % ref_max_s
    cur_d_ref = float(result_ego[1])

    # Normalize obstacle s
    obs_s_normalized = obs.s_center % ref_max_s
    obs = copy.deepcopy(obs)
    obs.s_center = obs_s_normalized

    print(f"\n>>> INITIAL POSITIONS:")
    print(f"    Current XY: ({cur_x:.4f}, {cur_y:.4f})")
    print(f"    Current Frenet: s={cur_s_ref:.4f}m, d={cur_d_ref:.4f}m")
    print(f"    Obstacle XY: s={obs.s_center:.4f}m, d={obs.d_center:.4f}m")
    print(f"    Obstacle bounds: d_left={obs.d_left:.4f}m, d_right={obs.d_right:.4f}m")

    # Check obstacle distance
    pre_dist_raw = obs.s_center - cur_s_ref
    pre_dist = pre_dist_raw % ref_max_s

    print(f"pre_dist_raw={pre_dist_raw:.2f}m, pre_dist={pre_dist:.2f}m")

    if pre_dist > ref_max_s / 2:
        print(f"ABORT: Obstacle behind! pre_dist={pre_dist:.2f}m > ref_max_s/2={ref_max_s/2:.2f}m")
        return None, None

    if pre_dist < 0.5:
        print(f"ABORT: Obstacle too close! pre_dist={pre_dist:.2f}m < 0.5m")
        return None, None

    obs_s_idx = int(obs.s_center / wpnt_dist) % ref_max_idx
    print(f"\n>>> OBSTACLE INDEX: obs_s_idx={obs_s_idx}")

    # Determine evasion side
    print(f"\n>>> BOUNDARY INFO AT obs_s_idx={obs_s_idx}:")
    print(f"    gb_wpnts[{obs_s_idx}].s_m = {gb_wpnts[obs_s_idx].s_m:.4f}m")
    print(f"    gb_wpnts[{obs_s_idx}].d_left = {gb_wpnts[obs_s_idx].d_left:.4f}m")
    print(f"    gb_wpnts[{obs_s_idx}].d_right = {gb_wpnts[obs_s_idx].d_right:.4f}m")
    print(f"    obs.d_left = {obs.d_left:.4f}m")
    print(f"    obs.d_right = {obs.d_right:.4f}m")

    wpnt_d_left = gb_wpnts[obs_s_idx].d_left
    wpnt_d_right = gb_wpnts[obs_s_idx].d_right
    left_gap = abs(wpnt_d_left - obs.d_left)
    right_gap = abs(wpnt_d_right + obs.d_right)
    min_space = evasion_dist + spline_bound_mindist

    print(f"\n>>> GAP CALCULATION:")
    print(f"    left_gap = |{wpnt_d_left:.4f} - {obs.d_left:.4f}| = {left_gap:.4f}m")
    print(f"    right_gap = |{wpnt_d_right:.4f} + {obs.d_right:.4f}| = {right_gap:.4f}m")
    print(f"    min_space = {evasion_dist:.4f} + {spline_bound_mindist:.4f} = {min_space:.4f}m")

    more_space, d_apex = _more_space(obs, gb_wpnts, obs_s_idx, evasion_dist, spline_bound_mindist)
    print(f"\n>>> EVASION DECISION: {more_space} side, d_apex={d_apex:.4f}m")

    s_list = [obs.s_center]
    d_list = [d_apex]

    post_dist = min(min(max(pre_dist, post_min_dist), post_max_dist), ref_max_s / 2)
    num_post_ref = int((post_dist // sampling_dist)) + 1

    print(f"\n>>> POST-OBSTACLE TRAJECTORY:")
    print(f"    post_dist = min(min(max({pre_dist:.2f}, {post_min_dist:.2f}), {post_max_dist:.2f}), {ref_max_s/2:.2f}) = {post_dist:.2f}m")
    print(f"    num_post_ref = int({post_dist:.2f} // {sampling_dist:.2f}) + 1 = {num_post_ref}")

    print(f"\n>>> GENERATING CONTROL POINTS:")
    print(f"    Point 0 (apex): s={s_list[0]:.4f}m, d={d_list[0]:.4f}m")

    for i in range(num_post_ref):
        s_new = obs.s_center + post_dist * ((i + 1)/ num_post_ref)
        d_new = d_apex * (1 - (i + 1)/ num_post_ref)
        s_list.append(s_new)
        d_list.append(d_new)
        print(f"    Point {i+1}: s={s_new:.4f}m, d={d_new:.4f}m (decay factor={(1 - (i + 1)/ num_post_ref):.4f})")

    s_array = np.array(s_list)
    d_array = np.array(d_list)
    s_array = s_array % ref_max_s
    s_idx = np.round((s_array / wpnt_dist)).astype(int) % ref_max_idx

    print(f"\n>>> CONTROL POINTS (NORMALIZED):")
    for i in range(len(s_list)):
        print(f"    Point {i}: s={s_array[i]:.4f}m, d={d_array[i]:.4f}m, idx={s_idx[i]}")

    # Frenet to Cartesian
    resp = fixed_converter.get_cartesian(s_array, d_array)

    print(f"\n>>> FRENET TO CARTESIAN CONVERSION:")
    for i in range(len(s_array)):
        print(f"    Point {i}: (s={s_array[i]:.4f}, d={d_array[i]:.4f}) -> XY=({resp[0,i]:.4f}, {resp[1,i]:.4f})")

    # ===== DEBUG: Test Fixed Frenet coordinate system =====
    print(f"\n>>> FIXED FRENET COORDINATE SYSTEM TEST:")

    # Test 1: Show actual waypoint coordinates around obstacle
    print(f"\n1. Waypoint coordinates around obstacle (idx={obs_s_idx}):")
    for offset in [-2, -1, 0, 1, 2]:
        idx = (obs_s_idx + offset) % len(gb_wpnts)
        wpnt = gb_wpnts[idx]
        print(f"   wpnt[{idx}]: s_GB={wpnt.s_m:.4f}, xy=({wpnt.x_m:.4f}, {wpnt.y_m:.4f}), psi={wpnt.psi_rad:.4f}rad ({np.degrees(wpnt.psi_rad):.2f}deg)")

    # Test 2: Convert waypoint XY back to Fixed Frenet to verify
    test_idx = obs_s_idx
    test_wpnt = gb_wpnts[test_idx]
    test_s, test_d = fixed_converter.get_frenet(np.array([test_wpnt.x_m]), np.array([test_wpnt.y_m]))
    print(f"\n2. Verify waypoint[{test_idx}] round-trip conversion:")
    print(f"   XY=({test_wpnt.x_m:.4f}, {test_wpnt.y_m:.4f}) -> Fixed Frenet (s={test_s[0]:.4f}, d={test_d[0]:.4f})")

    # Test 3: Test d offset directions at obstacle s position
    print(f"\n3. Test d offsets at s={obs.s_center:.4f}:")
    test_s_val = obs.s_center
    test_d_values = [0.0, 0.5, -0.5]
    for d_val in test_d_values:
        xy = fixed_converter.get_cartesian(np.array([test_s_val]), np.array([d_val]))
        print(f"   (s={test_s_val:.4f}, d={d_val:+.4f}) -> XY=({xy[0,0]:.4f}, {xy[1,0]:.4f})")

    # Calculate which direction is "left" vs "right"
    xy_center = fixed_converter.get_cartesian(np.array([test_s_val]), np.array([0.0]))
    xy_plus = fixed_converter.get_cartesian(np.array([test_s_val]), np.array([0.5]))
    xy_minus = fixed_converter.get_cartesian(np.array([test_s_val]), np.array([-0.5]))

    dx_plus = xy_plus[0,0] - xy_center[0,0]
    dy_plus = xy_plus[1,0] - xy_center[1,0]
    dx_minus = xy_minus[0,0] - xy_center[0,0]
    dy_minus = xy_minus[1,0] - xy_center[1,0]

    print(f"\n4. Normal vector directions:")
    print(f"   d=0 -> d=+0.5: dx={dx_plus:+.4f}, dy={dy_plus:+.4f}, angle={np.degrees(np.arctan2(dy_plus, dx_plus)):.2f}deg")
    print(f"   d=0 -> d=-0.5: dx={dx_minus:+.4f}, dy={dy_minus:+.4f}, angle={np.degrees(np.arctan2(dy_minus, dx_minus)):.2f}deg")

    # Calculate path tangent direction at this s
    s_before = test_s_val - 0.1
    s_after = test_s_val + 0.1
    xy_before = fixed_converter.get_cartesian(np.array([s_before]), np.array([0.0]))
    xy_after = fixed_converter.get_cartesian(np.array([s_after]), np.array([0.0]))
    path_dx = xy_after[0,0] - xy_before[0,0]
    path_dy = xy_after[1,0] - xy_before[1,0]
    path_angle = np.degrees(np.arctan2(path_dy, path_dx))

    print(f"\n5. Path tangent direction at s={test_s_val:.4f}:")
    print(f"   s={s_before:.4f} -> s={s_after:.4f}: dx={path_dx:+.4f}, dy={path_dy:+.4f}, angle={path_angle:.2f}deg")
    print(f"   This should match wpnt[{test_idx}].psi_rad={np.degrees(test_wpnt.psi_rad):.2f}deg")
    print(f"   DIFF: {path_angle - np.degrees(test_wpnt.psi_rad):.2f}deg")

    # ===== DETAILED PSI VS ACTUAL TANGENT COMPARISON =====
    print(f"\n6. PSI_RAD vs ACTUAL PATH TANGENT (waypoints around obstacle):")
    print(f"   Fixed path psi_rad WITHOUT π/2 conversion (WRONG)")
    print(f"   {'Idx':<5} {'s_GB':<10} {'XY':<25} {'psi_rad':<12} {'actual_tangent':<15} {'DIFF':<10}")

    for offset in [-2, -1, 0, 1, 2]:
        idx = (obs_s_idx + offset) % len(gb_wpnts)
        wpnt = gb_wpnts[idx]

        # Get Fixed Frenet s for this waypoint
        s_fixed, d_fixed = fixed_converter.get_frenet(np.array([wpnt.x_m]), np.array([wpnt.y_m]))

        # Calculate actual tangent from neighboring waypoints
        if idx > 0 and idx < len(gb_wpnts) - 1:
            prev_wpnt = gb_wpnts[idx - 1]
            next_wpnt = gb_wpnts[idx + 1]
            dx_actual = next_wpnt.x_m - prev_wpnt.x_m
            dy_actual = next_wpnt.y_m - prev_wpnt.y_m
            actual_angle = np.degrees(np.arctan2(dy_actual, dx_actual))
        else:
            actual_angle = float('nan')

        psi_deg = np.degrees(wpnt.psi_rad)
        diff = actual_angle - psi_deg if not np.isnan(actual_angle) else float('nan')

        print(f"   {idx:<5} {wpnt.s_m:<10.4f} ({wpnt.x_m:.2f}, {wpnt.y_m:.2f}){'':<10} {psi_deg:<12.2f} {actual_angle:<15.2f} {diff:<10.2f}")

    # ===== TEST π/2 CONVERSION (GB optimizer style) =====
    print(f"\n7. PSI_RAD WITH π/2 CONVERSION (GB optimizer method - should be CORRECT):")
    print(f"   Fixed path psi_rad + π/2 (tph convention -> ROS convention)")
    print(f"   {'Idx':<5} {'s_GB':<10} {'XY':<25} {'psi_orig':<12} {'psi_conv':<12} {'actual':<12} {'DIFF':<10}")

    for offset in [-2, -1, 0, 1, 2]:
        idx = (obs_s_idx + offset) % len(gb_wpnts)
        wpnt = gb_wpnts[idx]

        # Get Fixed Frenet s for this waypoint
        s_fixed, d_fixed = fixed_converter.get_frenet(np.array([wpnt.x_m]), np.array([wpnt.y_m]))

        # Calculate actual tangent from neighboring waypoints
        if idx > 0 and idx < len(gb_wpnts) - 1:
            prev_wpnt = gb_wpnts[idx - 1]
            next_wpnt = gb_wpnts[idx + 1]
            dx_actual = next_wpnt.x_m - prev_wpnt.x_m
            dy_actual = next_wpnt.y_m - prev_wpnt.y_m
            actual_angle = np.degrees(np.arctan2(dy_actual, dx_actual))
        else:
            actual_angle = float('nan')

        # Apply π/2 conversion (same as global_planner_node.py)
        psi_tph = wpnt.psi_rad  # tph output (0 = north/y-axis)
        psi_ros = psi_tph + np.pi / 2  # Convert to ROS (0 = east/x-axis)
        if psi_ros > np.pi:
            psi_ros = psi_ros - 2 * np.pi

        psi_orig_deg = np.degrees(wpnt.psi_rad)
        psi_conv_deg = np.degrees(psi_ros)
        diff = actual_angle - psi_conv_deg if not np.isnan(actual_angle) else float('nan')

        print(f"   {idx:<5} {wpnt.s_m:<10.4f} ({wpnt.x_m:.2f}, {wpnt.y_m:.2f}){'':<10} {psi_orig_deg:<12.2f} {psi_conv_deg:<12.2f} {actual_angle:<12.2f} {diff:<10.2f}")

    # ===== COMPARE WITH GB GLOBAL WAYPOINTS =====
    print(f"\n\n{'='*80}")
    print(f"COMPARISON: GB GLOBAL WAYPOINTS vs FIXED PATH WAYPOINTS")
    print(f"{'='*80}")

    # Use subscribed GB global waypoints
    if gb_global_waypoints is not None and len(gb_global_waypoints) > 0:
        print(f"\n9. GB GLOBAL WAYPOINTS (original raceline) - PSI_RAD vs ACTUAL TANGENT:")
        print(f"   (These should have DIFF ≈ 0 because GB optimizer already applies π/2 conversion)")
        print(f"   {'Idx':<5} {'s_GB':<10} {'XY':<25} {'psi_rad':<12} {'actual_tangent':<15} {'DIFF':<10}")

        # Find waypoints around current position using XY distance
        cur_pos = np.array([cur_x, cur_y])
        gb_dists = [np.linalg.norm(np.array([wpnt.x_m, wpnt.y_m]) - cur_pos) for wpnt in gb_global_waypoints]
        gb_closest_idx = np.argmin(gb_dists)

        for offset in [-2, -1, 0, 1, 2]:
            idx = (gb_closest_idx + offset) % len(gb_global_waypoints)
            wpnt = gb_global_waypoints[idx]

            # Calculate actual tangent from neighboring waypoints
            if idx > 0 and idx < len(gb_global_waypoints) - 1:
                prev_wpnt = gb_global_waypoints[idx - 1]
                next_wpnt = gb_global_waypoints[idx + 1]
                dx_actual = next_wpnt.x_m - prev_wpnt.x_m
                dy_actual = next_wpnt.y_m - prev_wpnt.y_m
                actual_angle = np.degrees(np.arctan2(dy_actual, dx_actual))
            else:
                actual_angle = float('nan')

            psi_deg = np.degrees(wpnt.psi_rad)
            diff = actual_angle - psi_deg if not np.isnan(actual_angle) else float('nan')

            print(f"   {idx:<5} {wpnt.s_m:<10.4f} ({wpnt.x_m:.2f}, {wpnt.y_m:.2f}){'':<10} {psi_deg:<12.2f} {actual_angle:<15.2f} {diff:<10.2f}")

        print(f"\n   NOTE: GB global waypoints are the ORIGINAL raceline (not Fixed path)")
        print(f"         DIFF ≈ 0 confirms that GB optimizer's π/2 conversion is correct!")
    else:
        print(f"\n9. GB GLOBAL WAYPOINTS: Not subscribed yet")

    # ===== ALSO CHECK CURRENT POSITION =====
    print(f"\n10. Current position tangent check (Fixed path):")
    print(f"   Current: xy=({cur_x:.4f}, {cur_y:.4f}), cur_yaw={np.degrees(cur_yaw):.2f}deg")

    # Find closest waypoint to current position
    dists = [np.sqrt((wpnt.x_m - cur_x)**2 + (wpnt.y_m - cur_y)**2) for wpnt in gb_wpnts]
    closest_idx = np.argmin(dists)
    closest_wpnt = gb_wpnts[closest_idx]

    print(f"   Closest waypoint[{closest_idx}]: xy=({closest_wpnt.x_m:.4f}, {closest_wpnt.y_m:.4f}), psi={np.degrees(closest_wpnt.psi_rad):.2f}deg")
    print(f"   Difference: {np.degrees(cur_yaw) - np.degrees(closest_wpnt.psi_rad):.2f}deg")

    # Calculate actual tangent at current position
    if closest_idx > 0 and closest_idx < len(gb_wpnts) - 1:
        prev_wpnt = gb_wpnts[closest_idx - 1]
        next_wpnt = gb_wpnts[closest_idx + 1]
        dx_actual = next_wpnt.x_m - prev_wpnt.x_m
        dy_actual = next_wpnt.y_m - prev_wpnt.y_m
        actual_angle = np.degrees(np.arctan2(dy_actual, dx_actual))
        print(f"   Actual path tangent at closest waypoint: {actual_angle:.2f}deg")
        print(f"   cur_yaw vs actual tangent: {np.degrees(cur_yaw) - actual_angle:.2f}deg")
    # ===== END DEBUG =====

    # Build spline
    points = [[cur_x, cur_y]]

    # Collect all control points first
    for i in range(len(s_idx)):
        points.append(resp[:, i])

    points = np.asarray(points)

    print(f"\n>>> BUILDING SPLINE (FIXED: tangents from point-to-point directions):")
    print(f"    Start point 0: xy=({cur_x:.4f}, {cur_y:.4f}), cur_yaw={cur_yaw:.4f}rad ({np.degrees(cur_yaw):.2f}deg)")

    # Calculate tangents based on actual point-to-point directions
    tangents = []
    for i in range(len(points)):
        if i == 0:
            # First point: use current yaw
            tangent = np.array([np.cos(cur_yaw), np.sin(cur_yaw)])
            print(f"    Point 0 tangent: ({tangent[0]:.4f}, {tangent[1]:.4f}) from cur_yaw")
        elif i < len(points) - 1:
            # Middle points: use direction to next point
            dx = points[i+1, 0] - points[i, 0]
            dy = points[i+1, 1] - points[i, 1]
            tangent_angle = np.arctan2(dy, dx)
            tangent = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])

            # Compare with waypoint heading
            wpnt_heading = gb_wpnts[s_idx[i-1]].psi_rad
            diff_deg = np.degrees(tangent_angle - wpnt_heading)
            print(f"    Point {i} tangent: ({tangent[0]:.4f}, {tangent[1]:.4f}), angle={tangent_angle:.4f}rad ({np.degrees(tangent_angle):.2f}deg)")
            print(f"             vs wpnt_heading={wpnt_heading:.4f}rad ({np.degrees(wpnt_heading):.2f}deg), DIFF={diff_deg:.2f}deg")
        else:
            # Last point: use direction from previous point
            dx = points[i, 0] - points[i-1, 0]
            dy = points[i, 1] - points[i-1, 1]
            tangent_angle = np.arctan2(dy, dx)
            tangent = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])

            wpnt_heading = gb_wpnts[s_idx[i-1]].psi_rad
            diff_deg = np.degrees(tangent_angle - wpnt_heading)
            print(f"    Point {i} tangent: ({tangent[0]:.4f}, {tangent[1]:.4f}), angle={tangent_angle:.4f}rad ({np.degrees(tangent_angle):.2f}deg)")
            print(f"             vs wpnt_heading={wpnt_heading:.4f}rad ({np.degrees(wpnt_heading):.2f}deg), DIFF={diff_deg:.2f}deg")

        tangents.append(tangent)

    tangents = np.dot(tangents, spline_scale*np.eye(2))
    nPoints, dim = points.shape

    print(f"\n>>> SPLINE PARAMETERS:")
    print(f"    Total control points: {nPoints}")
    print(f"    Spline scale: {spline_scale}")

    # Parametrization
    dp = np.diff(points, axis=0)
    dp = np.linalg.norm(dp, axis=1)
    d = np.cumsum(dp)
    d = np.hstack([[0],d])
    l = d[-1]
    nSamples = int(l/wpnt_dist)
    s,r = np.linspace(0,l,nSamples,retstep=True)

    print(f"\n>>> PARAMETRIZATION:")
    print(f"    Spline length: {l:.4f}m")
    print(f"    wpnt_dist: {wpnt_dist:.4f}m")
    print(f"    Number of samples: {nSamples}")

    # Build spline
    assert(len(points) == len(tangents))
    spline_result = np.empty([nPoints, dim], dtype=object)
    for i,ref in enumerate(points):
        t = tangents[i]
        assert(t is None or len(t)==dim)
        fuse = list(zip(ref,t) if t is not None else zip(ref,))
        spline_result[i,:] = fuse

    # Compute splines
    samples = np.zeros([nSamples, dim])
    for i in range(dim):
        poly = BPoly.from_derivatives(d, spline_result[:,i])
        samples[:,i] = poly(s)

    print(f"Spline samples generated: {samples.shape[0]}")

    # Add additional waypoints
    n_additional = 100
    xy_additional = np.array([
        (
            gb_wpnts[(s_idx[-1] + i + 1) % ref_max_idx].x_m,
            gb_wpnts[(s_idx[-1] + i + 1) % ref_max_idx].y_m
        )
        for i in range(n_additional)
    ])
    samples = np.vstack([samples, xy_additional])

    print(f"Total samples (with additional): {samples.shape[0]}")

    # Convert back to Frenet
    s_, d_ = fixed_converter.get_frenet(samples[:, 0], samples[:, 1])

    print(f"\n>>> FINAL SAMPLES (first 5 and last 5):")
    num_show = min(5, len(samples))
    for i in range(num_show):
        print(f"    Sample {i}: xy=({samples[i,0]:.4f}, {samples[i,1]:.4f}), frenet=(s={s_[i]:.4f}, d={d_[i]:.4f})")
    if len(samples) > 10:
        print(f"    ... ({len(samples)-10} samples omitted) ...")
        for i in range(len(samples)-num_show, len(samples)):
            print(f"    Sample {i}: xy=({samples[i,0]:.4f}, {samples[i,1]:.4f}), frenet=(s={s_[i]:.4f}, d={d_[i]:.4f})")

    # Calculate heading and curvature
    psi_, kappa_ = tph.calc_head_curv_num.calc_head_curv_num(
        path=samples,
        el_lengths=0.1*np.ones(len(samples)-1),
        is_closed=False
    )

    print(f"\n>>> HEADING AND CURVATURE (first 5):")
    for i in range(min(5, len(psi_))):
        print(f"    Sample {i}: psi={psi_[i]:.4f}rad ({np.degrees(psi_[i]):.2f}deg), kappa={kappa_[i]:.6f}")

    # Check bounds (if map_filter provided)
    danger_flag = False
    bounds_violations = []
    if map_filter is not None:
        for i in range(samples.shape[0]):
            inside = map_filter.is_point_inside(samples[i, 0], samples[i, 1])
            if not inside:
                print(f"BOUNDS VIOLATION at sample {i}/{samples.shape[0]}: xy=({samples[i, 0]:.2f}, {samples[i, 1]:.2f})")
                danger_flag = True
                bounds_violations.append(i)
                break

    if danger_flag:
        print(f"ABORT: Bounds violation detected!")
        return None, None

    print(f"SUCCESS: Generated {len(samples)} waypoints")
    print("="*80)

    return samples, (s_, d_, psi_, kappa_)

def publish_markers():
    rospy.init_node('spline_input_visualizer', anonymous=True)

    # Visualization settings
    BOUNDARY_INTERVAL = 1  # Show boundary connection every N waypoints (1 = all, 2 = every other, etc.)

    # Subscriber for GB global waypoints
    rospy.Subscriber('/global_waypoints_scaled', WpntArray, gb_waypoints_callback, queue_size=1)

    # Publishers
    wpnt_pub = rospy.Publisher('/debug/spline_waypoints', MarkerArray, queue_size=1, latch=True)
    obs_pub = rospy.Publisher('/debug/spline_obstacle', MarkerArray, queue_size=1, latch=True)

    # Wait for GB waypoints
    rospy.loginfo("Waiting for /global_waypoints_scaled...")
    while gb_global_waypoints is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo(f"Received GB global waypoints: {len(gb_global_waypoints)} points")

    # Load data
    try:
        with open('/tmp/do_spline_input_fixed.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        rospy.logerr("File /tmp/do_spline_input_fixed.pkl not found!")
        rospy.logerr("Run the ROS node first to generate the data.")
        sys.exit(1)

    obs = data['obs']
    gb_wpnts = data['gb_wpnts']
    cur_x = data['cur_x']
    cur_y = data['cur_y']
    boundary_left_xy = data.get('boundary_left_xy', None)
    boundary_right_xy = data.get('boundary_right_xy', None)

    # Create Fixed FrenetConverter from waypoint XY
    import numpy as np
    sys.path.append('/home/hj/unicorn_ws/UNICORN/f110_utils/libs/frenet_conversion/src')
    from frenet_converter.frenet_converter import FrenetConverter

    wpnt_x = np.array([w.x_m for w in gb_wpnts])
    wpnt_y = np.array([w.y_m for w in gb_wpnts])
    fixed_converter = FrenetConverter(wpnt_x, wpnt_y)

    # Recalculate s,d in Fixed Frenet coordinate system
    s_fixed, d_fixed = fixed_converter.get_frenet(wpnt_x, wpnt_y)

    # Store Fixed Frenet coordinates in dictionary (Wpnt is read-only)
    wpnt_fixed_coords = {}
    for i in range(len(gb_wpnts)):
        wpnt_fixed_coords[i] = {'s': s_fixed[i], 'd': d_fixed[i]}

    # Recalculate current position in Fixed Frenet
    cur_s_fixed, cur_d_fixed = fixed_converter.get_frenet(np.array([cur_x]), np.array([cur_y]))

    rospy.loginfo("="*80)
    rospy.loginfo("SPLINE INPUT VISUALIZATION (FIXED FRENET)")
    rospy.loginfo("="*80)
    rospy.loginfo(f"Mode: {'FIXED' if data['use_fixed_path'] else 'GB'}")
    rospy.loginfo(f"Current: x={cur_x:.2f}, y={cur_y:.2f}")
    rospy.loginfo(f"         FIXED s={cur_s_fixed[0]:.2f}, d={cur_d_fixed[0]:.2f} (GB s={data['cur_s']:.2f})")
    rospy.loginfo(f"Obstacle: FIXED s={obs.s_center:.2f}, d={obs.d_center:.2f}")
    rospy.loginfo(f"Waypoints: {len(gb_wpnts)} points, raceline_length={fixed_converter.raceline_length:.2f}m")
    rospy.loginfo(f"  First: FIXED s={s_fixed[0]:.2f}, GB s={gb_wpnts[0].s_m:.2f}")
    rospy.loginfo(f"  Last:  FIXED s={s_fixed[-1]:.2f}, GB s={gb_wpnts[-1].s_m:.2f}")
    rospy.loginfo("")

    # ===== RUN DO_SPLINE DEBUG =====
    # Load cur_yaw from pkl (critical for correct spline tangent!)
    cur_yaw = data.get('cur_yaw', 0.0)
    rospy.loginfo(f"Loaded cur_yaw: {cur_yaw:.4f} rad ({np.degrees(cur_yaw):.2f} deg)")
    rospy.loginfo("Running do_spline_debug...")
    samples, frenet_data = do_spline_debug(
        obs=obs,
        gb_wpnts=gb_wpnts,
        fixed_converter=fixed_converter,
        cur_x=cur_x,
        cur_y=cur_y,
        cur_yaw=cur_yaw,
        spline_scale=1.5,
        post_min_dist=2.0,
        post_max_dist=10.0,
        sampling_dist=0.1,
        evasion_dist=0.5,
        spline_bound_mindist=0.5,
        map_filter=None  # TODO: load map_filter if needed
    )

    # Create waypoint markers
    marker_array = MarkerArray()

    # Visualize generated spline if successful
    if samples is not None:
        rospy.loginfo(f"do_spline_debug SUCCESS: Generated {len(samples)} samples")

        # Add spline path as RED line
        spline_marker = Marker()
        spline_marker.header.frame_id = "map"
        spline_marker.header.stamp = rospy.Time.now()
        spline_marker.ns = "spline_path"
        spline_marker.id = 0
        spline_marker.type = Marker.LINE_STRIP
        spline_marker.action = Marker.ADD
        spline_marker.scale.x = 0.08  # Line width
        spline_marker.color.r = 1.0
        spline_marker.color.g = 0.0
        spline_marker.color.b = 0.0
        spline_marker.color.a = 1.0

        for i in range(len(samples)):
            p = Point()
            p.x = samples[i, 0]
            p.y = samples[i, 1]
            p.z = 0.2  # Slightly elevated
            spline_marker.points.append(p)

        marker_array.markers.append(spline_marker)

        # Add sample points as small RED spheres (every 10th)
        for i in range(0, len(samples), 10):
            sample_marker = Marker()
            sample_marker.header.frame_id = "map"
            sample_marker.header.stamp = rospy.Time.now()
            sample_marker.ns = "spline_samples"
            sample_marker.id = i
            sample_marker.type = Marker.SPHERE
            sample_marker.action = Marker.ADD
            sample_marker.pose.position.x = samples[i, 0]
            sample_marker.pose.position.y = samples[i, 1]
            sample_marker.pose.position.z = 0.2
            sample_marker.pose.orientation.w = 1.0
            sample_marker.scale.x = 0.12
            sample_marker.scale.y = 0.12
            sample_marker.scale.z = 0.12
            sample_marker.color.r = 1.0
            sample_marker.color.g = 0.0
            sample_marker.color.b = 0.0
            sample_marker.color.a = 0.8
            marker_array.markers.append(sample_marker)

        rospy.loginfo(f"Added spline visualization: {len(spline_marker.points)} points")
    else:
        rospy.logwarn("do_spline_debug FAILED - no spline to visualize")

    # Waypoint path (line strip) - BLUE
    path_marker = Marker()
    path_marker.header.frame_id = "map"
    path_marker.header.stamp = rospy.Time.now()
    path_marker.ns = "waypoint_path"
    path_marker.id = 0
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    path_marker.scale.x = 0.08  # Line width
    path_marker.color.r = 0.0
    path_marker.color.g = 0.5
    path_marker.color.b = 1.0
    path_marker.color.a = 1.0

    for wpnt in gb_wpnts:
        p = Point()
        p.x = wpnt.x_m
        p.y = wpnt.y_m
        p.z = 0.1
        path_marker.points.append(p)

    marker_array.markers.append(path_marker)

    # Waypoint indices (every 20th)
    for i in range(0, len(gb_wpnts), 20):
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "waypoint_ids"
        text_marker.id = i
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = gb_wpnts[i].x_m
        text_marker.pose.position.y = gb_wpnts[i].y_m
        text_marker.pose.position.z = 0.4
        text_marker.scale.z = 0.15
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"#{i}\nFIX_s={wpnt_fixed_coords[i]['s']:.1f}\nGB_s={gb_wpnts[i].s_m:.1f}\nL/R:{gb_wpnts[i].d_left:.1f}/{gb_wpnts[i].d_right:.1f}"
        marker_array.markers.append(text_marker)

    # Current position marker - GREEN ARROW
    cur_marker = Marker()
    cur_marker.header.frame_id = "map"
    cur_marker.header.stamp = rospy.Time.now()
    cur_marker.ns = "current_position"
    cur_marker.id = 0
    cur_marker.type = Marker.ARROW
    cur_marker.action = Marker.ADD
    cur_marker.pose.position.x = cur_x
    cur_marker.pose.position.y = cur_y
    cur_marker.pose.position.z = 0.5
    cur_marker.pose.orientation.w = 1.0
    cur_marker.scale.x = 0.8
    cur_marker.scale.y = 0.15
    cur_marker.scale.z = 0.15
    cur_marker.color.r = 0.0
    cur_marker.color.g = 1.0
    cur_marker.color.b = 0.0
    cur_marker.color.a = 1.0
    marker_array.markers.append(cur_marker)

    # Obstacle markers
    obs_array = MarkerArray()

    # Obstacle cylinder - RED
    obs_marker = Marker()
    obs_marker.header.frame_id = "map"
    obs_marker.header.stamp = rospy.Time.now()
    obs_marker.ns = "obstacle"
    obs_marker.id = 0
    obs_marker.type = Marker.CYLINDER
    obs_marker.action = Marker.ADD
    obs_marker.pose.position.x = obs.x_m
    obs_marker.pose.position.y = obs.y_m
    obs_marker.pose.position.z = 0.25
    obs_marker.pose.orientation.w = 1.0
    obs_marker.scale.x = obs.size
    obs_marker.scale.y = obs.size
    obs_marker.scale.z = 0.5
    obs_marker.color.r = 1.0
    obs_marker.color.g = 0.0
    obs_marker.color.b = 0.0
    obs_marker.color.a = 0.8
    obs_array.markers.append(obs_marker)

    # Obstacle info text
    obs_text = Marker()
    obs_text.header.frame_id = "map"
    obs_text.header.stamp = rospy.Time.now()
    obs_text.ns = "obstacle_info"
    obs_text.id = 1
    obs_text.type = Marker.TEXT_VIEW_FACING
    obs_text.action = Marker.ADD
    obs_text.pose.position.x = obs.x_m
    obs_text.pose.position.y = obs.y_m
    obs_text.pose.position.z = 0.9
    obs_text.scale.z = 0.2
    obs_text.color.r = 1.0
    obs_text.color.g = 1.0
    obs_text.color.b = 0.0
    obs_text.color.a = 1.0
    obs_text.text = f"OBS #{obs.id}\ns={obs.s_center:.2f}\nd={obs.d_center:.2f}\ndL={obs.d_left:.2f}\ndR={obs.d_right:.2f}"
    obs_array.markers.append(obs_text)

    # Boundary visualization - LEFT (CYAN) and RIGHT (MAGENTA)
    # Use actual closest boundary points if available
    if boundary_left_xy is not None and boundary_right_xy is not None:
        rospy.loginfo(f"Using ACTUAL closest boundary points (interval={BOUNDARY_INTERVAL})")
        for i in range(len(gb_wpnts)):
            if i % BOUNDARY_INTERVAL != 0:  # Use configurable interval
                continue

            wpnt = gb_wpnts[i]
            left_xy = boundary_left_xy[i]
            right_xy = boundary_right_xy[i]

            # Left boundary marker - CYAN sphere
            left_marker = Marker()
            left_marker.header.frame_id = "map"
            left_marker.header.stamp = rospy.Time.now()
            left_marker.ns = "boundary_left"
            left_marker.id = i
            left_marker.type = Marker.SPHERE
            left_marker.action = Marker.ADD
            left_marker.pose.position.x = left_xy[0]
            left_marker.pose.position.y = left_xy[1]
            left_marker.pose.position.z = 0.1
            left_marker.pose.orientation.w = 1.0
            left_marker.scale.x = 0.15
            left_marker.scale.y = 0.15
            left_marker.scale.z = 0.15
            left_marker.color.r = 0.0
            left_marker.color.g = 1.0
            left_marker.color.b = 1.0  # Cyan
            left_marker.color.a = 0.8
            marker_array.markers.append(left_marker)

            # Right boundary marker - MAGENTA sphere
            right_marker = Marker()
            right_marker.header.frame_id = "map"
            right_marker.header.stamp = rospy.Time.now()
            right_marker.ns = "boundary_right"
            right_marker.id = i
            right_marker.type = Marker.SPHERE
            right_marker.action = Marker.ADD
            right_marker.pose.position.x = right_xy[0]
            right_marker.pose.position.y = right_xy[1]
            right_marker.pose.position.z = 0.1
            right_marker.pose.orientation.w = 1.0
            right_marker.scale.x = 0.15
            right_marker.scale.y = 0.15
            right_marker.scale.z = 0.15
            right_marker.color.r = 1.0
            right_marker.color.g = 0.0
            right_marker.color.b = 1.0  # Magenta
            right_marker.color.a = 0.8
            marker_array.markers.append(right_marker)

            # Add line from waypoint to boundary points for visualization
            line_left = Marker()
            line_left.header.frame_id = "map"
            line_left.header.stamp = rospy.Time.now()
            line_left.ns = "boundary_line_left"
            line_left.id = i
            line_left.type = Marker.LINE_STRIP
            line_left.action = Marker.ADD
            line_left.scale.x = 0.02  # Line width
            line_left.color.r = 0.0
            line_left.color.g = 1.0
            line_left.color.b = 1.0
            line_left.color.a = 0.5
            p1 = Point()
            p1.x = wpnt.x_m
            p1.y = wpnt.y_m
            p1.z = 0.05
            p2 = Point()
            p2.x = left_xy[0]
            p2.y = left_xy[1]
            p2.z = 0.05
            line_left.points = [p1, p2]
            marker_array.markers.append(line_left)

            line_right = Marker()
            line_right.header.frame_id = "map"
            line_right.header.stamp = rospy.Time.now()
            line_right.ns = "boundary_line_right"
            line_right.id = i
            line_right.type = Marker.LINE_STRIP
            line_right.action = Marker.ADD
            line_right.scale.x = 0.02  # Line width
            line_right.color.r = 1.0
            line_right.color.g = 0.0
            line_right.color.b = 1.0
            line_right.color.a = 0.5
            p1 = Point()
            p1.x = wpnt.x_m
            p1.y = wpnt.y_m
            p1.z = 0.05
            p2 = Point()
            p2.x = right_xy[0]
            p2.y = right_xy[1]
            p2.z = 0.05
            line_right.points = [p1, p2]
            marker_array.markers.append(line_right)
    else:
        rospy.logwarn("Boundary XY data not available in pkl file - skipping boundary visualization")

    # Publish
    rate = rospy.Rate(1)
    rospy.loginfo("Publishing markers to:")
    rospy.loginfo("  /debug/spline_waypoints (blue path + text + CYAN/MAGENTA boundaries)")
    rospy.loginfo("  /debug/spline_obstacle (red cylinder + text)")
    rospy.loginfo("")
    rospy.loginfo("Legend:")
    rospy.loginfo("  BLUE line = Fixed path waypoints")
    rospy.loginfo("  RED line = Generated spline path (from do_spline_debug)")
    rospy.loginfo("  RED spheres = Spline sample points (every 10th)")
    rospy.loginfo("  CYAN spheres + lines = LEFT boundary (actual closest point)")
    rospy.loginfo("  MAGENTA spheres + lines = RIGHT boundary (actual closest point)")
    rospy.loginfo("  GREEN arrow = Current position")
    rospy.loginfo("  RED cylinder = Obstacle")
    rospy.loginfo("")

    while not rospy.is_shutdown():
        wpnt_pub.publish(marker_array)
        obs_pub.publish(obs_array)
        rospy.loginfo_throttle(10.0, "Markers published. Check RViz!")
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_markers()
    except rospy.ROSInterruptException:
        pass
