#!/usr/bin/env python3
"""
문제 2 분석: Frenet 변환 오류 (16-52-19, 16-55-36 bags)
- 가장 가까운 웨이포인트가 멀리 점프
- lookahead가 비정상적으로 김
- z 필터링, 헤딩, 높이 문제 분석
"""
import rosbag
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
from scipy.interpolate import interp1d
import os
import json

BAG_PATHS = [
    '/home/unicorn/catkin_ws/src/race_stack/bag/2026-04-04-16-52-19.bag',
    '/home/unicorn/catkin_ws/src/race_stack/bag/2026-04-04-16-55-36.bag',
]
BAG_NAMES = ['bag1_1652', 'bag2_1655']
OUT_DIR = '/home/unicorn/catkin_ws/src/race_stack/HJ_docs/debug/figure/0404/frenet'
os.makedirs(OUT_DIR, exist_ok=True)

# Load global waypoints
MAP_DIR = '/home/unicorn/catkin_ws/src/race_stack/stack_master/maps/eng_0404_v2'
map_json = os.path.join(MAP_DIR, 'global_waypoints.json')
global_wpnts = None
if os.path.exists(map_json):
    with open(map_json) as f:
        data = json.load(f)
    # JSON is a dict with nested waypoint arrays
    wpnt_key = 'global_traj_wpnts_sp'
    if wpnt_key in data and 'wpnts' in data[wpnt_key]:
        wpnts_list = data[wpnt_key]['wpnts']
    elif 'centerline_waypoints' in data and 'wpnts' in data['centerline_waypoints']:
        wpnts_list = data['centerline_waypoints']['wpnts']
    else:
        wpnts_list = []
    if wpnts_list:
        global_wpnts = {
            'x': np.array([w['x_m'] for w in wpnts_list]),
            'y': np.array([w['y_m'] for w in wpnts_list]),
            'z': np.array([w.get('z_m', 0) for w in wpnts_list]),
            's': np.array([w['s_m'] for w in wpnts_list]),
            'vx': np.array([w['vx_mps'] for w in wpnts_list]),
        }
        print(f"Global waypoints loaded: {len(global_wpnts['x'])} points")


topics = [
    '/car_state/odom',
    '/car_state/odom_frenet',
    '/car_state/pose',
    '/glim_ros/base_odom',
    '/vesc/high_level/ackermann_cmd_mux/input/nav_1',
    '/l1_distance',
    '/lookahead_point',
    '/local_waypoints',
    '/centerline_waypoints',
]


def extract_bag(bag_path):
    """Extract relevant data from a bag file."""
    print(f"\n=== Extracting {bag_path} ===")
    bag = rosbag.Bag(bag_path, 'r')
    start_time = bag.get_start_time()

    car_odom = {'t': [], 'x': [], 'y': [], 'z': [], 'vx': [], 'vy': [], 'speed': [], 'yaw': []}
    car_frenet = {'t': [], 's': [], 'd': [], 'vs': [], 'vd': [], 'closest_idx': []}
    car_pose = {'t': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []}
    glim_odom = {'t': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []}
    ctrl_input = {'t': [], 'speed': [], 'steer': []}
    l1_dist = {'t': [], 'x': [], 'y': [], 'z': []}
    lookahead_pt = {'t': [], 'x': [], 'y': [], 'z': []}

    # For closest waypoint index from odom_frenet child_frame_id
    centerline_wpnts = None
    local_wpnts_data = {'t': [], 'first_s': [], 'last_s': [], 'n_wpnts': [], 'first_x': [], 'first_y': [], 'first_z': []}

    for topic, msg, t in bag.read_messages(topics=topics):
        ts = t.to_sec() - start_time

        if topic == '/car_state/odom':
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            v = msg.twist.twist.linear
            car_odom['t'].append(ts)
            car_odom['x'].append(p.x)
            car_odom['y'].append(p.y)
            car_odom['z'].append(p.z)
            car_odom['vx'].append(v.x)
            car_odom['vy'].append(v.y)
            car_odom['speed'].append(np.sqrt(v.x**2 + v.y**2))
            car_odom['yaw'].append(yaw)

        elif topic == '/car_state/odom_frenet':
            p = msg.pose.pose.position
            v = msg.twist.twist.linear
            car_frenet['t'].append(ts)
            car_frenet['s'].append(p.x)
            car_frenet['d'].append(p.y)
            car_frenet['vs'].append(v.x)
            car_frenet['vd'].append(v.y)
            # child_frame_id might contain closest waypoint index
            try:
                idx_str = msg.child_frame_id
                car_frenet['closest_idx'].append(int(idx_str) if idx_str.isdigit() else -1)
            except:
                car_frenet['closest_idx'].append(-1)

        elif topic == '/car_state/pose':
            p = msg.pose.position
            q = msg.pose.orientation
            roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            car_pose['t'].append(ts)
            car_pose['x'].append(p.x)
            car_pose['y'].append(p.y)
            car_pose['z'].append(p.z)
            car_pose['roll'].append(roll)
            car_pose['pitch'].append(pitch)
            car_pose['yaw'].append(yaw)

        elif topic == '/glim_ros/base_odom':
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            glim_odom['t'].append(ts)
            glim_odom['x'].append(p.x)
            glim_odom['y'].append(p.y)
            glim_odom['z'].append(p.z)
            glim_odom['roll'].append(roll)
            glim_odom['pitch'].append(pitch)
            glim_odom['yaw'].append(yaw)

        elif topic == '/vesc/high_level/ackermann_cmd_mux/input/nav_1':
            ctrl_input['t'].append(ts)
            ctrl_input['speed'].append(msg.drive.speed)
            ctrl_input['steer'].append(msg.drive.steering_angle)

        elif topic == '/l1_distance':
            l1_dist['t'].append(ts)
            l1_dist['x'].append(msg.x)
            l1_dist['y'].append(msg.y)
            l1_dist['z'].append(msg.z)

        elif topic == '/lookahead_point':
            lookahead_pt['t'].append(ts)
            lookahead_pt['x'].append(msg.pose.position.x)
            lookahead_pt['y'].append(msg.pose.position.y)
            lookahead_pt['z'].append(msg.pose.position.z)

        elif topic == '/local_waypoints':
            if len(msg.wpnts) > 0:
                local_wpnts_data['t'].append(ts)
                local_wpnts_data['first_s'].append(msg.wpnts[0].s_m)
                local_wpnts_data['last_s'].append(msg.wpnts[-1].s_m)
                local_wpnts_data['n_wpnts'].append(len(msg.wpnts))
                local_wpnts_data['first_x'].append(msg.wpnts[0].x_m)
                local_wpnts_data['first_y'].append(msg.wpnts[0].y_m)
                local_wpnts_data['first_z'].append(msg.wpnts[0].z_m if hasattr(msg.wpnts[0], 'z_m') else 0)

        elif topic == '/centerline_waypoints' and centerline_wpnts is None:
            centerline_wpnts = {
                'x': np.array([w.x_m for w in msg.wpnts]),
                'y': np.array([w.y_m for w in msg.wpnts]),
                'z': np.array([w.z_m for w in msg.wpnts]) if hasattr(msg.wpnts[0], 'z_m') else np.zeros(len(msg.wpnts)),
                's': np.array([w.s_m for w in msg.wpnts]),
            }

    bag.close()

    # Convert to numpy
    for d in [car_odom, car_frenet, car_pose, glim_odom, ctrl_input, l1_dist, lookahead_pt, local_wpnts_data]:
        for k, v in d.items():
            d[k] = np.array(v)

    return {
        'car_odom': car_odom,
        'car_frenet': car_frenet,
        'car_pose': car_pose,
        'glim_odom': glim_odom,
        'ctrl_input': ctrl_input,
        'l1_dist': l1_dist,
        'lookahead_pt': lookahead_pt,
        'local_wpnts': local_wpnts_data,
        'centerline_wpnts': centerline_wpnts,
    }


# Extract both bags
bags_data = {}
for path, name in zip(BAG_PATHS, BAG_NAMES):
    bags_data[name] = extract_bag(path)


def compute_closest_wpnt_distance(car_x, car_y, car_z, wpnts):
    """Compute distance from car to closest waypoint and its index."""
    if wpnts is None:
        return None, None
    dists_2d = np.sqrt((wpnts['x'] - car_x)**2 + (wpnts['y'] - car_y)**2)
    dists_3d = np.sqrt((wpnts['x'] - car_x)**2 + (wpnts['y'] - car_y)**2 + (wpnts['z'] - car_z)**2)
    idx_2d = np.argmin(dists_2d)
    idx_3d = np.argmin(dists_3d)
    return {
        'idx_2d': idx_2d, 'dist_2d': dists_2d[idx_2d],
        'idx_3d': idx_3d, 'dist_3d': dists_3d[idx_3d],
        's_2d': wpnts['s'][idx_2d], 's_3d': wpnts['s'][idx_3d],
        'dz': wpnts['z'][idx_3d] - car_z,
    }


# ============================================================
# Per-bag analysis
# ============================================================
for bag_name in BAG_NAMES:
    data = bags_data[bag_name]
    co = data['car_odom']
    cf = data['car_frenet']
    cp = data['car_pose']
    go = data['glim_odom']
    ci = data['ctrl_input']
    l1 = data['l1_dist']
    la = data['lookahead_pt']
    lw = data['local_wpnts']
    wpnts = global_wpnts if global_wpnts is not None else data['centerline_wpnts']

    prefix = f'{bag_name}_'
    print(f"\n=== Generating plots for {bag_name} ===")

    # --------------------------------------------------------
    # F1: XY trajectory + global path + lookahead points
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    ax = axes[0]
    if wpnts is not None:
        ax.plot(wpnts['x'], wpnts['y'], 'g-', linewidth=2, alpha=0.4, label='Global Path')
    sc = ax.scatter(co['x'], co['y'], c=co['t'], cmap='viridis', s=2, label='Actual')
    plt.colorbar(sc, ax=ax, label='Time (s)')
    if len(la['t']) > 0:
        ax.scatter(la['x'], la['y'], c='red', s=5, alpha=0.3, label='Lookahead Points')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{bag_name}: XY Trajectory + Lookahead')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if wpnts is not None:
        ax.plot(wpnts['x'], wpnts['y'], 'g-', linewidth=2, alpha=0.4, label='Global Path')
    # Color by z
    sc = ax.scatter(co['x'], co['y'], c=co['z'], cmap='coolwarm', s=2)
    plt.colorbar(sc, ax=ax, label='Z (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{bag_name}: Trajectory colored by Z')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{prefix}01_xy_trajectory.png'), dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # F2: Frenet s, d, closest_idx over time
    # --------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

    ax = axes[0]
    ax.plot(cf['t'], cf['s'], 'b-', linewidth=0.5)
    ax.set_ylabel('s (m)')
    ax.set_title(f'{bag_name}: Frenet Coordinates')
    ax.grid(True, alpha=0.3)

    # Detect s-jumps
    if len(cf['s']) > 1:
        ds = np.diff(cf['s'])
        s_total = wpnts['s'][-1] if wpnts is not None else np.max(cf['s'])
        # Jumps that are not wrap-arounds
        jump_mask = np.abs(ds) > 5.0  # 5m이상 갑자기 변화
        wrap_mask = np.abs(np.abs(ds) - s_total) < 10.0  # wrap-around
        anomaly_mask = jump_mask & ~wrap_mask
        if np.any(anomaly_mask):
            jump_times = cf['t'][1:][anomaly_mask]
            jump_vals = ds[anomaly_mask]
            for jt, jv in zip(jump_times, jump_vals):
                ax.axvline(x=jt, color='red', linestyle='--', alpha=0.5)
                ax.annotate(f'jump={jv:.1f}m', (jt, cf['s'][np.searchsorted(cf['t'], jt)]),
                           fontsize=7, color='red')

    ax = axes[1]
    ax.plot(cf['t'], cf['d'], 'r-', linewidth=0.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('d (m)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if len(cf['closest_idx']) > 0 and np.any(cf['closest_idx'] >= 0):
        ax.plot(cf['t'], cf['closest_idx'], 'g-', linewidth=0.5, label='Closest WP Index')
        # Detect index jumps
        valid = cf['closest_idx'] >= 0
        if np.sum(valid) > 1:
            didx = np.diff(cf['closest_idx'][valid])
            n_total = len(wpnts['x']) if wpnts is not None else 1000
            idx_jump = np.abs(didx) > 20  # index jump > 20
            wrap = np.abs(np.abs(didx) - n_total) < 30
            idx_anomaly = idx_jump & ~wrap
            if np.any(idx_anomaly):
                t_valid = cf['t'][valid]
                for jt in t_valid[1:][idx_anomaly]:
                    ax.axvline(x=jt, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Closest WP Index')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    if len(l1['t']) > 0:
        ax.plot(l1['t'], l1['x'], 'b-', linewidth=0.5, label='L1 distance')
    ax.set_ylabel('L1 Distance (m)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{prefix}02_frenet_timeseries.png'), dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # F3: Z analysis - car Z vs waypoint Z at closest point
    # --------------------------------------------------------
    if wpnts is not None and len(co['t']) > 0:
        fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

        # Compute expected Z from closest waypoint
        expected_z = []
        actual_z = []
        dz_list = []
        dist_to_closest_2d = []
        dist_to_closest_3d = []
        frenet_s_at_odom = []

        for i in range(len(co['t'])):
            result = compute_closest_wpnt_distance(co['x'][i], co['y'][i], co['z'][i], wpnts)
            if result is not None:
                expected_z.append(wpnts['z'][result['idx_2d']])
                actual_z.append(co['z'][i])
                dz_list.append(co['z'][i] - wpnts['z'][result['idx_2d']])
                dist_to_closest_2d.append(result['dist_2d'])
                dist_to_closest_3d.append(result['dist_3d'])

        expected_z = np.array(expected_z)
        actual_z = np.array(actual_z)
        dz_list = np.array(dz_list)
        dist_to_closest_2d = np.array(dist_to_closest_2d)
        dist_to_closest_3d = np.array(dist_to_closest_3d)

        ax = axes[0]
        ax.plot(co['t'], actual_z, 'b-', linewidth=0.5, label='Actual Z')
        ax.plot(co['t'], expected_z, 'g-', linewidth=0.5, label='Expected Z (2D closest)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{bag_name}: Z Height Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(co['t'], dz_list, 'r-', linewidth=0.5, label='ΔZ (actual - expected)')
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Height filter threshold (0.1m)')
        ax.axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5)
        ax.set_ylabel('ΔZ (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(co['t'], dist_to_closest_2d, 'b-', linewidth=0.5, label='2D dist to closest')
        ax.plot(co['t'], dist_to_closest_3d, 'r-', linewidth=0.5, alpha=0.5, label='3D dist to closest')
        ax.set_ylabel('Distance (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        ax.plot(go['t'], np.degrees(go['roll']), 'r-', linewidth=0.5, label='Roll')
        ax.plot(go['t'], np.degrees(go['pitch']), 'b-', linewidth=0.5, label='Pitch')
        ax.set_ylabel('Angle (deg)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'{prefix}03_z_analysis.png'), dpi=150)
        plt.close(fig)

    # --------------------------------------------------------
    # F4: Lookahead distance vs speed, steer, lateral error
    # --------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

    ax = axes[0]
    if len(l1['t']) > 0:
        ax.plot(l1['t'], l1['x'], 'b-', linewidth=1, label='L1 distance')
        # Mark abnormally long lookahead
        l1_median = np.median(l1['x'])
        l1_thresh = max(l1_median * 2, 2.0)
        long_la = l1['x'] > l1_thresh
        if np.any(long_la):
            ax.scatter(l1['t'][long_la], l1['x'][long_la], c='red', s=10, label=f'Abnormal (>{l1_thresh:.1f}m)')
    ax.set_ylabel('L1 Distance (m)')
    ax.set_title(f'{bag_name}: Lookahead Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(co['t'], co['speed'], 'g-', linewidth=0.5, label='Speed')
    ax.set_ylabel('Speed (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(cf['t'], cf['d'], 'r-', linewidth=0.5, label='Lateral Error d')
    ax.set_ylabel('d (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(ci['t'], ci['steer'], 'm-', linewidth=0.5, label='Steer Cmd')
    ax.set_ylabel('Steer (rad)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{prefix}04_lookahead_analysis.png'), dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # F5: Lookahead point vs car position on XY
    # --------------------------------------------------------
    if len(la['t']) > 0 and len(co['t']) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        if wpnts is not None:
            ax.plot(wpnts['x'], wpnts['y'], 'g-', linewidth=2, alpha=0.3, label='Global Path')

        ax.plot(co['x'], co['y'], 'b-', linewidth=1, alpha=0.5, label='Car Trajectory')

        # Draw lines from car to lookahead point (subsample)
        x_interp = interp1d(co['t'], co['x'], bounds_error=False, fill_value='extrapolate')
        y_interp = interp1d(co['t'], co['y'], bounds_error=False, fill_value='extrapolate')

        step = max(1, len(la['t']) // 100)
        for i in range(0, len(la['t']), step):
            cx = x_interp(la['t'][i])
            cy = y_interp(la['t'][i])
            dist = np.sqrt((la['x'][i] - cx)**2 + (la['y'][i] - cy)**2)
            color = 'red' if dist > 3.0 else 'blue'
            alpha = 0.8 if dist > 3.0 else 0.2
            ax.plot([cx, la['x'][i]], [cy, la['y'][i]], color=color, linewidth=0.5, alpha=alpha)

        ax.scatter(la['x'], la['y'], c='red', s=2, alpha=0.3, label='Lookahead Points')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{bag_name}: Car → Lookahead Lines (red=far, blue=normal)')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'{prefix}05_lookahead_xy.png'), dpi=150)
        plt.close(fig)

    # --------------------------------------------------------
    # F6: Frenet s reported vs 2D-closest s (detect mismatch)
    # --------------------------------------------------------
    if wpnts is not None and len(cf['t']) > 0 and len(co['t']) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

        # Compute 2D-closest s at car_odom times
        s_2d_closest = []
        for i in range(len(co['t'])):
            dists = np.sqrt((wpnts['x'] - co['x'][i])**2 + (wpnts['y'] - co['y'][i])**2)
            idx = np.argmin(dists)
            s_2d_closest.append(wpnts['s'][idx])
        s_2d_closest = np.array(s_2d_closest)

        # Interpolate frenet s to odom times
        s_frenet_interp = interp1d(cf['t'], cf['s'], bounds_error=False, fill_value='extrapolate')(co['t'])

        ax = axes[0]
        ax.plot(co['t'], s_frenet_interp, 'b-', linewidth=0.5, label='Frenet s (reported)')
        ax.plot(co['t'], s_2d_closest, 'r-', linewidth=0.5, alpha=0.5, label='2D closest s')
        ax.set_ylabel('s (m)')
        ax.set_title(f'{bag_name}: Frenet s vs 2D Closest s')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        # s difference (account for wrap-around)
        s_total = wpnts['s'][-1]
        ds = s_frenet_interp - s_2d_closest
        # Unwrap
        ds = np.where(ds > s_total/2, ds - s_total, ds)
        ds = np.where(ds < -s_total/2, ds + s_total, ds)
        ax.plot(co['t'], ds, 'r-', linewidth=0.5)
        ax.axhline(y=0, color='k', linestyle='--')
        ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='±5m threshold')
        ax.axhline(y=-5, color='orange', linestyle='--', alpha=0.5)
        ax.set_ylabel('Δs (frenet - 2D closest) (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(co['t'], co['z'], 'b-', linewidth=0.5, label='Car Z')
        ax.set_ylabel('Z (m)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'{prefix}06_frenet_s_mismatch.png'), dpi=150)
        plt.close(fig)

    # --------------------------------------------------------
    # F7: Z vs height filter analysis
    # --------------------------------------------------------
    if wpnts is not None and len(co['t']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))

        # For each car position, check how many waypoints pass height filter
        n_pass_height = []
        n_total = len(wpnts['x'])
        height_threshold = 0.10  # from code

        # Sample every Nth point for speed
        sample_step = max(1, len(co['t']) // 500)
        sample_idx = np.arange(0, len(co['t']), sample_step)

        for i in sample_idx:
            dx = wpnts['x'] - co['x'][i]
            dy = wpnts['y'] - co['y'][i]
            dz = wpnts['z'] - co['z'][i]

            # Simple height offset: just dz (simplified, actual uses surface normal)
            # More accurate: need mu (pitch) at each waypoint
            height_offsets = np.abs(dz)

            # Nearby waypoints (2D < 2m)
            dist_2d = np.sqrt(dx**2 + dy**2)
            nearby = dist_2d < 2.0

            n_nearby = np.sum(nearby)
            n_pass = np.sum(nearby & (height_offsets < height_threshold))
            n_pass_height.append(n_pass)

        n_pass_height = np.array(n_pass_height)

        ax = axes[0, 0]
        ax.plot(co['t'][sample_idx], n_pass_height, 'b-', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', label='No waypoints pass filter!')
        ax.set_ylabel('# Nearby WPs passing height filter')
        ax.set_title(f'{bag_name}: Height Filter Pass Count (nearby 2m, thresh={height_threshold}m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        # dz histogram
        all_dz = []
        for i in sample_idx[::10]:  # further subsample
            dz_vals = wpnts['z'] - co['z'][i]
            dist_2d = np.sqrt((wpnts['x'] - co['x'][i])**2 + (wpnts['y'] - co['y'][i])**2)
            nearby = dist_2d < 3.0
            if np.any(nearby):
                all_dz.extend(dz_vals[nearby].tolist())
        if len(all_dz) > 0:
            ax.hist(all_dz, bins=100, alpha=0.7)
            ax.axvline(x=height_threshold, color='r', linestyle='--', label=f'±{height_threshold}m')
            ax.axvline(x=-height_threshold, color='r', linestyle='--')
        ax.set_xlabel('ΔZ to nearby waypoints (m)')
        ax.set_ylabel('Count')
        ax.set_title('Height Offset Distribution (nearby waypoints)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(co['t'], co['z'], 'b-', linewidth=0.5, label='Car Z')
        if len(cf['t']) > 0:
            # Expected Z from frenet s
            s_interp_fn = interp1d(cf['t'], cf['s'], bounds_error=False, fill_value='extrapolate')
            s_at_odom = s_interp_fn(co['t'])
            expected_z_from_s = np.interp(s_at_odom % wpnts['s'][-1], wpnts['s'], wpnts['z'])
            ax.plot(co['t'], expected_z_from_s, 'g-', linewidth=0.5, alpha=0.5, label='Expected Z (from Frenet s)')
        ax.set_ylabel('Z (m)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        # XY colored by number of height-passing waypoints
        if wpnts is not None:
            ax.plot(wpnts['x'], wpnts['y'], 'g-', linewidth=1, alpha=0.3)
        sc = ax.scatter(co['x'][sample_idx], co['y'][sample_idx], c=n_pass_height,
                       cmap='RdYlGn', s=5, vmin=0, vmax=max(np.max(n_pass_height), 1))
        plt.colorbar(sc, ax=ax, label='# WPs passing height filter')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Map of Height Filter Pass Count')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'{prefix}07_height_filter.png'), dpi=150)
        plt.close(fig)

    # --------------------------------------------------------
    # F8: Lookahead distance distribution & anomalies
    # --------------------------------------------------------
    if len(l1['t']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax = axes[0, 0]
        ax.hist(l1['x'], bins=50, alpha=0.7)
        ax.set_xlabel('L1 Distance (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'{bag_name}: L1 Distance Distribution')
        ax.axvline(x=np.median(l1['x']), color='r', linestyle='--', label=f'Median={np.median(l1["x"]):.2f}m')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # L1 distance vs speed
        ax = axes[0, 1]
        if len(co['t']) > 1:
            speed_at_l1 = interp1d(co['t'], co['speed'], bounds_error=False, fill_value='extrapolate')(l1['t'])
            ax.scatter(speed_at_l1, l1['x'], s=1, alpha=0.3)
            ax.set_xlabel('Speed (m/s)')
            ax.set_ylabel('L1 Distance (m)')
            ax.set_title('L1 Distance vs Speed')
            ax.grid(True, alpha=0.3)

        # Actual lookahead distance (car to lookahead point)
        ax = axes[1, 0]
        if len(la['t']) > 0 and len(co['t']) > 1:
            cx = interp1d(co['t'], co['x'], bounds_error=False, fill_value='extrapolate')(la['t'])
            cy = interp1d(co['t'], co['y'], bounds_error=False, fill_value='extrapolate')(la['t'])
            cz = interp1d(co['t'], co['z'], bounds_error=False, fill_value='extrapolate')(la['t'])
            actual_la_dist = np.sqrt((la['x'] - cx)**2 + (la['y'] - cy)**2 + (la['z'] - cz)**2)
            ax.plot(la['t'], actual_la_dist, 'b-', linewidth=0.5, label='Actual 3D dist to LA point')
            if len(l1['t']) > 0:
                ax.plot(l1['t'], l1['x'], 'r-', linewidth=0.5, alpha=0.5, label='Commanded L1')
            ax.set_ylabel('Distance (m)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Actual vs Commanded Lookahead Distance')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # L1 vs |d| (does error affect lookahead?)
        ax = axes[1, 1]
        if len(cf['t']) > 1 and len(l1['t']) > 0:
            d_at_l1 = interp1d(cf['t'], np.abs(cf['d']), bounds_error=False, fill_value='extrapolate')(l1['t'])
            ax.scatter(d_at_l1, l1['x'], s=1, alpha=0.3)
            ax.set_xlabel('|d| Lateral Error (m)')
            ax.set_ylabel('L1 Distance (m)')
            ax.set_title('L1 Distance vs Lateral Error')
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'{prefix}08_lookahead_stats.png'), dpi=150)
        plt.close(fig)

    print(f"  Plots for {bag_name} saved to {OUT_DIR}")


# ============================================================
# Combined comparison plot
# ============================================================
print("\n=== Generating combined comparison ===")

fig, axes = plt.subplots(2, 2, figsize=(24, 16))

for i, bag_name in enumerate(BAG_NAMES):
    data = bags_data[bag_name]
    co = data['car_odom']
    cf = data['car_frenet']

    ax = axes[0, i]
    if global_wpnts is not None:
        ax.plot(global_wpnts['x'], global_wpnts['y'], 'g-', linewidth=2, alpha=0.3)
    sc = ax.scatter(co['x'], co['y'], c=co['z'], cmap='coolwarm', s=2)
    plt.colorbar(sc, ax=ax, label='Z (m)')
    ax.set_title(f'{bag_name}: XY (colored by Z)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1, i]
    ax.plot(cf['t'], cf['s'], 'b-', linewidth=0.5)
    ax.set_ylabel('s (m)')
    ax2 = ax.twinx()
    ax2.plot(cf['t'], cf['d'], 'r-', linewidth=0.5)
    ax2.set_ylabel('d (m)', color='r')
    ax.set_title(f'{bag_name}: Frenet s & d')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'combined_comparison.png'), dpi=150)
plt.close(fig)
print("  combined_comparison.png saved")

print(f"\n=== All Frenet analysis plots saved to {OUT_DIR} ===")
