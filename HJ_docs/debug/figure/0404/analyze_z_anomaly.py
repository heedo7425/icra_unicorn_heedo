#!/usr/bin/env python3
"""
Z 이상치 분석: 경사면 vs 평탄구간에서의 Z 거동
- 맵의 경사 프로파일과 실제 차량 Z를 대조
- 평탄구간에서 Z가 비정상적으로 튀는지 확인
- height filter 문제의 근본 원인 파악
"""
import rosbag
import rospy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
from scipy.interpolate import interp1d
import os
import json

OUT_DIR = '/home/unicorn/catkin_ws/src/race_stack/HJ_docs/debug/figure/0404/z_anomaly'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Load map data
# ============================================================
MAP_DIR = '/home/unicorn/catkin_ws/src/race_stack/stack_master/maps/eng_0404_v2'
d = json.load(open(os.path.join(MAP_DIR, 'global_waypoints.json')))
iqp = d['global_traj_wpnts_iqp']['wpnts']

map_s = np.array([w['s_m'] for w in iqp])
map_x = np.array([w['x_m'] for w in iqp])
map_y = np.array([w['y_m'] for w in iqp])
map_z = np.array([w['z_m'] for w in iqp])
map_mu = np.array([w['mu_rad'] for w in iqp])
map_psi = np.array([w['psi_rad'] for w in iqp])
s_total = map_s[-1]

# Compute slope
map_dz_ds = np.gradient(map_z, map_s)

# Slope classification
SLOPE_THRESH = 0.03  # |dz/ds| > 0.03 => slope region
is_slope = np.abs(map_dz_ds) > SLOPE_THRESH

print(f"Track: {s_total:.1f}m, Z range: {map_z.min():.3f}~{map_z.max():.3f}m")
print(f"Slope region: s=27.7~36.2m (hill), Z delta≈0.5m")
print(f"Flat regions: s=0~27m, s=36~60m (Z≈-0.05~-0.13m)")

# ============================================================
# 2. Helper: find closest waypoint (2D) and expected Z
# ============================================================
def get_expected_z_and_s(car_x, car_y):
    """Return expected Z and s from 2D closest waypoint"""
    d2 = (map_x - car_x)**2 + (map_y - car_y)**2
    idx = np.argmin(d2)
    return map_z[idx], map_s[idx], idx, np.sqrt(d2[idx])

def compute_height_offset_proper(car_x, car_y, car_z, wp_idx):
    """Compute height offset using surface normal projection (matching C++ code)"""
    dx = car_x - map_x[wp_idx]
    dy = car_y - map_y[wp_idx]
    dz = car_z - map_z[wp_idx]
    mu = map_mu[wp_idx]
    psi = map_psi[wp_idx]
    # Surface normal projection: cos(psi)*sin(mu)*dx + sin(psi)*sin(mu)*dy + cos(mu)*dz
    h = np.cos(psi)*np.sin(mu)*dx + np.sin(psi)*np.sin(mu)*dy + np.cos(mu)*dz
    return h

# ============================================================
# 3. Process all 3 bags
# ============================================================
BAG_CONFIGS = [
    ('/home/unicorn/catkin_ws/src/race_stack/bag/2026-04-04-16-52-19.bag', 'bag1_1652', 0),
    ('/home/unicorn/catkin_ws/src/race_stack/bag/2026-04-04-16-55-36.bag', 'bag2_1655', 0),
    ('/home/unicorn/catkin_ws/src/race_stack/bag/2026-04-04-17-07-32.bag', 'bag3_1707', 90),
]

for bag_path, bag_name, start_offset in BAG_CONFIGS:
    print(f"\n=== Processing {bag_name} ===")
    bag = rosbag.Bag(bag_path, 'r')
    t0 = bag.get_start_time() + start_offset

    car = {'t': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []}
    frenet = {'t': [], 's': [], 'd': []}

    for topic, msg, t in bag.read_messages(
            topics=['/glim_ros/base_odom', '/car_state/odom_frenet'],
            start_time=rospy.Time.from_sec(t0)):
        ts = t.to_sec() - t0

        if topic == '/glim_ros/base_odom':
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            car['t'].append(ts)
            car['x'].append(p.x)
            car['y'].append(p.y)
            car['z'].append(p.z)
            car['roll'].append(roll)
            car['pitch'].append(pitch)
            car['yaw'].append(yaw)

        elif topic == '/car_state/odom_frenet':
            frenet['t'].append(ts)
            frenet['s'].append(msg.pose.pose.position.x)
            frenet['d'].append(msg.pose.pose.position.y)

    bag.close()

    for dd in [car, frenet]:
        for k, v in dd.items():
            dd[k] = np.array(v)

    if len(car['t']) == 0:
        print(f"  No data, skipping")
        continue

    print(f"  {len(car['t'])} odom messages, {len(frenet['t'])} frenet messages")

    # ---- Compute expected Z and height offsets ----
    expected_z = np.zeros(len(car['t']))
    car_s = np.zeros(len(car['t']))
    dist_2d = np.zeros(len(car['t']))
    simple_dz = np.zeros(len(car['t']))
    proper_h = np.zeros(len(car['t']))
    wp_mu = np.zeros(len(car['t']))
    wp_is_slope = np.zeros(len(car['t']), dtype=bool)

    for i in range(len(car['t'])):
        ez, es, idx, d2d = get_expected_z_and_s(car['x'][i], car['y'][i])
        expected_z[i] = ez
        car_s[i] = es
        dist_2d[i] = d2d
        simple_dz[i] = car['z'][i] - ez
        proper_h[i] = compute_height_offset_proper(car['x'][i], car['y'][i], car['z'][i], idx)
        wp_mu[i] = map_mu[idx]
        wp_is_slope[i] = is_slope[idx]

    # ---- Classify: flat vs slope ----
    flat_mask = ~wp_is_slope
    slope_mask = wp_is_slope

    # ============================================================
    # PLOT Z1: Z profile comparison (car vs map)
    # ============================================================
    fig, axes = plt.subplots(5, 1, figsize=(22, 22), sharex=True)

    ax = axes[0]
    ax.plot(car['t'], car['z'], 'b-', linewidth=0.8, label='Car Z (GLIL)')
    ax.plot(car['t'], expected_z, 'g-', linewidth=0.8, alpha=0.7, label='Expected Z (map)')
    ax.fill_between(car['t'], expected_z - 0.1, expected_z + 0.1, alpha=0.15, color='green', label='±0.1m band')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'{bag_name}: Z Profile — Car vs Map Expected')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(car['t'], simple_dz, 'r-', linewidth=0.5, alpha=0.7, label='Simple ΔZ (car - map)')
    ax.plot(car['t'], proper_h, 'b-', linewidth=0.5, alpha=0.7, label='Proper height offset (surface normal)')
    ax.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='Height filter ±0.10m')
    ax.axhline(y=-0.10, color='orange', linestyle='--', alpha=0.7)
    ax.set_ylabel('Height Offset (m)')
    ax.set_title('Height Offset: Simple ΔZ vs Surface Normal Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # Color by flat/slope
    t_flat = car['t'][flat_mask]
    h_flat = proper_h[flat_mask]
    t_slope = car['t'][slope_mask]
    h_slope = proper_h[slope_mask]
    ax.scatter(t_flat, h_flat, c='blue', s=1, alpha=0.5, label=f'Flat region ({np.sum(flat_mask)} pts)')
    ax.scatter(t_slope, h_slope, c='red', s=1, alpha=0.5, label=f'Slope region ({np.sum(slope_mask)} pts)')
    ax.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='Filter ±0.10m')
    ax.axhline(y=-0.10, color='orange', linestyle='--', alpha=0.7)
    ax.set_ylabel('Height Offset (m)')
    ax.set_title('Height Offset by Region Type')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(car['t'], car_s, 'purple', linewidth=0.5, label='Car s (track position)')
    # Mark slope regions
    ax.axhspan(27.7, 36.2, alpha=0.2, color='red', label='Hill region (s=27.7~36.2m)')
    ax.set_ylabel('s (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(car['t'], np.degrees(wp_mu), 'g-', linewidth=0.5, label='Track pitch μ at car pos (deg)')
    ax.plot(car['t'], np.degrees(car['pitch']), 'b-', linewidth=0.5, alpha=0.5, label='Car pitch (GLIL)')
    ax.set_ylabel('Pitch (deg)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{bag_name}_z1_profile.png'), dpi=150)
    plt.close(fig)
    print(f"  {bag_name}_z1_profile.png saved")

    # ============================================================
    # PLOT Z2: Height offset statistics
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Histogram: flat region height offsets
    ax = axes[0, 0]
    if np.any(flat_mask):
        ax.hist(proper_h[flat_mask], bins=80, alpha=0.7, color='blue', density=True, label='Flat')
    ax.axvline(x=0.10, color='r', linestyle='--', label='±0.10m threshold')
    ax.axvline(x=-0.10, color='r', linestyle='--')
    ax.set_xlabel('Height Offset (m)')
    ax.set_ylabel('Density')
    ax.set_title('Flat Region: Height Offset Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stats
    if np.any(flat_mask):
        flat_h = proper_h[flat_mask]
        pct_pass = np.sum(np.abs(flat_h) < 0.10) / len(flat_h) * 100
        ax.text(0.02, 0.95, f'Mean: {np.mean(flat_h):.4f}m\nStd: {np.std(flat_h):.4f}m\n'
                f'Pass 0.1m: {pct_pass:.1f}%\nPass 0.2m: {np.sum(np.abs(flat_h)<0.2)/len(flat_h)*100:.1f}%',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Histogram: slope region
    ax = axes[0, 1]
    if np.any(slope_mask):
        ax.hist(proper_h[slope_mask], bins=80, alpha=0.7, color='red', density=True, label='Slope')
    ax.axvline(x=0.10, color='r', linestyle='--')
    ax.axvline(x=-0.10, color='r', linestyle='--')
    ax.set_xlabel('Height Offset (m)')
    ax.set_title('Slope Region: Height Offset Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if np.any(slope_mask):
        slope_h = proper_h[slope_mask]
        pct_pass = np.sum(np.abs(slope_h) < 0.10) / len(slope_h) * 100
        ax.text(0.02, 0.95, f'Mean: {np.mean(slope_h):.4f}m\nStd: {np.std(slope_h):.4f}m\n'
                f'Pass 0.1m: {pct_pass:.1f}%\nPass 0.2m: {np.sum(np.abs(slope_h)<0.2)/len(slope_h)*100:.1f}%',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Simple ΔZ vs proper height offset scatter
    ax = axes[0, 2]
    ax.scatter(simple_dz, proper_h, c=np.degrees(wp_mu), cmap='coolwarm', s=1, alpha=0.3)
    ax.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.3, label='1:1 line')
    ax.set_xlabel('Simple ΔZ (m)')
    ax.set_ylabel('Proper Height Offset (m)')
    ax.set_title('Simple ΔZ vs Surface Normal Projection\n(colored by track pitch)')
    plt.colorbar(ax.collections[0], ax=ax, label='Track pitch μ (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # XY colored by height offset
    ax = axes[1, 0]
    ax.plot(map_x, map_y, 'g-', linewidth=1, alpha=0.3)
    sc = ax.scatter(car['x'], car['y'], c=proper_h, cmap='RdBu_r', s=2, vmin=-0.15, vmax=0.15)
    plt.colorbar(sc, ax=ax, label='Height Offset (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Map: Height Offset (blue=below, red=above)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # XY colored by |offset| > threshold (pass/fail)
    ax = axes[1, 1]
    ax.plot(map_x, map_y, 'g-', linewidth=1, alpha=0.3)
    pass_mask = np.abs(proper_h) < 0.10
    fail_mask = ~pass_mask
    ax.scatter(car['x'][pass_mask], car['y'][pass_mask], c='green', s=2, alpha=0.3, label=f'PASS ({np.sum(pass_mask)})')
    ax.scatter(car['x'][fail_mask], car['y'][fail_mask], c='red', s=4, alpha=0.8, label=f'FAIL ({np.sum(fail_mask)})')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Height Filter Pass/Fail (threshold=0.10m)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # s-based view: height offset vs s
    ax = axes[1, 2]
    ax.scatter(car_s, proper_h, c=car['t'], cmap='viridis', s=1, alpha=0.3)
    ax.axhline(y=0.10, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.10, color='r', linestyle='--', alpha=0.5)
    ax.axvspan(27.7, 36.2, alpha=0.15, color='red', label='Hill')
    ax.set_xlabel('s (m)')
    ax.set_ylabel('Height Offset (m)')
    ax.set_title('Height Offset vs Track Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='Time (s)')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{bag_name}_z2_stats.png'), dpi=150)
    plt.close(fig)
    print(f"  {bag_name}_z2_stats.png saved")

    # ============================================================
    # PLOT Z3: Z anomaly detection — flat region 에서 unexpected Z jumps
    # ============================================================
    fig, axes = plt.subplots(3, 1, figsize=(22, 14), sharex=True)

    ax = axes[0]
    # On flat regions, z should be nearly constant (matching map_z)
    # Anomaly = |simple_dz| > 0.05 on flat region
    z_anomaly = flat_mask & (np.abs(simple_dz) > 0.05)
    z_normal = flat_mask & (np.abs(simple_dz) <= 0.05)

    ax.plot(car['t'], car['z'], 'b-', linewidth=0.5, alpha=0.5, label='Car Z')
    ax.plot(car['t'], expected_z, 'g-', linewidth=0.5, alpha=0.5, label='Map Z')
    if np.any(z_anomaly):
        ax.scatter(car['t'][z_anomaly], car['z'][z_anomaly], c='red', s=10, zorder=5,
                  label=f'Z Anomaly on flat ({np.sum(z_anomaly)} pts)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'{bag_name}: Z Anomaly Detection (flat region, |ΔZ|>0.05m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Rate of Z change (dz/dt) — should be near 0 on flat, smooth on slopes
    if len(car['t']) > 1:
        dt = np.diff(car['t'])
        dz_dt = np.diff(car['z']) / np.maximum(dt, 1e-6)
        ax.plot(car['t'][1:], dz_dt, 'b-', linewidth=0.3, alpha=0.3)
        # Smooth
        w = min(20, len(dz_dt))
        if w > 1:
            dz_dt_smooth = np.convolve(dz_dt, np.ones(w)/w, mode='same')
            ax.plot(car['t'][1:], dz_dt_smooth, 'b-', linewidth=1, label='dz/dt (smooth)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('dz/dt (m/s)')
        ax.set_title('Z Rate of Change')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    # How many nearby waypoints (2D<2m) would pass different thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    sample_step = max(1, len(car['t']) // 500)
    sample_idx = np.arange(0, len(car['t']), sample_step)

    for thresh in thresholds:
        n_pass = np.zeros(len(sample_idx))
        for j, i in enumerate(sample_idx):
            dx = map_x - car['x'][i]
            dy = map_y - car['y'][i]
            dz = map_z - car['z'][i]
            d2d = np.sqrt(dx**2 + dy**2)
            nearby = d2d < 2.0
            if np.any(nearby):
                nearby_idx = np.where(nearby)[0]
                h_offsets = np.array([compute_height_offset_proper(
                    car['x'][i], car['y'][i], car['z'][i], wi) for wi in nearby_idx])
                n_pass[j] = np.sum(np.abs(h_offsets) < thresh)
        ax.plot(car['t'][sample_idx], n_pass, linewidth=0.8, label=f'thresh={thresh}m')

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('# Nearby WPs passing filter')
    ax.set_xlabel('Time (s)')
    ax.set_title('Height Filter Pass Count with Different Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{bag_name}_z3_anomaly.png'), dpi=150)
    plt.close(fig)
    print(f"  {bag_name}_z3_anomaly.png saved")

    # ============================================================
    # PLOT Z4: Frenet s consistency check — does s jump on slope?
    # ============================================================
    if len(frenet['t']) > 10:
        fig, axes = plt.subplots(3, 1, figsize=(22, 14), sharex=True)

        # Interpolate frenet s to car times
        s_frenet = interp1d(frenet['t'], frenet['s'], bounds_error=False, fill_value='extrapolate')(car['t'])
        d_frenet = interp1d(frenet['t'], frenet['d'], bounds_error=False, fill_value='extrapolate')(car['t'])

        ax = axes[0]
        ax.plot(car['t'], s_frenet, 'b-', linewidth=0.5, label='Frenet s')
        ax.plot(car['t'], car_s, 'r-', linewidth=0.5, alpha=0.5, label='2D-closest s')
        ax.axhspan(27.7, 36.2, alpha=0.15, color='red', label='Hill')
        ax.set_ylabel('s (m)')
        ax.set_title(f'{bag_name}: Frenet s vs 2D-closest s')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ds = s_frenet - car_s
        # Unwrap
        ds = np.where(ds > s_total/2, ds - s_total, ds)
        ds = np.where(ds < -s_total/2, ds + s_total, ds)
        ax.plot(car['t'], ds, 'r-', linewidth=0.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        # Color background by slope/flat
        for i in range(len(car['t'])-1):
            if wp_is_slope[i]:
                ax.axvspan(car['t'][i], car['t'][i+1], alpha=0.05, color='red')
        ax.set_ylabel('Δs (frenet - 2D closest)')
        ax.set_title('s Mismatch (red background = slope region)')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(car['t'], d_frenet, 'g-', linewidth=0.5, label='Frenet d')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        for i in range(len(car['t'])-1):
            if wp_is_slope[i]:
                ax.axvspan(car['t'][i], car['t'][i+1], alpha=0.05, color='red')
        ax.set_ylabel('d (m)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Lateral Error (red background = slope region)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'{bag_name}_z4_frenet_slope.png'), dpi=150)
        plt.close(fig)
        print(f"  {bag_name}_z4_frenet_slope.png saved")

print(f"\n=== All Z anomaly analysis saved to {OUT_DIR} ===")
