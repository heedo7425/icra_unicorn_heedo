#!/usr/bin/env python3
"""
문제 1 분석: 경로 추종 실패 (2026-04-04-17-07-32.bag, --start 90)
- 오버슛, 직선 밀림, 롤 기울기 영향 분석
"""
import rosbag
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tf.transformations import euler_from_quaternion
import os
import json

BAG_PATH = '/home/unicorn/catkin_ws/src/race_stack/bag/2026-04-04-17-07-32.bag'
OUT_DIR = '/home/unicorn/catkin_ws/src/race_stack/HJ_docs/debug/figure/0404/tracking'
START_OFFSET = 90.0  # --start 90

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 데이터 추출
# ============================================================
print("=== Extracting data from bag ===")

topics = [
    '/car_state/odom',           # nav_msgs/Odometry - 차량 속도, 위치
    '/car_state/odom_frenet',    # nav_msgs/Odometry - frenet 좌표 (s, d)
    '/car_state/pose',           # geometry_msgs/PoseStamped - 차량 위치/자세
    '/glim_ros/base_odom',       # nav_msgs/Odometry - GLIL odom (roll, pitch 포함)
    '/vesc/high_level/ackermann_cmd_mux/input/nav_1',  # 제어 입력
    '/vesc/low_level/ackermann_cmd_mux/output',        # 실제 출력 명령
    '/vesc/commands/servo/position',   # 실제 서보 명령
    '/vesc/commands/motor/speed',      # 실제 모터 명령
    '/local_waypoints',          # 로컬 웨이포인트
    '/l1_distance',              # lookahead 거리
    '/lookahead_point',          # lookahead 포인트 위치
    '/ekf/imu/data',            # IMU 데이터
    '/centerline_waypoints',     # 센터라인
    '/state_machine',            # 상태머신
    '/behavior_strategy',        # 행동 전략
]

bag = rosbag.Bag(BAG_PATH, 'r')
import rospy
start_time_sec = bag.get_start_time() + START_OFFSET
start_time_ros = rospy.Time.from_sec(start_time_sec)

# Data containers
car_odom = {'t': [], 'x': [], 'y': [], 'z': [], 'vx': [], 'vy': [], 'speed': [], 'yaw': []}
car_frenet = {'t': [], 's': [], 'd': [], 'vs': [], 'vd': []}
car_pose = {'t': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []}
glim_odom = {'t': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': [], 'vx': [], 'vy': [], 'vz': []}
ctrl_input = {'t': [], 'speed': [], 'steer': []}
ctrl_output = {'t': [], 'speed': [], 'steer': []}
servo_cmd = {'t': [], 'val': []}
motor_cmd = {'t': [], 'val': []}
l1_dist = {'t': [], 'x': [], 'y': [], 'z': []}
lookahead_pt = {'t': [], 'x': [], 'y': [], 'z': []}
imu_data = {'t': [], 'roll': [], 'pitch': [], 'yaw': [], 'ax': [], 'ay': [], 'az': [],
            'wx': [], 'wy': [], 'wz': []}
local_wpnts_speed = {'t': [], 'speeds': []}  # speed of waypoints near car
state_machine = {'t': [], 'state': []}

# Global waypoints (first message only)
global_wpnts = None

print("Reading bag...")
for topic, msg, t in bag.read_messages(topics=topics, start_time=start_time_ros):
    ts = t.to_sec() - start_time_sec  # relative time from start

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
        v = msg.twist.twist.linear
        glim_odom['t'].append(ts)
        glim_odom['x'].append(p.x)
        glim_odom['y'].append(p.y)
        glim_odom['z'].append(p.z)
        glim_odom['roll'].append(roll)
        glim_odom['pitch'].append(pitch)
        glim_odom['yaw'].append(yaw)
        glim_odom['vx'].append(v.x)
        glim_odom['vy'].append(v.y)
        glim_odom['vz'].append(v.z)

    elif topic == '/vesc/high_level/ackermann_cmd_mux/input/nav_1':
        ctrl_input['t'].append(ts)
        ctrl_input['speed'].append(msg.drive.speed)
        ctrl_input['steer'].append(msg.drive.steering_angle)

    elif topic == '/vesc/low_level/ackermann_cmd_mux/output':
        ctrl_output['t'].append(ts)
        ctrl_output['speed'].append(msg.drive.speed)
        ctrl_output['steer'].append(msg.drive.steering_angle)

    elif topic == '/vesc/commands/servo/position':
        servo_cmd['t'].append(ts)
        servo_cmd['val'].append(msg.data)

    elif topic == '/vesc/commands/motor/speed':
        motor_cmd['t'].append(ts)
        motor_cmd['val'].append(msg.data)

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

    elif topic == '/ekf/imu/data':
        q = msg.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        imu_data['t'].append(ts)
        imu_data['roll'].append(roll)
        imu_data['pitch'].append(pitch)
        imu_data['yaw'].append(yaw)
        imu_data['ax'].append(msg.linear_acceleration.x)
        imu_data['ay'].append(msg.linear_acceleration.y)
        imu_data['az'].append(msg.linear_acceleration.z)
        imu_data['wx'].append(msg.angular_velocity.x)
        imu_data['wy'].append(msg.angular_velocity.y)
        imu_data['wz'].append(msg.angular_velocity.z)

    elif topic == '/local_waypoints':
        # Store speed of first few waypoints (near car)
        if len(msg.wpnts) > 0:
            speeds = [w.vx_mps for w in msg.wpnts[:5]]
            local_wpnts_speed['t'].append(ts)
            local_wpnts_speed['speeds'].append(np.mean(speeds))

    elif topic == '/centerline_waypoints' and global_wpnts is None:
        global_wpnts = {
            'x': [w.x_m for w in msg.wpnts],
            'y': [w.y_m for w in msg.wpnts],
            'z': [w.z_m for w in msg.wpnts] if hasattr(msg.wpnts[0], 'z_m') else [0]*len(msg.wpnts),
            's': [w.s_m for w in msg.wpnts],
            'vx': [w.vx_mps for w in msg.wpnts],
        }

    elif topic == '/state_machine':
        state_machine['t'].append(ts)
        state_machine['state'].append(msg.data)

bag.close()

# Convert all to numpy
for d in [car_odom, car_frenet, car_pose, glim_odom, ctrl_input, ctrl_output,
          servo_cmd, motor_cmd, l1_dist, lookahead_pt, imu_data, local_wpnts_speed, state_machine]:
    for k, v in d.items():
        if k != 'state':
            d[k] = np.array(v)

print(f"Data extracted: car_odom={len(car_odom['t'])}, car_frenet={len(car_frenet['t'])}, "
      f"glim_odom={len(glim_odom['t'])}, ctrl_input={len(ctrl_input['t'])}")

# ============================================================
# 2. 글로벌 웨이포인트 로드 (맵 파일에서)
# ============================================================
MAP_DIR = '/home/unicorn/catkin_ws/src/race_stack/stack_master/maps/eng_0404_v2'
map_json = os.path.join(MAP_DIR, 'global_waypoints.json')
if os.path.exists(map_json) and global_wpnts is None:
    with open(map_json) as f:
        data = json.load(f)
    # JSON is a dict with nested waypoint arrays
    wpnt_key = 'global_traj_wpnts_sp'  # speed-profile optimized trajectory
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

if global_wpnts is not None:
    for k, v in global_wpnts.items():
        global_wpnts[k] = np.array(v)

# ============================================================
# PLOT 1: 전체 궤적 (XY) + 글로벌 경로 비교
# ============================================================
print("=== Generating plots ===")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# XY trajectory
ax = axes[0]
if global_wpnts is not None:
    ax.plot(global_wpnts['x'], global_wpnts['y'], 'g-', linewidth=1, alpha=0.5, label='Global Path')
sc = ax.scatter(car_odom['x'], car_odom['y'], c=car_odom['t'], cmap='viridis', s=1, label='Actual Trajectory')
plt.colorbar(sc, ax=ax, label='Time (s)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('XY Trajectory vs Global Path')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

# XY trajectory colored by speed
ax = axes[1]
if global_wpnts is not None:
    ax.plot(global_wpnts['x'], global_wpnts['y'], 'g-', linewidth=1, alpha=0.5, label='Global Path')
sc = ax.scatter(car_odom['x'], car_odom['y'], c=car_odom['speed'], cmap='jet', s=1, vmin=0, vmax=8)
plt.colorbar(sc, ax=ax, label='Speed (m/s)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('XY Trajectory (colored by speed)')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '01_xy_trajectory.png'), dpi=150)
plt.close(fig)
print("  01_xy_trajectory.png saved")

# ============================================================
# PLOT 2: Frenet 좌표 - s, d over time
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

ax = axes[0]
ax.plot(car_frenet['t'], car_frenet['s'], 'b-', linewidth=0.5)
ax.set_ylabel('s (m)')
ax.set_title('Frenet Coordinates over Time')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(car_frenet['t'], car_frenet['d'], 'r-', linewidth=0.5)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.set_ylabel('d (m) - Lateral Error')
ax.grid(True, alpha=0.3)

# Add threshold lines
ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='d=±0.3m')
ax.axhline(y=-0.3, color='orange', linestyle='--', alpha=0.5)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='d=±0.5m')
ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
ax.legend()

ax = axes[2]
ax.plot(car_frenet['t'], car_frenet['vd'], 'm-', linewidth=0.5, label='vd (lateral vel)')
ax.set_ylabel('vd (m/s)')
ax.set_xlabel('Time (s)')
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '02_frenet_coords.png'), dpi=150)
plt.close(fig)
print("  02_frenet_coords.png saved")

# ============================================================
# PLOT 3: 속도 분석 - 명령 vs 실제
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

ax = axes[0]
ax.plot(car_odom['t'], car_odom['speed'], 'b-', linewidth=0.5, label='Actual Speed')
ax.plot(ctrl_input['t'], ctrl_input['speed'], 'r-', linewidth=0.5, alpha=0.5, label='Cmd Speed')
if len(local_wpnts_speed['t']) > 0:
    ax.plot(local_wpnts_speed['t'], local_wpnts_speed['speeds'], 'g-', linewidth=0.5, alpha=0.5, label='Wpnt Speed (avg)')
ax.set_ylabel('Speed (m/s)')
ax.set_title('Speed: Command vs Actual')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ctrl_input['t'], ctrl_input['steer'], 'r-', linewidth=0.5, label='Cmd Steer (nav)')
ax.plot(ctrl_output['t'], ctrl_output['steer'], 'b-', linewidth=0.5, alpha=0.5, label='Output Steer')
ax.set_ylabel('Steering Angle (rad)')
ax.set_title('Steering Input')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
if len(servo_cmd['t']) > 0:
    ax.plot(servo_cmd['t'], servo_cmd['val'], 'g-', linewidth=0.5, label='Servo Position')
ax.set_ylabel('Servo Value')
ax.set_xlabel('Time (s)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '03_speed_steer.png'), dpi=150)
plt.close(fig)
print("  03_speed_steer.png saved")

# ============================================================
# PLOT 4: Roll/Pitch/Yaw 분석 (3D 지형 영향)
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True)

ax = axes[0]
ax.plot(glim_odom['t'], np.degrees(glim_odom['roll']), 'r-', linewidth=0.5, label='GLIL Roll')
if len(imu_data['t']) > 0:
    # Downsample IMU for plotting (every 10th)
    idx = np.arange(0, len(imu_data['t']), 10)
    ax.plot(imu_data['t'][idx], np.degrees(imu_data['roll'][idx]), 'b-', linewidth=0.3, alpha=0.3, label='EKF/IMU Roll')
ax.set_ylabel('Roll (deg)')
ax.set_title('Vehicle Orientation (Roll/Pitch/Yaw)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(glim_odom['t'], np.degrees(glim_odom['pitch']), 'r-', linewidth=0.5, label='GLIL Pitch')
if len(imu_data['t']) > 0:
    ax.plot(imu_data['t'][idx], np.degrees(imu_data['pitch'][idx]), 'b-', linewidth=0.3, alpha=0.3, label='EKF/IMU Pitch')
ax.set_ylabel('Pitch (deg)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(glim_odom['t'], np.degrees(glim_odom['yaw']), 'r-', linewidth=0.5, label='GLIL Yaw')
ax.set_ylabel('Yaw (deg)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[3]
ax.plot(glim_odom['t'], glim_odom['z'], 'b-', linewidth=0.5, label='Z position (GLIL)')
ax.set_ylabel('Z (m)')
ax.set_xlabel('Time (s)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '04_orientation_rpy.png'), dpi=150)
plt.close(fig)
print("  04_orientation_rpy.png saved")

# ============================================================
# PLOT 5: Lateral Error vs Roll (핵심 상관관계)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

# Interpolate roll to frenet timestamps
from scipy.interpolate import interp1d
if len(glim_odom['t']) > 1 and len(car_frenet['t']) > 1:
    roll_interp = interp1d(glim_odom['t'], np.degrees(glim_odom['roll']),
                           bounds_error=False, fill_value='extrapolate')
    pitch_interp = interp1d(glim_odom['t'], np.degrees(glim_odom['pitch']),
                            bounds_error=False, fill_value='extrapolate')
    z_interp = interp1d(glim_odom['t'], glim_odom['z'],
                        bounds_error=False, fill_value='extrapolate')
    speed_interp = interp1d(car_odom['t'], car_odom['speed'],
                            bounds_error=False, fill_value='extrapolate')

    roll_at_frenet = roll_interp(car_frenet['t'])
    pitch_at_frenet = pitch_interp(car_frenet['t'])
    z_at_frenet = z_interp(car_frenet['t'])
    speed_at_frenet = speed_interp(car_frenet['t'])

    ax = axes[0]
    ax.plot(car_frenet['t'], car_frenet['d'], 'r-', linewidth=0.5, label='Lateral Error d')
    ax.set_ylabel('d (m)', color='r')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(car_frenet['t'], roll_at_frenet, 'b-', linewidth=0.5, alpha=0.5, label='Roll')
    ax2.set_ylabel('Roll (deg)', color='b')
    ax.set_title('Lateral Error vs Roll Angle')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    ax = axes[1]
    ax.plot(car_frenet['t'], car_frenet['d'], 'r-', linewidth=0.5, label='Lateral Error d')
    ax.set_ylabel('d (m)', color='r')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(car_frenet['t'], pitch_at_frenet, 'g-', linewidth=0.5, alpha=0.5, label='Pitch')
    ax2.set_ylabel('Pitch (deg)', color='g')
    ax.set_title('Lateral Error vs Pitch Angle')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    ax = axes[2]
    ax.scatter(roll_at_frenet, car_frenet['d'], c=speed_at_frenet, cmap='jet', s=1, alpha=0.3)
    ax.set_xlabel('Roll (deg)')
    ax.set_ylabel('Lateral Error d (m)')
    ax.set_title('Scatter: Roll vs Lateral Error (colored by speed)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='Speed (m/s)')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '05_lateral_vs_roll.png'), dpi=150)
plt.close(fig)
print("  05_lateral_vs_roll.png saved")

# ============================================================
# PLOT 6: L1 Distance (lookahead) 분석
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

ax = axes[0]
if len(l1_dist['t']) > 0:
    # l1_distance is geometry_msgs/Point: x=L1_dist, y=?, z=?
    ax.plot(l1_dist['t'], l1_dist['x'], 'b-', linewidth=0.5, label='L1 distance (x)')
ax.set_ylabel('L1 Distance (m)')
ax.set_title('Lookahead Distance')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(car_odom['t'], car_odom['speed'], 'b-', linewidth=0.5, label='Speed')
ax.set_ylabel('Speed (m/s)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(ctrl_input['t'], ctrl_input['steer'], 'r-', linewidth=0.5, label='Steer Cmd')
ax.set_ylabel('Steer (rad)')
ax.set_xlabel('Time (s)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '06_l1_distance.png'), dpi=150)
plt.close(fig)
print("  06_l1_distance.png saved")

# ============================================================
# PLOT 7: 고에러 구간 vs 저에러 구간 비교
# ============================================================
if len(car_frenet['t']) > 0 and len(glim_odom['t']) > 1:
    d_abs = np.abs(car_frenet['d'])

    # 고에러 구간 (|d| > 0.3m)
    high_err_mask = d_abs > 0.3
    low_err_mask = d_abs < 0.1

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Roll distribution comparison
    ax = axes[0, 0]
    if np.any(high_err_mask):
        ax.hist(roll_at_frenet[high_err_mask], bins=50, alpha=0.5, color='r', label=f'High Error (|d|>0.3m, n={np.sum(high_err_mask)})', density=True)
    if np.any(low_err_mask):
        ax.hist(roll_at_frenet[low_err_mask], bins=50, alpha=0.5, color='b', label=f'Low Error (|d|<0.1m, n={np.sum(low_err_mask)})', density=True)
    ax.set_xlabel('Roll (deg)')
    ax.set_ylabel('Density')
    ax.set_title('Roll Distribution: High vs Low Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pitch distribution
    ax = axes[0, 1]
    if np.any(high_err_mask):
        ax.hist(pitch_at_frenet[high_err_mask], bins=50, alpha=0.5, color='r', label='High Error', density=True)
    if np.any(low_err_mask):
        ax.hist(pitch_at_frenet[low_err_mask], bins=50, alpha=0.5, color='b', label='Low Error', density=True)
    ax.set_xlabel('Pitch (deg)')
    ax.set_title('Pitch Distribution: High vs Low Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Speed distribution
    ax = axes[0, 2]
    if np.any(high_err_mask):
        ax.hist(speed_at_frenet[high_err_mask], bins=50, alpha=0.5, color='r', label='High Error', density=True)
    if np.any(low_err_mask):
        ax.hist(speed_at_frenet[low_err_mask], bins=50, alpha=0.5, color='b', label='Low Error', density=True)
    ax.set_xlabel('Speed (m/s)')
    ax.set_title('Speed Distribution: High vs Low Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Z position distribution
    ax = axes[1, 0]
    if np.any(high_err_mask):
        ax.hist(z_at_frenet[high_err_mask], bins=50, alpha=0.5, color='r', label='High Error', density=True)
    if np.any(low_err_mask):
        ax.hist(z_at_frenet[low_err_mask], bins=50, alpha=0.5, color='b', label='Low Error', density=True)
    ax.set_xlabel('Z (m)')
    ax.set_title('Z Position Distribution: High vs Low Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Steer cmd interpolated at frenet time
    if len(ctrl_input['t']) > 1:
        steer_interp = interp1d(ctrl_input['t'], ctrl_input['steer'],
                                bounds_error=False, fill_value='extrapolate')
        steer_at_frenet = steer_interp(car_frenet['t'])
        ax = axes[1, 1]
        if np.any(high_err_mask):
            ax.hist(steer_at_frenet[high_err_mask], bins=50, alpha=0.5, color='r', label='High Error', density=True)
        if np.any(low_err_mask):
            ax.hist(steer_at_frenet[low_err_mask], bins=50, alpha=0.5, color='b', label='Low Error', density=True)
        ax.set_xlabel('Steer Cmd (rad)')
        ax.set_title('Steering Distribution: High vs Low Error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # d-error in s-space (where on track are the errors)
    ax = axes[1, 2]
    ax.scatter(car_frenet['s'], car_frenet['d'], c=car_frenet['t'], cmap='viridis', s=1, alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.3, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('s (m)')
    ax.set_ylabel('d (m)')
    ax.set_title('Lateral Error vs Track Position (s)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='Time (s)')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '07_high_vs_low_error.png'), dpi=150)
    plt.close(fig)
    print("  07_high_vs_low_error.png saved")

# ============================================================
# PLOT 8: XY 궤적에 에러 레벨 표시
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
if global_wpnts is not None:
    ax.plot(global_wpnts['x'], global_wpnts['y'], 'g-', linewidth=2, alpha=0.4, label='Global Path')

# Color by lateral error
if len(car_frenet['t']) > 1 and len(car_odom['t']) > 1:
    d_interp = interp1d(car_frenet['t'], car_frenet['d'], bounds_error=False, fill_value='extrapolate')
    d_at_odom = d_interp(car_odom['t'])
    sc = ax.scatter(car_odom['x'], car_odom['y'], c=np.abs(d_at_odom), cmap='RdYlGn_r',
                    s=2, vmin=0, vmax=0.6)
    plt.colorbar(sc, ax=ax, label='|d| Lateral Error (m)')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Trajectory colored by Lateral Error Magnitude')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '08_xy_error_map.png'), dpi=150)
plt.close(fig)
print("  08_xy_error_map.png saved")

# ============================================================
# PLOT 9: 코너 진입/탈출 분석 - 곡률 vs 속도 vs 에러
# ============================================================
if len(ctrl_input['t']) > 1 and len(car_frenet['t']) > 1:
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

    # Curvature from steering angle (approximate)
    steer_interp_fn = interp1d(ctrl_input['t'], ctrl_input['steer'],
                                bounds_error=False, fill_value='extrapolate')

    ax = axes[0]
    ax.plot(car_odom['t'], car_odom['speed'], 'b-', linewidth=0.5, label='Speed')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Corner Entry/Exit Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    steer_at_odom = steer_interp_fn(car_odom['t'])
    ax.plot(car_odom['t'], steer_at_odom, 'r-', linewidth=0.5, label='Steering')
    ax.set_ylabel('Steer (rad)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(car_frenet['t'], car_frenet['d'], 'm-', linewidth=0.5, label='Lateral Error d')
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_ylabel('d (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    # Rate of change of lateral error (dd/dt)
    if len(car_frenet['t']) > 2:
        dt = np.diff(car_frenet['t'])
        dd = np.diff(car_frenet['d'])
        dd_dt = dd / np.maximum(dt, 1e-6)
        ax.plot(car_frenet['t'][1:], dd_dt, 'c-', linewidth=0.3, alpha=0.5, label='dd/dt')
        # Smooth
        window = min(20, len(dd_dt))
        if window > 0:
            dd_dt_smooth = np.convolve(dd_dt, np.ones(window)/window, mode='same')
            ax.plot(car_frenet['t'][1:], dd_dt_smooth, 'c-', linewidth=1, label='dd/dt (smooth)')
    ax.set_ylabel('dd/dt (m/s)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '09_corner_analysis.png'), dpi=150)
    plt.close(fig)
    print("  09_corner_analysis.png saved")

# ============================================================
# PLOT 10: IMU 가속도 분석 (lateral force)
# ============================================================
if len(imu_data['t']) > 0:
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

    idx = np.arange(0, len(imu_data['t']), 5)  # downsample

    ax = axes[0]
    ax.plot(imu_data['t'][idx], imu_data['ay'][idx], 'r-', linewidth=0.3, alpha=0.5, label='ay (lateral)')
    # smooth
    window = min(50, len(idx))
    if window > 1:
        ay_smooth = np.convolve(imu_data['ay'][idx], np.ones(window)/window, mode='same')
        ax.plot(imu_data['t'][idx], ay_smooth, 'r-', linewidth=1, label='ay (smooth)')
    ax.set_ylabel('Lateral Accel (m/s²)')
    ax.set_title('IMU Lateral Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(imu_data['t'][idx], imu_data['ax'][idx], 'b-', linewidth=0.3, alpha=0.5, label='ax (longitudinal)')
    if window > 1:
        ax_smooth = np.convolve(imu_data['ax'][idx], np.ones(window)/window, mode='same')
        ax.plot(imu_data['t'][idx], ax_smooth, 'b-', linewidth=1, label='ax (smooth)')
    ax.set_ylabel('Longitudinal Accel (m/s²)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(imu_data['t'][idx], imu_data['wz'][idx], 'g-', linewidth=0.3, alpha=0.5, label='wz (yaw rate)')
    if window > 1:
        wz_smooth = np.convolve(imu_data['wz'][idx], np.ones(window)/window, mode='same')
        ax.plot(imu_data['t'][idx], wz_smooth, 'g-', linewidth=1, label='wz (smooth)')
    ax.set_ylabel('Yaw Rate (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '10_imu_accel.png'), dpi=150)
    plt.close(fig)
    print("  10_imu_accel.png saved")

# ============================================================
# PLOT 11: XY 궤적에 롤 각도 표시 (기울기 맵)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

if len(glim_odom['t']) > 0:
    ax = axes[0]
    if global_wpnts is not None:
        ax.plot(global_wpnts['x'], global_wpnts['y'], 'k-', linewidth=1, alpha=0.3)
    sc = ax.scatter(glim_odom['x'], glim_odom['y'], c=np.degrees(glim_odom['roll']),
                    cmap='coolwarm', s=2, vmin=-5, vmax=5)
    plt.colorbar(sc, ax=ax, label='Roll (deg)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory colored by Roll Angle')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if global_wpnts is not None:
        ax.plot(global_wpnts['x'], global_wpnts['y'], 'k-', linewidth=1, alpha=0.3)
    sc = ax.scatter(glim_odom['x'], glim_odom['y'], c=np.degrees(glim_odom['pitch']),
                    cmap='coolwarm', s=2, vmin=-10, vmax=10)
    plt.colorbar(sc, ax=ax, label='Pitch (deg)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory colored by Pitch Angle')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '11_xy_roll_pitch_map.png'), dpi=150)
plt.close(fig)
print("  11_xy_roll_pitch_map.png saved")

# ============================================================
# PLOT 12: Lateral Error 큰 구간 Zoom-in (상위 5개)
# ============================================================
if len(car_frenet['t']) > 100:
    d_abs = np.abs(car_frenet['d'])

    # Find peak error regions
    from scipy.signal import find_peaks
    peaks, props = find_peaks(d_abs, height=0.25, distance=50, prominence=0.1)

    if len(peaks) > 0:
        # Sort by height
        sorted_idx = np.argsort(props['peak_heights'])[::-1]
        n_peaks = min(6, len(peaks))

        fig, axes = plt.subplots(n_peaks, 1, figsize=(20, 5*n_peaks))
        if n_peaks == 1:
            axes = [axes]

        for i, idx in enumerate(sorted_idx[:n_peaks]):
            peak_t = car_frenet['t'][peaks[idx]]
            t_start = max(0, peak_t - 5)
            t_end = peak_t + 5

            ax = axes[i]

            # Frenet d
            mask_f = (car_frenet['t'] >= t_start) & (car_frenet['t'] <= t_end)
            ax.plot(car_frenet['t'][mask_f], car_frenet['d'][mask_f], 'r-', linewidth=1.5, label='d (lateral)')

            # Speed on twin axis
            ax2 = ax.twinx()
            mask_o = (car_odom['t'] >= t_start) & (car_odom['t'] <= t_end)
            ax2.plot(car_odom['t'][mask_o], car_odom['speed'][mask_o], 'b-', linewidth=1, alpha=0.5, label='Speed')
            ax2.set_ylabel('Speed (m/s)', color='b')

            # Steer on the first axis
            mask_c = (ctrl_input['t'] >= t_start) & (ctrl_input['t'] <= t_end)
            ax.plot(ctrl_input['t'][mask_c], ctrl_input['steer'][mask_c], 'g-', linewidth=1, alpha=0.7, label='Steer')

            # Roll
            mask_g = (glim_odom['t'] >= t_start) & (glim_odom['t'] <= t_end)
            if np.any(mask_g):
                ax.plot(glim_odom['t'][mask_g], np.degrees(glim_odom['roll'][mask_g])/10,
                       'orange', linewidth=1, alpha=0.7, label='Roll/10 (deg)')

            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_ylabel('d (m) / Steer (rad)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Peak Error #{i+1}: t={peak_t:.1f}s, d={car_frenet["d"][peaks[idx]]:.3f}m, '
                        f's={car_frenet["s"][peaks[idx]]:.1f}m')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, '12_peak_error_zoomin.png'), dpi=150)
        plt.close(fig)
        print("  12_peak_error_zoomin.png saved")

# ============================================================
# PLOT 13: XY에서 에러 큰 구간 하이라이트
# ============================================================
if len(car_frenet['t']) > 1 and len(car_odom['t']) > 1 and global_wpnts is not None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.plot(global_wpnts['x'], global_wpnts['y'], 'g-', linewidth=2, alpha=0.4, label='Global Path')

    d_interp_fn = interp1d(car_frenet['t'], car_frenet['d'], bounds_error=False, fill_value='extrapolate')
    d_at_odom = d_interp_fn(car_odom['t'])

    # Normal trajectory
    low_mask = np.abs(d_at_odom) < 0.3
    high_mask = np.abs(d_at_odom) >= 0.3

    ax.scatter(car_odom['x'][low_mask], car_odom['y'][low_mask], c='blue', s=1, alpha=0.3, label='|d|<0.3m')
    ax.scatter(car_odom['x'][high_mask], car_odom['y'][high_mask], c='red', s=5, alpha=0.8, label='|d|>=0.3m')

    # Annotate worst spots
    if len(peaks) > 0:
        s_at_peaks = car_frenet['s'][peaks[sorted_idx[:min(5, len(sorted_idx))]]]
        d_at_peaks = car_frenet['d'][peaks[sorted_idx[:min(5, len(sorted_idx))]]]
        t_at_peaks = car_frenet['t'][peaks[sorted_idx[:min(5, len(sorted_idx))]]]

        x_interp = interp1d(car_odom['t'], car_odom['x'], bounds_error=False, fill_value='extrapolate')
        y_interp = interp1d(car_odom['t'], car_odom['y'], bounds_error=False, fill_value='extrapolate')

        for j, (tp, sp, dp) in enumerate(zip(t_at_peaks, s_at_peaks, d_at_peaks)):
            xp = x_interp(tp)
            yp = y_interp(tp)
            ax.annotate(f'#{j+1}\nd={dp:.2f}m\ns={sp:.0f}m', (xp, yp),
                       fontsize=8, color='red', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red'),
                       xytext=(xp+0.5, yp+0.5))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory with High Error Regions Highlighted')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '13_xy_error_highlight.png'), dpi=150)
    plt.close(fig)
    print("  13_xy_error_highlight.png saved")

# ============================================================
# PLOT 14: 속도 vs 에러 시간지연 (코너 탈출 후 오버슛 분석)
# ============================================================
if len(car_frenet['t']) > 10 and len(car_odom['t']) > 10:
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    # Cross-correlation between |steering| and |d| with time lag
    from scipy.signal import correlate

    # Resample to uniform time
    t_uniform = np.arange(car_frenet['t'][0], car_frenet['t'][-1], 0.02)
    d_uniform = interp1d(car_frenet['t'], np.abs(car_frenet['d']), bounds_error=False, fill_value=0)(t_uniform)

    if len(ctrl_input['t']) > 1:
        steer_uniform = interp1d(ctrl_input['t'], np.abs(ctrl_input['steer']),
                                  bounds_error=False, fill_value=0)(t_uniform)
        speed_uniform = interp1d(car_odom['t'], car_odom['speed'],
                                  bounds_error=False, fill_value=0)(t_uniform)

        # Cross-correlation steer vs d
        corr = correlate(d_uniform - np.mean(d_uniform), steer_uniform - np.mean(steer_uniform), mode='full')
        lags = np.arange(-len(d_uniform)+1, len(d_uniform)) * 0.02
        corr = corr / np.max(np.abs(corr))

        ax = axes[0]
        mask = np.abs(lags) < 3.0  # ±3s
        ax.plot(lags[mask], corr[mask], 'b-')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        peak_lag = lags[mask][np.argmax(corr[mask])]
        ax.axvline(x=peak_lag, color='r', linestyle='--', alpha=0.5, label=f'Peak lag={peak_lag:.2f}s')
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Normalized Cross-Correlation')
        ax.set_title('Cross-correlation: |Steering| vs |Lateral Error|')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cross-correlation speed vs d
        corr2 = correlate(d_uniform - np.mean(d_uniform), speed_uniform - np.mean(speed_uniform), mode='full')
        corr2 = corr2 / np.max(np.abs(corr2))

        ax = axes[1]
        ax.plot(lags[mask], corr2[mask], 'r-')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        peak_lag2 = lags[mask][np.argmax(corr2[mask])]
        ax.axvline(x=peak_lag2, color='r', linestyle='--', alpha=0.5, label=f'Peak lag={peak_lag2:.2f}s')
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Normalized Cross-Correlation')
        ax.set_title('Cross-correlation: Speed vs |Lateral Error|')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '14_cross_correlation.png'), dpi=150)
    plt.close(fig)
    print("  14_cross_correlation.png saved")

# ============================================================
# PLOT 15: s 기반 에러 맵 (트랙 위치별 에러 패턴)
# ============================================================
if len(car_frenet['t']) > 0 and global_wpnts is not None:
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

    ax = axes[0]
    ax.plot(car_frenet['s'], car_frenet['d'], 'r.', markersize=0.5, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_ylabel('d (m)')
    ax.set_title('Track Position (s) based Analysis')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(global_wpnts['s'], global_wpnts['vx'], 'g-', linewidth=1, label='Planned Speed')
    # Interpolate actual speed to s
    if len(car_odom['t']) > 1:
        s_interp = interp1d(car_frenet['t'], car_frenet['s'], bounds_error=False, fill_value='extrapolate')
        s_at_odom = s_interp(car_odom['t'])
        ax.scatter(s_at_odom, car_odom['speed'], c='b', s=0.5, alpha=0.3, label='Actual Speed')
    ax.set_ylabel('Speed (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # Roll vs s
    if len(glim_odom['t']) > 1:
        s_at_glim = s_interp(glim_odom['t'])
        ax.scatter(s_at_glim, np.degrees(glim_odom['roll']), c='r', s=0.5, alpha=0.3, label='Roll')
        ax.scatter(s_at_glim, np.degrees(glim_odom['pitch']), c='b', s=0.5, alpha=0.3, label='Pitch')
    ax.set_ylabel('Angle (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    # Z vs s
    if len(glim_odom['t']) > 1:
        ax.scatter(s_at_glim, glim_odom['z'], c='purple', s=0.5, alpha=0.3, label='Actual Z')
    ax.plot(global_wpnts['s'], global_wpnts['z'], 'g-', linewidth=1, label='Waypoint Z')
    ax.set_ylabel('Z (m)')
    ax.set_xlabel('s (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '15_s_based_analysis.png'), dpi=150)
    plt.close(fig)
    print("  15_s_based_analysis.png saved")

# ============================================================
# PLOT 16: Lateral Velocity vs Steering (제어 효율성)
# ============================================================
if len(car_frenet['t']) > 1 and len(ctrl_input['t']) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    steer_at_frenet = interp1d(ctrl_input['t'], ctrl_input['steer'],
                               bounds_error=False, fill_value='extrapolate')(car_frenet['t'])

    # Steer vs vd
    ax = axes[0, 0]
    ax.scatter(steer_at_frenet, car_frenet['vd'], c=speed_at_frenet, cmap='jet', s=1, alpha=0.3)
    ax.set_xlabel('Steer Cmd (rad)')
    ax.set_ylabel('Lateral Velocity vd (m/s)')
    ax.set_title('Steering Effectiveness (colored by speed)')
    ax.grid(True, alpha=0.3)

    # Roll vs vd
    ax = axes[0, 1]
    ax.scatter(roll_at_frenet, car_frenet['vd'], c=speed_at_frenet, cmap='jet', s=1, alpha=0.3)
    ax.set_xlabel('Roll (deg)')
    ax.set_ylabel('Lateral Velocity vd (m/s)')
    ax.set_title('Roll Effect on Lateral Velocity')
    ax.grid(True, alpha=0.3)

    # d vs steer (is controller reacting correctly?)
    ax = axes[1, 0]
    ax.scatter(car_frenet['d'], steer_at_frenet, c=speed_at_frenet, cmap='jet', s=1, alpha=0.3)
    ax.set_xlabel('Lateral Error d (m)')
    ax.set_ylabel('Steer Cmd (rad)')
    ax.set_title('Controller Response: d vs Steer')
    ax.grid(True, alpha=0.3)

    # Speed vs |d|
    ax = axes[1, 1]
    ax.scatter(speed_at_frenet, np.abs(car_frenet['d']), c=np.abs(roll_at_frenet),
               cmap='Reds', s=1, alpha=0.3)
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('|d| (m)')
    ax.set_title('Speed vs Error (colored by |Roll|)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='|Roll| (deg)')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '16_ctrl_effectiveness.png'), dpi=150)
    plt.close(fig)
    print("  16_ctrl_effectiveness.png saved")

print("\n=== All tracking analysis plots saved to", OUT_DIR, "===")
