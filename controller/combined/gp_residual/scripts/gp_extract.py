#!/usr/bin/env python3
"""
GP Data Extractor — rosbag에서 GP 학습용 데이터를 CSV로 추출.

사용법:
    python3 gp_extract.py --bag /path/to/recording.bag --output ../data/run01.csv

필요 토픽:
    /car_state/odom        — 속도 (twist.linear.x)
    /car_state/pose        — 위치, yaw
    /imu/data              — yaw_rate (-angular_velocity.z), ax (-linear_acceleration.y)
    /vesc/.../nav_1        — steering command
    /behavior_strategy     — waypoint array (kappa)
    /car_state/odom_frenet — frenet s, d (lateral error)

출력 CSV columns:
    t, v, delta_cmd, kappa, yaw_rate, ax, lat_error, s_position
"""

import argparse
import csv
import os
import sys
import numpy as np

try:
    import rosbag
except ImportError:
    print("ERROR: rosbag not found. Run inside Docker or source ROS setup.")
    sys.exit(1)

from tf.transformations import euler_from_quaternion


def extract(bag_path, output_path, drive_topic=None):
    bag = rosbag.Bag(bag_path, 'r')
    info = bag.get_type_and_topic_info()
    topics = list(info.topics.keys())

    # auto-detect drive topic
    if drive_topic is None:
        candidates = [t for t in topics if 'ackermann_cmd_mux' in t and 'nav' in t]
        if candidates:
            drive_topic = candidates[0]
        else:
            drive_topic = '/vesc/high_level/ackermann_cmd_mux/input/nav_1'
    print(f"Drive topic: {drive_topic}")

    needed = ['/car_state/odom', '/car_state/pose', '/imu/data',
              drive_topic, '/behavior_strategy', '/car_state/odom_frenet']
    for t in needed:
        if t not in topics:
            print(f"  WARNING: {t} not found in bag")

    latest = {
        'v': 0.0, 'yaw': 0.0, 'pos_x': 0.0, 'pos_y': 0.0,
        'yaw_rate': 0.0, 'ax': 0.0, 'delta_cmd': 0.0,
        'kappa': 0.0, 's': 0.0, 'lat_error': 0.0,
        'wpnts': None,
    }

    rows = []
    read_topics = ['/car_state/odom', '/car_state/pose', '/imu/data',
                   drive_topic, '/behavior_strategy', '/car_state/odom_frenet']

    for topic, msg, t in bag.read_messages(topics=read_topics):
        ts = t.to_sec()

        if topic == '/car_state/odom':
            latest['v'] = msg.twist.twist.linear.x

        elif topic == '/car_state/pose':
            q = msg.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            latest['yaw'] = yaw
            latest['pos_x'] = msg.pose.position.x
            latest['pos_y'] = msg.pose.position.y

        elif topic == '/imu/data':
            latest['yaw_rate'] = -msg.angular_velocity.z   # VESC 90deg rotation
            latest['ax'] = -msg.linear_acceleration.y      # VESC 90deg: -y = longitudinal

        elif topic == drive_topic:
            latest['delta_cmd'] = msg.drive.steering_angle

        elif topic == '/behavior_strategy':
            wpnts = np.array([[wp.x_m, wp.y_m, wp.kappa_radpm]
                              for wp in msg.local_wpnts])
            latest['wpnts'] = wpnts

        elif topic == '/car_state/odom_frenet':
            latest['s'] = msg.pose.pose.position.x
            latest['lat_error'] = msg.pose.pose.position.y

        # sample at every IMU message (~100Hz, downsampled later)
        if topic == '/imu/data' and latest['v'] > 0.3:
            kappa = 0.0
            if latest['wpnts'] is not None and len(latest['wpnts']) > 0:
                pos = np.array([latest['pos_x'], latest['pos_y']])
                dists = np.linalg.norm(latest['wpnts'][:, :2] - pos, axis=1)
                idx = np.argmin(dists)
                kappa = latest['wpnts'][idx, 2]

            rows.append([
                f"{ts:.4f}",
                f"{latest['v']:.4f}",
                f"{latest['delta_cmd']:.6f}",
                f"{kappa:.6f}",
                f"{latest['yaw_rate']:.6f}",
                f"{latest['ax']:.6f}",
                f"{latest['lat_error']:.6f}",
                f"{latest['s']:.4f}",
            ])

    bag.close()

    # downsample to ~40Hz
    if len(rows) > 0:
        raw_hz = len(rows) / (float(rows[-1][0]) - float(rows[0][0]) + 1e-6)
        skip = max(1, int(raw_hz / 40))
        rows = rows[::skip]

    # write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'v', 'delta_cmd', 'kappa', 'yaw_rate', 'ax',
                          'lat_error', 's_position'])
        writer.writerows(rows)

    print(f"Extracted {len(rows)} samples → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Extract GP data from rosbag')
    parser.add_argument('--bag', required=True, help='Path to rosbag file')
    parser.add_argument('--output', default=None, help='Output CSV path')
    parser.add_argument('--drive-topic', default=None, help='Drive command topic')
    args = parser.parse_args()

    if args.output is None:
        bag_name = os.path.splitext(os.path.basename(args.bag))[0]
        args.output = os.path.join(os.path.dirname(__file__), '..', 'data',
                                   f'{bag_name}.csv')

    extract(args.bag, args.output, args.drive_topic)


if __name__ == '__main__':
    main()
