#!/usr/bin/env python3
"""
MPC sign convention experimental check.

Steps:
  1. Read wpnt[0] (known raceline position + heading)
  2. Publish /initialpose with position = wpnt[0].xy offset LEFT by +0.5m
     (left = rotate tangent-normal in the "positive n" convention)
  3. Wait for simulator to apply pose, MPC to see new state
  4. Print observed vs expected:
     - /car_state/odom_frenet d (from the repo's Frenet converter — ground truth)
     - MPC's internal x0[1] (from log — our computation)
     - /vesc/.../nav_1 steering_angle (MPC's reaction)

If n_local sign is correct:
  - d should be ≈ +0.5 (left of raceline)
  - MPC x0[1] should be ≈ +0.5
  - MPC steer should be NEGATIVE (turn right → reduce n)
If sign is flipped, signs will disagree.
"""

from __future__ import annotations

import math
import time

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def _yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def main():
    rospy.init_node("mpc_sign_test", anonymous=True)

    rospy.loginfo("[sign_test] waiting for /global_waypoints ...")
    wp = rospy.wait_for_message("/global_waypoints", WpntArray, timeout=30)
    w = wp.wpnts[0]
    # Positive n = left of tangent. Left-normal = (-sin psi, cos psi).
    offset = 0.5  # left
    n_expected = offset
    x_off = w.x_m + (-math.sin(w.psi_rad)) * offset
    y_off = w.y_m + math.cos(w.psi_rad) * offset
    rospy.loginfo(
        f"[sign_test] wpnt[0]=({w.x_m:.2f},{w.y_m:.2f}) psi={w.psi_rad:.3f} "
        f"→ offset LEFT +{offset}m → publish ({x_off:.2f},{y_off:.2f})"
    )

    pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1, latch=True)
    ipose = PoseWithCovarianceStamped()
    ipose.header.frame_id = "map"
    ipose.pose.pose.position.x = x_off
    ipose.pose.pose.position.y = y_off
    qx, qy, qz, qw = _yaw_to_quat(w.psi_rad)
    ipose.pose.pose.orientation.x = qx
    ipose.pose.pose.orientation.y = qy
    ipose.pose.pose.orientation.z = qz
    ipose.pose.pose.orientation.w = qw

    # Publish multiple times to win over spawn_supervisor & ensure sim applies
    for _ in range(20):
        ipose.header.stamp = rospy.Time.now()
        pub.publish(ipose)
        rospy.sleep(0.1)

    rospy.loginfo("[sign_test] pose offset applied. Sampling 5s...")

    # Sample Frenet d (ground truth) and MPC steer output
    frenet_samples = []
    drive_samples = []

    def fcb(msg):
        frenet_samples.append((rospy.Time.now().to_sec(), msg.pose.pose.position.y))  # y = d

    def dcb(msg):
        drive_samples.append((rospy.Time.now().to_sec(), msg.drive.steering_angle, msg.drive.speed))

    rospy.Subscriber("/car_state/odom_frenet", Odometry, fcb)
    rospy.Subscriber("/vesc/high_level/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, dcb)

    t0 = rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec() - t0 < 5 and not rospy.is_shutdown():
        rospy.sleep(0.1)

    if not frenet_samples or not drive_samples:
        rospy.logwarn("[sign_test] no samples received")
        return

    # Report
    ds = [s[1] for s in frenet_samples]
    steers = [s[1] for s in drive_samples]
    speeds = [s[2] for s in drive_samples]
    d_avg = sum(ds) / len(ds)
    steer_avg = sum(steers) / len(steers)
    speed_avg = sum(speeds) / len(speeds)

    rospy.loginfo("=" * 60)
    rospy.loginfo(f"[sign_test] EXPECTED:  n ≈ +{n_expected}  →  MPC should steer < 0 (right)")
    rospy.loginfo(
        f"[sign_test] OBSERVED:  d(Frenet)={d_avg:+.3f}  "
        f"MPC steer={steer_avg:+.4f}  speed={speed_avg:+.3f}"
    )
    d_ok = d_avg > 0.2
    steer_ok = steer_avg < -0.01
    rospy.loginfo(
        f"[sign_test] d sign matches expected: {d_ok}   steer sign correct: {steer_ok}"
    )
    if d_ok and steer_ok:
        rospy.loginfo("[sign_test] ✅ sign convention CONSISTENT")
    else:
        rospy.logwarn("[sign_test] ❌ sign convention MISMATCH")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
