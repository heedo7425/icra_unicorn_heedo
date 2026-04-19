#!/usr/bin/env python3
"""
Spawn the sim car on the first global waypoint.

f1tenth_simulator subscribes to /initialpose (PoseWithCovarianceStamped,
same topic as RViz's "2D Pose Estimate"). We wait for /global_waypoints,
take wpnt[0], and publish once so the car snaps onto the raceline before
MPC tries to compute Frenet state.

One-shot node: publishes a few times (subscriber may not be ready on the
very first publish), then exits.
"""

from __future__ import annotations

import math
import time

import rospy
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import PoseWithCovarianceStamped


def _yaw_to_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def main() -> None:
    rospy.init_node("spawn_on_waypoint", anonymous=True)
    topic = rospy.get_param("~initialpose_topic", "/initialpose")
    wpnt_idx = int(rospy.get_param("~wpnt_idx", 0))
    n_publishes = int(rospy.get_param("~n_publishes", 60))
    publish_hz = float(rospy.get_param("~publish_hz", 10.0))
    supervise = bool(rospy.get_param("~supervise", False))
    stuck_sec = float(rospy.get_param("~stuck_sec", 2.0))

    pub = rospy.Publisher(topic, PoseWithCovarianceStamped, queue_size=1, latch=True)

    rospy.loginfo(f"[spawn_on_waypoint] waiting for /global_waypoints...")
    wp_msg: WpntArray = rospy.wait_for_message("/global_waypoints", WpntArray, timeout=30.0)
    if wpnt_idx >= len(wp_msg.wpnts):
        wpnt_idx = 0
    wp = wp_msg.wpnts[wpnt_idx]

    # Compute tangent direction directly from wp → wp_next (wrap-safe).
    # This is independent of psi_rad so we don't inherit any upstream bug.
    look_ahead = int(rospy.get_param("~look_ahead_wpnts", 5))
    wp_next = wp_msg.wpnts[(wpnt_idx + look_ahead) % len(wp_msg.wpnts)]
    tangent_yaw = math.atan2(wp_next.y_m - wp.y_m, wp_next.x_m - wp.x_m)
    rospy.loginfo(
        f"[spawn_on_waypoint] wpnt[{wpnt_idx}] psi_rad={wp.psi_rad:.3f} "
        f"vs tangent(→wp[{wpnt_idx+look_ahead}])={tangent_yaw:.3f} — using tangent"
    )

    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"
    msg.pose.pose.position.x = float(wp.x_m)
    msg.pose.pose.position.y = float(wp.y_m)
    msg.pose.pose.position.z = 0.0
    qx, qy, qz, qw = _yaw_to_quat(tangent_yaw)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    # Small covariance (sim is deterministic).
    msg.pose.covariance = [0.0] * 36

    rospy.loginfo(
        f"[spawn_on_waypoint] publishing initialpose "
        f"(x={wp.x_m:.2f} y={wp.y_m:.2f} yaw={wp.psi_rad:.2f}) to {topic} "
        f"×{n_publishes}"
    )
    rate = rospy.Rate(publish_hz)
    for _ in range(n_publishes):
        if rospy.is_shutdown():
            break
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()

    rospy.loginfo(f"[spawn_on_waypoint] initial spawn done.")
    if not supervise:
        return

    # Supervisor mode: watch /car_state/pose; if car stuck (Δpose < 0.1m over
    # stuck_sec) OR MPC self-reports OFF-TRACK, republish initialpose to snap
    # back to wpnt[0].
    from geometry_msgs.msg import PoseStamped
    from rosgraph_msgs.msg import Log
    rospy.loginfo(f"[spawn_on_waypoint] supervisor armed.")
    last_pose = [None, None, rospy.Time.now().to_sec()]
    off_track_seen = [False]

    def pose_cb(pm: PoseStamped):
        last_pose[0], last_pose[1], last_pose[2] = pm.pose.position.x, pm.pose.position.y, rospy.Time.now().to_sec()

    def rosout_cb(r: Log):
        if "OFF-TRACK" in r.msg:
            off_track_seen[0] = True

    rospy.Subscriber("/car_state/pose", PoseStamped, pose_cb)
    rospy.Subscriber("/rosout", Log, rosout_cb)

    last_sample_t = rospy.Time.now().to_sec()
    last_sample_x = None
    last_sample_y = None
    r = rospy.Rate(2)
    while not rospy.is_shutdown():
        r.sleep()
        now = rospy.Time.now().to_sec()
        if last_pose[0] is None:
            continue

        # Sample pose every stuck_sec
        if now - last_sample_t >= stuck_sec:
            stuck = False
            if last_sample_x is not None:
                ddx = last_pose[0] - last_sample_x
                ddy = last_pose[1] - last_sample_y
                moved = (ddx * ddx + ddy * ddy) ** 0.5
                if moved < 0.2:
                    stuck = True

            if stuck or off_track_seen[0]:
                rospy.logwarn(
                    f"[spawn_on_waypoint] RESPAWN trigger "
                    f"(stuck={stuck}, off_track={off_track_seen[0]}) → re-publishing initialpose"
                )
                # Re-publish latch initialpose 10× at 10Hz
                for _ in range(10):
                    msg.header.stamp = rospy.Time.now()
                    pub.publish(msg)
                    rospy.sleep(0.1)
                off_track_seen[0] = False
                # reset sampling
                last_sample_t = rospy.Time.now().to_sec()
                last_sample_x = last_pose[0]
                last_sample_y = last_pose[1]
                continue

            last_sample_t = now
            last_sample_x = last_pose[0]
            last_sample_y = last_pose[1]


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
