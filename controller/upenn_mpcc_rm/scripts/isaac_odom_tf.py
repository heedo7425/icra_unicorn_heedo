#!/usr/bin/env python3
"""Isaac Sim → TF adapter.

The Isaac Sim scene publishes /car_state/odom (nav_msgs/Odometry) through
rosbridge. For RViz visualization we need the matching TF:
    map → base_link  (from odom pose)
    base_link → livox_frame (static, from URDF)

Usage (Auto-included by upenn_mpc_isaac.launch):
    rosrun controller isaac_odom_tf.py
"""
import math

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


class IsaacOdomTF:
    def __init__(self):
        rospy.init_node("isaac_odom_tf")
        self.parent_frame = rospy.get_param("~parent_frame", "map")
        self.child_frame  = rospy.get_param("~child_frame", "base_link")

        self.br = tf2_ros.TransformBroadcaster()
        self.static_br = tf2_ros.StaticTransformBroadcaster()

        # Static TF from base_link → livox_frame (lidar mount).
        # Values mirror Isaac scene mount_offset=(0.12, 0, 0.15).
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.child_frame
        t.child_frame_id = "livox_frame"
        t.transform.translation.x = 0.12
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.15
        t.transform.rotation.w = 1.0
        self.static_br.sendTransform(t)

        self.sub = rospy.Subscriber("/car_state/odom", Odometry,
                                     self._cb, queue_size=50)
        rospy.loginfo("[isaac_odom_tf] broadcasting TF %s -> %s (and livox_frame)",
                      self.parent_frame, self.child_frame)

    def _cb(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        t.header.frame_id = self.parent_frame
        t.child_frame_id  = self.child_frame
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        # Guard against zero-quaternion (can happen before first sample).
        q = t.transform.rotation
        mag = math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
        if mag < 1e-6:
            t.transform.rotation.w = 1.0
        self.br.sendTransform(t)


def main():
    IsaacOdomTF()
    rospy.spin()


if __name__ == "__main__":
    main()
