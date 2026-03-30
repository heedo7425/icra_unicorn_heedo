#!/usr/bin/env python3
### HJ : Gazebo static obstacle → /tracking/obstacles publisher
### Bypasses detect + multi_tracking pipeline for Gazebo environments

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64MultiArray
from f110_msgs.msg import ObstacleArray, Obstacle, WpntArray
from visualization_msgs.msg import Marker, MarkerArray
from frenet_converter.frenet_converter import FrenetConverter


class GazeboStaticObstaclePublisher:
    def __init__(self):
        rospy.init_node("gazebo_static_obstacle_publisher", anonymous=False)
        self.rate = rospy.Rate(rospy.get_param("~rate", 10))

        # configurable topics
        self.in_poses_topic = rospy.get_param("~in_poses_topic", "/gazebo/static_obstacles/poses")
        self.in_radii_topic = rospy.get_param("~in_radii_topic", "/gazebo/static_obstacles/radii")
        self.out_obstacles_topic = rospy.get_param("~out_obstacles_topic", "/tracking/obstacles")
        self.out_markers_topic = rospy.get_param("~out_markers_topic", "/gazebo_static_obstacle_markers")

        # state
        self.poses = None
        self.radii = None
        self.converter = None
        self.track_length = None

        # wait for global waypoints to build FrenetConverter
        rospy.loginfo("[GazeboStaticObsPub] Waiting for /global_waypoints...")
        waypoints_msg = rospy.wait_for_message("/global_waypoints", WpntArray)
        wpts = np.array([[w.x_m, w.y_m, w.z_m] for w in waypoints_msg.wpnts])
        self.converter = FrenetConverter(wpts[:, 0], wpts[:, 1], wpts[:, 2])
        self.track_length = rospy.get_param("/global_republisher/track_length")
        rospy.logwarn(f"[GazeboStaticObsPub] FrenetConverter ready, track_length={self.track_length:.2f}m")

        # publishers
        self.obstacle_pub = rospy.Publisher(self.out_obstacles_topic, ObstacleArray, queue_size=10)
        self.marker_pub = rospy.Publisher(self.out_markers_topic, MarkerArray, queue_size=10)

        # subscribers
        rospy.Subscriber(self.in_poses_topic, PoseArray, self.poses_cb)
        rospy.Subscriber(self.in_radii_topic, Float64MultiArray, self.radii_cb)

        rospy.loginfo("[GazeboStaticObsPub] Initialized, waiting for obstacle data...")

    def poses_cb(self, msg):
        self.poses = msg.poses

    def radii_cb(self, msg):
        self.radii = msg.data

    def run(self):
        while not rospy.is_shutdown():
            if self.poses is not None and self.radii is not None:
                self.publish_obstacles()
            self.rate.sleep()

    def publish_obstacles(self):
        obstacle_msg = ObstacleArray()
        obstacle_msg.header.stamp = rospy.Time.now()
        obstacle_msg.header.frame_id = "map"

        marker_msg = MarkerArray()

        n_obs = min(len(self.poses), len(self.radii))

        for i in range(n_obs):
            pose = self.poses[i]
            radius = self.radii[i]

            x = pose.position.x
            y = pose.position.y
            z = pose.position.z

            # frenet conversion (3D)
            try:
                s_d = self.converter.get_frenet_3d(
                    np.array([x]), np.array([y]), np.array([z]))
                s_center = float(s_d[0][0])
                d_center = float(s_d[1][0])
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[GazeboStaticObsPub] Frenet conversion failed: {e}")
                continue

            # build Obstacle message
            obs = Obstacle()
            obs.id = i
            obs.x_m = x
            obs.y_m = y
            obs.z_m = z
            obs.s_center = s_center
            obs.d_center = d_center
            obs.s_start = (s_center - radius) % self.track_length
            obs.s_end = (s_center + radius) % self.track_length
            obs.d_right = d_center - radius
            obs.d_left = d_center + radius
            obs.size = radius * 2
            obs.vs = 0.0
            obs.vd = 0.0
            obs.is_static = True
            obs.is_visible = True
            obs.is_actually_a_gap = False
            obs.sector_id = -1
            obs.in_static_obs_sector = False

            obstacle_msg.obstacles.append(obs)

            # visualization marker
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = rospy.Time.now()
            mrk.ns = "gazebo_static_obs"
            mrk.id = i
            mrk.type = Marker.CYLINDER
            mrk.action = Marker.ADD
            mrk.pose.position.x = x
            mrk.pose.position.y = y
            mrk.pose.position.z = z
            mrk.pose.orientation.w = 1.0
            mrk.scale.x = radius * 2
            mrk.scale.y = radius * 2
            mrk.scale.z = 0.3
            mrk.color.r = 1.0
            mrk.color.g = 0.3
            mrk.color.b = 0.0
            mrk.color.a = 0.8
            marker_msg.markers.append(mrk)

        self.obstacle_pub.publish(obstacle_msg)
        self.marker_pub.publish(marker_msg)

    def shutdown(self):
        rospy.loginfo("[GazeboStaticObsPub] Shutdown, clearing obstacles")
        self.obstacle_pub.publish(ObstacleArray())


if __name__ == "__main__":
    node = GazeboStaticObstaclePublisher()
    rospy.on_shutdown(node.shutdown)
    node.run()
