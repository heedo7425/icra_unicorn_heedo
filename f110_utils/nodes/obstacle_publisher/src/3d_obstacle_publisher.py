#!/usr/bin/env python3

### HJ : 3D port of obstacle_publisher.py
###      - Glob2FrenetArr now requires (x, y, z); pass wpnt.z_m
###      - Frenet2GlobArr now returns z; fill Obstacle.z_m and OppWpnt height
###      - Marker visualization uses obstacle.z_m so it sits on the track surface
###      - Node name / topics unchanged so it can drop-in replace the 2D node
### HJ : end

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from f110_msgs.msg import ObstacleArray, Obstacle, WpntArray, Wpnt, OpponentTrajectory, OppWpnt
from visualization_msgs.msg import Marker, MarkerArray
from frenet_conversion.srv import Glob2FrenetArr, Frenet2GlobArr
from nav_msgs.msg import Odometry


class ObstaclePublisher3D:
    """3D-aware dynamic obstacle publisher.

    Same behavior as obstacle_publisher.py but exercises the updated 3D
    Frenet service signatures and propagates z_m through the obstacle/marker
    pipeline so visualization sits on the actual track surface.
    """
    def __init__(self):
        looprate = 50
        self.rate = rospy.Rate(looprate)
        self.looptime = 1/looprate

        self.dynamic_obstacle = self.init_dynamic_obstacle()
        self.obj_len = 0.5

        # Parameters (use ~private namespace via param server, mirroring 2D node)
        self.speed_scaler = rospy.get_param("obstacle_publisher/speed_scaler", 1)
        self.constant = rospy.get_param("obstacle_publisher/constant_speed", False)

        self.waypoints_type = rospy.get_param("/obstacle_publisher/trajectory", "min_curv")
        if self.waypoints_type == "min_curv":
            self.waypoints_topic = "/global_waypoints"
        elif self.waypoints_type == "shortest_path":
            self.waypoints_topic = "/global_waypoints/shortest_path"
        elif self.waypoints_type == "centerline":
            self.waypoints_topic = "/centerline_waypoints"
        elif self.waypoints_type == "updated":
            self.waypoints_topic = "/global_waypoints_updated"
            print("Using updated waypoints")
        elif self.waypoints_type == "min_time":
            raise NotImplementedError(
                "LTO Trajectory is not currently implemented. Choose another trajectory type."
            )
        else:
            raise ValueError(
                f"Waypoints of type {self.waypoints_type} are not supported."
            )

        self.starting_s = rospy.get_param("/obstacle_publisher/start_s", 0)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.odom_cb)
        self.car_odom = Odometry()

        self.obstacle_pub = rospy.Publisher("/tracking/obstacles", ObstacleArray, queue_size=10)
        self.obstacle_mrk_pub = rospy.Publisher("/dummy_obstacle_markers", MarkerArray, queue_size=10)
        self.opponent_traj_pub = rospy.Publisher("/opponent_waypoints", OpponentTrajectory, queue_size=10)

        rospy.wait_for_service("convert_glob2frenet_service")
        # BUGFIX: persistent=True reuses one TCP connection across calls.
        # Without it rospy opens/closes a socket per call -> with a non-loopback
        # ROS_MASTER_URI the kernel can't recycle the (local_ip, port) tuple
        # (TIME_WAIT ~60s) and ephemeral ports for that destination exhaust
        # after sustained high-rate calls, killing the node with
        # OSError 99 / ServiceException("unable to contact master").
        self.glob2frenet = rospy.ServiceProxy(
            "convert_glob2frenetarr_service", Glob2FrenetArr, persistent=True)
        self.frenet2glob = rospy.ServiceProxy(
            "convert_frenet2globarr_service", Frenet2GlobArr, persistent=True)
        self.mincurv_wpnts = None

    def init_dynamic_obstacle(self) -> Obstacle:
        dynamic_obstacle = Obstacle()
        dynamic_obstacle.id = 1
        dynamic_obstacle.d_right = -0.1
        dynamic_obstacle.d_left = 0.1
        dynamic_obstacle.is_actually_a_gap = False
        return dynamic_obstacle

    ### CALLBACKS ###
    def wpnts_cb(self, data: WpntArray):
        wpnts = data.wpnts[:-1]
        max_s = wpnts[-1].s_m
        return wpnts, max_s

    def odom_cb(self, data: Odometry):
        self.car_odom = data

    ### HELPERS ###
    def publish_obstacle_cartesian(self, obstacles):
        """Visualizes obstacles in cartesian frame (3D-aware)."""
        obs_markers = MarkerArray()
        for obs in obstacles:
            obs_marker = Marker(header=rospy.Header(frame_id="map"), id=obs.id, type=Marker.SPHERE)
            obs_marker.scale.x = 0.5
            obs_marker.scale.y = 0.5
            obs_marker.scale.z = 0.5
            obs_marker.color.a = 0.5
            obs_marker.color.b = 0.5
            obs_marker.color.r = 0.5

            obs_marker.pose.position.x = obs.x_m
            obs_marker.pose.position.y = obs.y_m
            ### HJ : use z_m for 3D marker visualization
            obs_marker.pose.position.z = obs.z_m
            ### HJ : end
            obs_marker.pose.orientation.w = 1
            obs_markers.markers.append(obs_marker)

        self.obstacle_mrk_pub.publish(obs_markers)

    def shutdown(self):
        rospy.loginfo("BEEP BOOP DUMMY OD SHUTDOWN")
        self.obstacle_pub.publish(ObstacleArray())

    ### MAIN ###
    def ros_loop(self):
        rospy.loginfo("3D Dummy Obstacle Publisher waiting for waypoints...")
        rospy.wait_for_service("convert_frenet2globarr_service")
        rospy.wait_for_service("convert_glob2frenetarr_service")

        if self.waypoints_type == "updated":
            global_wpnts_msg = rospy.wait_for_message("/global_waypoints_updated", WpntArray)
        else:
            global_wpnts_msg = rospy.wait_for_message("/global_waypoints", WpntArray)
        global_wpnts, max_s = self.wpnts_cb(data=global_wpnts_msg)
        s_array = np.array([wpnt.s_m for wpnt in global_wpnts])

        if self.constant:
            for i in range(len(global_wpnts)):
                global_wpnts[i].vx_mps = 1 * self.speed_scaler
        else:
            for i in range(len(global_wpnts)):
                global_wpnts[i].vx_mps = global_wpnts[i].vx_mps*self.speed_scaler

        opponent_wpnts_msg = rospy.wait_for_message(self.waypoints_topic, WpntArray)
        opponent_wpnts_list, _ = self.wpnts_cb(data=opponent_wpnts_msg)

        ### HJ : pass z_m for 3D Glob2FrenetArr
        opponent_xy = self.glob2frenet(
            [wpnt.x_m for wpnt in opponent_wpnts_list],
            [wpnt.y_m for wpnt in opponent_wpnts_list],
            [wpnt.z_m for wpnt in opponent_wpnts_list],
        )
        ### HJ : end
        opponent_s = opponent_xy.s
        opponent_d = opponent_xy.d
        sorted_indices = sorted(range(len(opponent_s)), key=lambda i: opponent_s[i])
        opponent_s_sorted = [opponent_s[i] for i in sorted_indices]
        opponent_d_sorted = [opponent_d[i] for i in sorted_indices]
        resampeld_opponent_d = np.interp(s_array, opponent_s_sorted, opponent_d_sorted)
        resampeld_opponent_vs = [wpnt.vx_mps for wpnt in global_wpnts]
        resampled_opponent_xy = self.frenet2glob(s_array, resampeld_opponent_d)

        self.opponent_wpnts = OpponentTrajectory()
        for i in range(len(s_array)):
            wpnt = OppWpnt()
            wpnt.x_m = resampled_opponent_xy.x[i]
            wpnt.y_m = resampled_opponent_xy.y[i]
            ### HJ : OppWpnt has no z_m field — z is consumed downstream via Obstacle.z_m only
            wpnt.proj_vs_mps = resampeld_opponent_vs[i]
            wpnt.s_m = s_array[i]
            wpnt.d_m = resampeld_opponent_d[i]
            self.opponent_wpnts.oppwpnts.append(wpnt)

        rospy.sleep(0.1)

        self.dynamic_obstacle.s_center = self.starting_s

        opponent_s_array = np.array([wpnt.s_m for wpnt in self.opponent_wpnts.oppwpnts])
        rospy.loginfo("3D Dummy Obstacle Publisher ready.")

        counter = 0
        while not rospy.is_shutdown():
            obstacle_msg = ObstacleArray()
            obstacle_msg.header.stamp = rospy.Time.now()
            obstacle_msg.header.frame_id = "frenet"

            s = self.dynamic_obstacle.s_center
            approx_idx = np.abs(opponent_s_array - s).argmin()

            self.dyn_obstacle_speed = self.opponent_wpnts.oppwpnts[approx_idx].proj_vs_mps
            self.dynamic_obstacle.s_center = (self.dynamic_obstacle.s_center + self.dyn_obstacle_speed * self.looptime) % max_s
            self.dynamic_obstacle.s_start = (self.dynamic_obstacle.s_center - self.obj_len/2) % max_s
            self.dynamic_obstacle.s_end = (self.dynamic_obstacle.s_center + self.obj_len/2) % max_s
            self.dynamic_obstacle.d_center = self.opponent_wpnts.oppwpnts[approx_idx].d_m

            size = 0.4
            self.dynamic_obstacle.size = size
            self.dynamic_obstacle.d_right = self.dynamic_obstacle.d_center - size/2
            self.dynamic_obstacle.d_left = self.dynamic_obstacle.d_center + size/2

            self.dynamic_obstacle.vs = self.dyn_obstacle_speed
            resp = self.frenet2glob([self.dynamic_obstacle.s_center], [self.dynamic_obstacle.d_center])
            self.dynamic_obstacle.x_m = resp.x[0]
            self.dynamic_obstacle.y_m = resp.y[0]
            ### HJ : fill z_m from 3D Frenet2GlobArr response
            self.dynamic_obstacle.z_m = resp.z[0] if hasattr(resp, 'z') and len(resp.z) > 0 else 0.0
            ### HJ : end

            obstacle_msg.obstacles.append(self.dynamic_obstacle)
            self.publish_obstacle_cartesian(obstacle_msg.obstacles)

            self.obstacle_pub.publish(obstacle_msg)

            counter = counter + 1

            if counter > 25:
                opponent_traj_msg = OpponentTrajectory(header=rospy.Header(frame_id="map", stamp=rospy.Time.now()), lap_count=2)
                opponent_traj_msg.oppwpnts = self.opponent_wpnts.oppwpnts
                self.opponent_traj_pub.publish(opponent_traj_msg)
                counter = 0

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("obstacle_publisher", anonymous=False, log_level=rospy.INFO)
    obstacle_publisher = ObstaclePublisher3D()
    rospy.on_shutdown(obstacle_publisher.shutdown)
    obstacle_publisher.ros_loop()
