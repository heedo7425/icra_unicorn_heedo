#!/usr/bin/env python3
"""Standalone ROS node for friction-aware Pure Pursuit (mu_ppc).

Subscribes to the same topics the existing controller_manager uses and
publishes Ackermann commands. Intended for bench/sim runs of mu_ppc without
booting the full controller_manager pipeline. For race-day, mu_ppc is also
selectable through controller_manager via ctrl_algo='MU_PPC'.
"""

import os
import sys
import threading
import numpy as np
import rospy

# Make `controller/` importable so `mu_ppc.src.*` resolves the same way
# combined.src.Controller is loaded by controller_manager.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import BehaviorStrategy, WpntArray
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Bool
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

from mu_ppc.src.MuPPC import MuPPC


class MuPPCNode:
    def __init__(self):
        rospy.init_node('mu_ppc', anonymous=False)
        self.lock = threading.Lock()
        self.rate_hz = rospy.get_param('~rate', 50)
        wb = rospy.get_param('/vesc/wheelbase', 0.33)

        self.ctrl = MuPPC(
            wheelbase=wb,
            loop_rate=self.rate_hz,
            ld_base_min=rospy.get_param('~ld_base_min', 0.6),
            ld_base_max=rospy.get_param('~ld_base_max', 3.0),
            ld_speed_slope=rospy.get_param('~ld_speed_slope', 0.30),
            ld_speed_intercept=rospy.get_param('~ld_speed_intercept', 0.20),
            max_steer=rospy.get_param('~max_steer', 0.42),
            logger_info=rospy.loginfo,
            logger_warn=rospy.logwarn,
        )

        self._apply_cfg()
        self.ctrl.set_zone_map_from_rosparam(rospy)

        self.position = None
        self.frenet = None
        self.speed = 0.0
        self.acc_y_buf = np.zeros(10)
        self.wpts = None
        self.state = ''
        self.opponent = None
        self.track_length = rospy.get_param('/global_republisher/track_length', 0.0)

        rospy.Subscriber('/behavior_strategy', BehaviorStrategy, self.behavior_cb)
        rospy.Subscriber('/car_state/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/car_state/pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/car_state/odom_frenet', Odometry, self.frenet_cb)
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/vesc/odom', Odometry, self.vesc_cb)

        topic = rospy.get_param('~drive_topic',
                                '/vesc/high_level/ackermann_cmd_mux/input/nav_1')
        self.drive_pub = rospy.Publisher(topic, AckermannDriveStamped, queue_size=10)
        self.la_pub = rospy.Publisher('/mu_ppc/lookahead', Marker, queue_size=10)
        self.dbg_pub = rospy.Publisher('/mu_ppc/debug', Point, queue_size=10)
        self.alpha_pub = rospy.Publisher('/mu_ppc/alpha', Float32, queue_size=10)

    def _apply_cfg(self):
        z = self.ctrl.zone_map
        z.edge_blend = rospy.get_param('~mu_zone_edge_blend', 2.0)
        z.default_mu = rospy.get_param('~mu_default', 1.0)
        s = self.ctrl.slip
        s.yaw_thr = rospy.get_param('~slip_yaw_thr', 0.15)
        s.long_thr = rospy.get_param('~slip_long_thr', 0.10)
        s.ay_thr = rospy.get_param('~slip_ay_thr', 0.20)
        s.tau_up = rospy.get_param('~slip_tau_up', 0.05)
        s.tau_down = rospy.get_param('~slip_tau_down', 0.30)
        g = self.ctrl.sched
        g.ld_low = rospy.get_param('~ld_scale_low', 1.6)
        g.ld_high = rospy.get_param('~ld_scale_high', 0.8)
        g.sr_low = rospy.get_param('~steer_rate_low', 2.5)
        g.sr_high = rospy.get_param('~steer_rate_high', 8.0)
        g.ax_low = rospy.get_param('~ax_max_low', 2.0)
        g.ax_high = rospy.get_param('~ax_max_high', 6.0)
        g.br_low = rospy.get_param('~brake_low', 2.5)
        g.br_high = rospy.get_param('~brake_high', 7.0)
        g.tau_tighten = rospy.get_param('~gain_tau_tighten', 0.08)
        g.tau_relax = rospy.get_param('~gain_tau_relax', 0.6)
        g.prior_weight = rospy.get_param('~prior_weight', 0.5)

    # callbacks -----------------------------------------------------------------
    def odom_cb(self, m):
        self.speed = m.twist.twist.linear.x

    def vesc_cb(self, m):
        self.ctrl.set_wheel_speed(m.twist.twist.linear.x)

    def pose_cb(self, m):
        x = m.pose.position.x
        y = m.pose.position.y
        q = m.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.position = np.array([[x, y, yaw]])

    def frenet_cb(self, m):
        self.frenet = np.array([m.pose.pose.position.x, m.pose.pose.position.y,
                                m.twist.twist.linear.x, m.twist.twist.linear.y])

    def imu_cb(self, m):
        self.acc_y_buf[1:] = self.acc_y_buf[:-1]
        self.acc_y_buf[0] = -m.linear_acceleration.y
        self.ctrl.set_yaw_rate(-m.angular_velocity.z)

    def behavior_cb(self, m):
        self.state = m.state
        if not m.local_wpnts:
            self.wpts = None
            return
        arr = []
        for w in m.local_wpnts:
            arr.append([w.x_m, w.y_m, w.z_m, w.vx_mps, 0.0,
                        w.s_m, w.kappa_radpm, w.psi_rad, w.ax_mps2, w.d_m])
        self.wpts = np.array(arr)

    # main loop -----------------------------------------------------------------
    def spin(self):
        rospy.wait_for_message('/behavior_strategy', BehaviorStrategy)
        rospy.wait_for_message('/car_state/pose', PoseStamped)
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.position is None or self.wpts is None:
                rate.sleep()
                continue
            speed, accel, jerk, steer, la, lad, idx, kappa, fut = self.ctrl.main_loop(
                self.state, self.position, self.wpts, self.speed,
                self.opponent, self.frenet, self.acc_y_buf, self.track_length)
            msg = AckermannDriveStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'base_link'
            msg.drive.speed = speed
            msg.drive.acceleration = accel
            msg.drive.steering_angle = steer
            self.drive_pub.publish(msg)
            self.alpha_pub.publish(Float32(self.ctrl.sched._state.get('ld_scale', 1.0)))
            self.dbg_pub.publish(Point(x=self.ctrl.slip.level,
                                       y=self.ctrl.sched._state['ax_max'],
                                       z=self.ctrl.sched._state['steer_rate']))
            rate.sleep()


if __name__ == '__main__':
    MuPPCNode().spin()
