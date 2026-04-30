#!/usr/bin/env python3
"""Launch-mode node: current-based throttle override for race start.

Sits in parallel to the regular controller (mu_ppc / upenn_mpc / mpcc_dyna).
When the human arms the launch channel and the FSM fires, this node:

  * publishes raw motor current at high rate to /vesc/commands/motor/current,
    bypassing the speed-PID wind-up of VESC's commanded_speed loop;
  * publishes a high-priority Ackermann to the high_level mux carrying the
    pass-through steering from the active controller, so the steer keeps
    using whatever the controller (PPC/MPC) computed;
  * publishes /launch_mode/active so other nodes (and the controller) can
    avoid fighting for the throttle channel.

After exit conditions trigger (distance, speed ratio, time, corner), the
node hands off back to the regular controller by stopping current commands
and dropping /launch_mode/active to False. Hand-off is *speed-PID
pre-loaded* by also publishing a one-shot speed command equal to the most
recently observed v_meas, so VESC's PI integrator doesn't start from zero.

VESC firmware limits (motor_current_max, abs_max_current,
battery_current_max) are still the absolute ceiling; this node does NOT
attempt to raise them.

Topics consumed
---------------
/joy                                          : sensor_msgs/Joy
/car_state/odom                               : nav_msgs/Odometry  (v_body)
/car_state/odom_frenet                        : nav_msgs/Odometry  (s, v)
/vesc/sensors/core                            : vesc_msgs/VescStateStamped
/vesc/odom                                    : nav_msgs/Odometry  (v_wheel)
/global_waypoints                             : f110_msgs/WpntArray (kappa)
<controller_ackermann_topic>                  : ackermann_msgs/AckermannDriveStamped
                                                (steering pass-through)

Topics produced
---------------
/vesc/commands/motor/current                  : std_msgs/Float64
/vesc/commands/motor/speed                    : std_msgs/Float64 (pre-load on hand-off)
<launch_ackermann_topic> (high-priority mux)  : ackermann_msgs/AckermannDriveStamped
/launch_mode/active                           : std_msgs/Bool
/launch_mode/state                            : std_msgs/Int8
/launch_mode/debug                            : std_msgs/Float32MultiArray
"""

import os
import sys
import threading

import numpy as np
import rospy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import Bool, ColorRGBA, Float32MultiArray, Float64, Int8, String
from visualization_msgs.msg import Marker, MarkerArray

try:
    from f110_msgs.msg import WpntArray
    HAVE_WPNT = True
except Exception:
    HAVE_WPNT = False

try:
    from vesc_msgs.msg import VescStateStamped
    HAVE_VESC = True
except Exception:
    HAVE_VESC = False

from launch_mode.src.launch_fsm import LaunchFSM, LaunchState
from launch_mode.src.slip_guard import SlipGuard
from launch_mode.src.proximity_guard import ProximityGuard


class LaunchModeNode:
    def __init__(self):
        rospy.init_node('launch_mode', anonymous=False)
        self.lock = threading.Lock()
        self.rate_hz = rospy.get_param('~rate', 200)

        # ---- joy channel mapping (ELRS via elrs_joy_node) ----
        self.auto_arm_button = rospy.get_param('~auto_arm_button', 5)
        self.launch_arm_axis = rospy.get_param('~launch_arm_axis', -1)
        self.launch_arm_button = rospy.get_param('~launch_arm_button', 7)
        self.launch_arm_threshold = rospy.get_param('~launch_arm_threshold', 0.5)
        self.go_axis = rospy.get_param('~go_axis', -1)
        self.go_button = rospy.get_param('~go_button', 6)
        self.go_threshold = rospy.get_param('~go_threshold', 0.5)
        self.joy_timeout_s = rospy.get_param('~joy_timeout_s', 0.5)

        # ---- physical / VESC limits ----
        self.i_launch_max = rospy.get_param('~i_launch_max', 40.0)        # [A]
        self.i_firmware_cap = rospy.get_param('~i_firmware_cap', 60.0)    # [A]

        # ---- topics ----
        self.joy_topic = rospy.get_param('~joy_topic', '/joy')
        self.odom_topic = rospy.get_param('~odom_topic', '/car_state/odom')
        self.frenet_odom_topic = rospy.get_param('~frenet_odom_topic',
                                                 '/car_state/odom_frenet')
        self.vesc_odom_topic = rospy.get_param('~vesc_odom_topic', '/vesc/odom')
        self.vesc_core_topic = rospy.get_param('~vesc_core_topic',
                                               '/vesc/sensors/core')
        self.waypoints_topic = rospy.get_param('~waypoints_topic',
                                               '/global_waypoints')
        self.controller_ackermann_topic = rospy.get_param(
            '~controller_ackermann_topic',
            '/vesc/high_level/ackermann_cmd_mux/input/nav_1')
        self.current_topic = rospy.get_param('~current_topic',
                                             '/vesc/commands/motor/current')
        self.speed_topic = rospy.get_param('~speed_topic',
                                           '/vesc/commands/motor/speed')
        self.launch_ackermann_topic = rospy.get_param(
            '~launch_ackermann_topic',
            '/vesc/high_level/ackermann_cmd_mux/input/launch')
        self.servo_command_topic = rospy.get_param(
            '~servo_command_topic',
            '/vesc/sensors/servo_position_command')
        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.viz_frame = rospy.get_param('~viz_frame', 'base_link')
        self.viz_text_height = float(rospy.get_param('~viz_text_height', 0.5))
        self.viz_text_size = float(rospy.get_param('~viz_text_size', 0.4))

        # ---- waypoint lookahead window ----
        self.wpnt_ds = rospy.get_param('~wpnt_ds', 0.10)
        self.wpnt_lookahead_m = rospy.get_param('~wpnt_lookahead_m', 2.0)

        # ---- grid / proximity ----
        self.grid_slot = str(rospy.get_param('~grid_slot', 'rear')).lower()
        self.reduced_i_factor = float(rospy.get_param('~reduced_i_factor', 0.6))
        self.reduced_s_launch_m = float(rospy.get_param('~reduced_s_launch_m', 1.5))
        self.proximity = ProximityGuard(
            front_x_max=rospy.get_param('~front_roi_x_max', 1.5),
            front_y_half=rospy.get_param('~front_roi_y_half', 0.30),
            side_x_min=rospy.get_param('~side_roi_x_min', -0.5),
            side_x_max=rospy.get_param('~side_roi_x_max', 1.5),
            side_y_min=rospy.get_param('~side_roi_y_min', 0.30),
            side_y_max=rospy.get_param('~side_roi_y_max', 1.50),
            roi_min_points=rospy.get_param('~roi_min_points', 5),
            safe_lateral_gap_m=rospy.get_param('~safe_lateral_gap_m', 0.6),
            front_ttc_th_s=rospy.get_param('~front_ttc_th_s', 0.4),
            front_proximity_min_m=rospy.get_param('~front_proximity_min_m', 0.3),
            scan_timeout_s=rospy.get_param('~scan_timeout_s', 0.3),
        )

        # ---- FSM tunables ----
        fsm = LaunchFSM(
            v_stationary=rospy.get_param('~v_stationary', 0.1),
            s_launch_m=rospy.get_param('~s_launch_m', 5.0),
            v_ratio_exit=rospy.get_param('~v_ratio_exit', 0.7),
            t_max_s=rospy.get_param('~t_max_s', 1.5),
            steer_exit_rad=rospy.get_param('~steer_exit_rad', 0.2),
            kappa_exit=rospy.get_param('~kappa_exit', 0.5),
        )
        self.fsm = fsm

        # ---- slip guard ----
        self.slip = SlipGuard(
            slip_threshold=rospy.get_param('~slip_threshold', 0.20),
            confirm_n=rospy.get_param('~slip_confirm_n', 2),
            cut_factor=rospy.get_param('~slip_cut_factor', 0.8),
            i_floor_ratio=rospy.get_param('~i_floor_ratio', 0.4),
        )

        # ---- state buffers ----
        self.t_last_joy = 0.0
        self.autonomous_armed = False
        self.launch_arm = False
        self.go_signal = False

        self.v_body = 0.0
        self.v_wheel = 0.0
        self.s_now = 0.0
        self.v_ref = 0.0
        self.kappa_ahead = 0.0
        self.steer_passthrough = 0.0
        self.speed_passthrough = 0.0
        self.has_wpnts = False

        self.track_length = rospy.get_param('/global_republisher/track_length', 0.0)

        # ---- publishers ----
        self.pub_current = rospy.Publisher(self.current_topic, Float64, queue_size=1)
        self.pub_speed = rospy.Publisher(self.speed_topic, Float64, queue_size=1)
        self.pub_ackermann = rospy.Publisher(self.launch_ackermann_topic,
                                             AckermannDriveStamped, queue_size=1)
        self.pub_active = rospy.Publisher('/launch_mode/active', Bool,
                                          queue_size=1, latch=True)
        self.pub_state = rospy.Publisher('/launch_mode/state', Int8,
                                         queue_size=1, latch=True)
        self.pub_dbg = rospy.Publisher('/launch_mode/debug',
                                       Float32MultiArray, queue_size=10)
        self.pub_intent = rospy.Publisher('/launch_mode/intent', String,
                                          queue_size=1, latch=True)
        # advisory feasibility — continuous, latched so consumers see latest
        self.pub_feasible = rospy.Publisher('/launch_mode/feasible', Bool,
                                            queue_size=1, latch=True)
        self.pub_feasibility = rospy.Publisher('/launch_mode/feasibility',
                                               String, queue_size=1, latch=True)
        self.pub_markers = rospy.Publisher('/launch_mode/markers',
                                           MarkerArray, queue_size=1, latch=True)
        self._last_feasibility = ''

        # ---- subscribers ----
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=20)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(self.frenet_odom_topic, Odometry, self._frenet_cb,
                         queue_size=10)
        rospy.Subscriber(self.vesc_odom_topic, Odometry, self._vesc_odom_cb,
                         queue_size=10)
        if HAVE_VESC:
            rospy.Subscriber(self.vesc_core_topic, VescStateStamped,
                             self._vesc_core_cb, queue_size=10)
        rospy.Subscriber(self.controller_ackermann_topic,
                         AckermannDriveStamped, self._ackermann_cb, queue_size=5)
        if HAVE_WPNT:
            rospy.Subscriber(self.waypoints_topic, WpntArray, self._wpnts_cb,
                             queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self._scan_cb, queue_size=1)

        rospy.loginfo('[launch_mode] ready. i_launch_max=%.1fA cap=%.1fA',
                      self.i_launch_max, self.i_firmware_cap)
        self._publish_active(False)
        self._publish_state(LaunchState.IDLE)

    # ----------------------------- callbacks --------------------------------
    def _joy_cb(self, msg):
        with self.lock:
            self.t_last_joy = rospy.get_time()
            n_buttons = len(msg.buttons)
            n_axes = len(msg.axes)

            self.autonomous_armed = (
                0 <= self.auto_arm_button < n_buttons
                and bool(msg.buttons[self.auto_arm_button])
            )

            if self.launch_arm_axis >= 0 and self.launch_arm_axis < n_axes:
                self.launch_arm = msg.axes[self.launch_arm_axis] > self.launch_arm_threshold
            elif 0 <= self.launch_arm_button < n_buttons:
                self.launch_arm = bool(msg.buttons[self.launch_arm_button])
            else:
                self.launch_arm = False

            if self.go_axis >= 0 and self.go_axis < n_axes:
                self.go_signal = msg.axes[self.go_axis] > self.go_threshold
            elif 0 <= self.go_button < n_buttons:
                self.go_signal = bool(msg.buttons[self.go_button])
            else:
                self.go_signal = False

    def _odom_cb(self, msg):
        with self.lock:
            self.v_body = float(msg.twist.twist.linear.x)

    def _frenet_cb(self, msg):
        with self.lock:
            self.s_now = float(msg.pose.pose.position.x)

    def _vesc_odom_cb(self, msg):
        with self.lock:
            self.v_wheel = float(msg.twist.twist.linear.x)

    def _vesc_core_cb(self, msg):
        # placeholder: motor current/temperature monitoring hook
        pass

    def _scan_cb(self, msg):
        with self.lock:
            self.proximity.update(msg, t_now=rospy.get_time())

    def _ackermann_cb(self, msg):
        with self.lock:
            self.steer_passthrough = float(msg.drive.steering_angle)
            self.speed_passthrough = float(msg.drive.speed)

    def _wpnts_cb(self, msg):
        with self.lock:
            self.has_wpnts = True
            if not msg.wpnts:
                return
            ds = self.wpnt_ds
            n_look = max(1, int(round(self.wpnt_lookahead_m / ds)))
            v_ref = 0.0
            kappa_max = 0.0
            i0 = int(round(self.s_now / ds)) % len(msg.wpnts)
            for k in range(n_look):
                w = msg.wpnts[(i0 + k) % len(msg.wpnts)]
                v_ref = max(v_ref, getattr(w, 'vx_mps', 0.0))
                kap = abs(getattr(w, 'kappa_radpm', 0.0))
                kappa_max = max(kappa_max, kap)
            self.v_ref = v_ref
            self.kappa_ahead = kappa_max

    # ----------------------------- helpers ----------------------------------
    def _publish_active(self, active):
        self.pub_active.publish(Bool(data=bool(active)))

    def _publish_state(self, state):
        self.pub_state.publish(Int8(data=int(state)))

    def _publish_current(self, i_cmd):
        i_cmd = max(0.0, min(i_cmd, self.i_firmware_cap))
        self.pub_current.publish(Float64(data=i_cmd))

    def _publish_speed_preload(self, v_meas):
        self.pub_speed.publish(Float64(data=max(0.0, v_meas)))

    def _feasibility_color(self, feas):
        # RGBA on [0..1]; alpha kept high so it's readable in RViz
        if feas == 'GREEN':
            return ColorRGBA(0.10, 0.85, 0.10, 0.95)
        if feas == 'YELLOW':
            return ColorRGBA(0.95, 0.85, 0.10, 0.95)
        if feas == 'RED':
            return ColorRGBA(0.95, 0.10, 0.10, 0.95)
        return ColorRGBA(0.5, 0.5, 0.5, 0.85)  # UNKNOWN

    def _box_corners(self, x0, x1, y0, y1, z=0.05):
        # closed rectangle (LINE_STRIP needs the loop closed manually)
        return [Point(x0, y0, z), Point(x1, y0, z),
                Point(x1, y1, z), Point(x0, y1, z),
                Point(x0, y0, z)]

    def _publish_feasibility_markers(self, feas, prox):
        ma = MarkerArray()
        stamp = rospy.Time.now()
        color = self._feasibility_color(feas)

        # 1) feasibility text floating above the car
        m_txt = Marker()
        m_txt.header.frame_id = self.viz_frame
        m_txt.header.stamp = stamp
        m_txt.ns = 'launch_mode/feasibility'
        m_txt.id = 0
        m_txt.type = Marker.TEXT_VIEW_FACING
        m_txt.action = Marker.ADD
        m_txt.pose.position.x = 0.0
        m_txt.pose.position.y = 0.0
        m_txt.pose.position.z = self.viz_text_height
        m_txt.pose.orientation.w = 1.0
        m_txt.scale.z = self.viz_text_size
        m_txt.color = color
        front_obs = bool(prox.get('front_obstacle', False))
        lat = prox.get('lateral_gap', float('inf'))
        lat_str = f'{lat:.2f}m' if lat != float('inf') else '--'
        m_txt.text = f'LAUNCH: {feas}\nfront_obs={front_obs} lat={lat_str}'
        ma.markers.append(m_txt)

        # 2) front ROI rectangle
        m_front = Marker()
        m_front.header.frame_id = self.viz_frame
        m_front.header.stamp = stamp
        m_front.ns = 'launch_mode/roi_front'
        m_front.id = 1
        m_front.type = Marker.LINE_STRIP
        m_front.action = Marker.ADD
        m_front.scale.x = 0.02
        m_front.color = color
        m_front.pose.orientation.w = 1.0
        x0, x1 = 0.0, self.proximity.front_x_max
        y0, y1 = -self.proximity.front_y_half, self.proximity.front_y_half
        m_front.points = self._box_corners(x0, x1, y0, y1)
        ma.markers.append(m_front)

        # 3) side ROI rectangles (left + right)
        side_color = ColorRGBA(0.4, 0.7, 1.0, 0.9)
        for idx, sign in enumerate((+1, -1)):
            m_side = Marker()
            m_side.header.frame_id = self.viz_frame
            m_side.header.stamp = stamp
            m_side.ns = 'launch_mode/roi_side'
            m_side.id = 10 + idx
            m_side.type = Marker.LINE_STRIP
            m_side.action = Marker.ADD
            m_side.scale.x = 0.02
            m_side.color = side_color
            m_side.pose.orientation.w = 1.0
            sx0, sx1 = self.proximity.side_x_min, self.proximity.side_x_max
            if sign > 0:
                sy0, sy1 = self.proximity.side_y_min, self.proximity.side_y_max
            else:
                sy0, sy1 = -self.proximity.side_y_max, -self.proximity.side_y_min
            m_side.points = self._box_corners(sx0, sx1, sy0, sy1)
            ma.markers.append(m_side)

        self.pub_markers.publish(ma)

    def _publish_launch_ackermann(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.pub_ackermann.publish(msg)

    def _joy_alive(self):
        if self.t_last_joy <= 0.0:
            return False
        return (rospy.get_time() - self.t_last_joy) < self.joy_timeout_s

    # ----------------------------- main loop --------------------------------
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        prev_state = LaunchState.IDLE
        while not rospy.is_shutdown():
            with self.lock:
                t_now = rospy.get_time()
                joy_alive = self._joy_alive()

                # continuous feasibility advisory (GREEN/YELLOW/RED/UNKNOWN)
                feasibility = self.proximity.feasibility(self.grid_slot,
                                                         t_now=t_now)
                prox_snapshot = dict(self.proximity.last)
                if feasibility != self._last_feasibility:
                    self._last_feasibility = feasibility
                    self.pub_feasibility.publish(String(data=feasibility))
                    self.pub_feasible.publish(
                        Bool(data=feasibility in ('GREEN', 'YELLOW')))
                    self._publish_feasibility_markers(feasibility, prox_snapshot)
                    rospy.loginfo('[launch_mode] feasibility=%s', feasibility)

                # decide intent only at ARMED -> LAUNCH transition; the FSM
                # only consults this on the GO rising edge.
                proximity_intent = self.proximity.decide_launch_intent(
                    self.grid_slot, t_now=t_now)
                # in-launch runtime abort: scan-driven cutoff
                proximity_runtime_abort = (
                    self.fsm.state == LaunchState.LAUNCH
                    and self.proximity.runtime_abort(self.v_body, t_now=t_now)
                )

                state = self.fsm.update(
                    t_now=t_now,
                    autonomous_armed=self.autonomous_armed,
                    launch_arm=self.launch_arm,
                    go_signal=self.go_signal,
                    v_now=self.v_body,
                    s_now=self.s_now,
                    v_ref=self.v_ref,
                    steer_now=self.steer_passthrough,
                    kappa_ahead=self.kappa_ahead,
                    joy_alive=joy_alive,
                    proximity_intent=proximity_intent,
                    proximity_runtime_abort=proximity_runtime_abort,
                    reduced_s_launch_m=self.reduced_s_launch_m,
                )

                v_body = self.v_body
                v_wheel = self.v_wheel if self.v_wheel > 0.0 else self.v_body
                steer = self.steer_passthrough
                v_ref = self.v_ref
                intent = self.fsm.intent
                prox = dict(self.proximity.last)

            if state != prev_state:
                rospy.loginfo('[launch_mode] %s -> %s (v=%.2f s=%.2f vref=%.2f)',
                              prev_state.name, state.name, v_body, self.s_now, v_ref)
                self._publish_state(state)
                # entering LAUNCH: reset slip guard
                if state == LaunchState.LAUNCH:
                    self.slip.reset()
                # leaving LAUNCH: pre-load speed PID and drop active
                if prev_state == LaunchState.LAUNCH and state != LaunchState.LAUNCH:
                    self._publish_speed_preload(v_body)
                    self._publish_active(False)
                # entering LAUNCH: announce active
                if state == LaunchState.LAUNCH:
                    self._publish_active(True)
                    self.pub_intent.publish(String(data=intent))
                    rospy.loginfo('[launch_mode] intent=%s grid=%s '
                                  'front_obs=%s lat_gap=%.2f',
                                  intent, self.grid_slot,
                                  prox['front_obstacle'], prox['lateral_gap'])
                prev_state = state

            if state == LaunchState.LAUNCH:
                # current scaled by intent (FULL or REDUCED)
                base_i = self.i_launch_max
                if intent == 'REDUCED':
                    base_i *= self.reduced_i_factor
                self.slip.update(v_wheel=v_wheel, v_body=v_body)
                i_cmd = self.slip.apply(base_i)
                self._publish_current(i_cmd)
                # high-priority ackermann carrying steer (speed is informational)
                self._publish_launch_ackermann(speed=max(v_body, 1.0),
                                               steer=steer)
                dbg = Float32MultiArray()
                dbg.data = [float(state),
                            i_cmd,
                            self.slip.slip,
                            self.slip.scale,
                            v_body,
                            v_wheel,
                            v_ref,
                            self.kappa_ahead,
                            float(prox.get('front_obstacle', False)),
                            prox.get('front_min_range', float('inf')),
                            prox.get('lateral_gap', float('inf'))]
                self.pub_dbg.publish(dbg)
            # other states: be silent on /commands/motor/current so the regular
            # speed path drives the motor.

            rate.sleep()


def main():
    node = LaunchModeNode()
    node.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
