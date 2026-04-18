#!/usr/bin/env python3
"""
MPCC Planner Node.

Ported from icra_unicorn_heedo/mpc/scripts/mpcc_node.py. Used here as a
trajectory planner (no control output). The solver is a 2D kinematic
bicycle MPCC; this node slices the local reference window directly from
/global_waypoints (no BehaviorStrategy dependency) and publishes the
predicted MPC trajectory as a Path + MarkerArray for RViz.

Subscribes:
  /global_waypoints  (f110_msgs/WpntArray)
  /car_state/pose    (geometry_msgs/PoseStamped)
  /car_state/odom    (nav_msgs/Odometry)

Publishes:
  /planner/mpc/trajectory  (nav_msgs/Path)
  /planner/mpc/markers     (visualization_msgs/MarkerArray)
"""

import os
import sys
import time
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import ColorRGBA
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import WpntArray

# Import solver from sibling src/ directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.normpath(os.path.join(_this_dir, '..', 'src'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from mpcc_solver import MPCCSolver  # noqa: E402


class MPCPlannerNode:

    def __init__(self):
        rospy.init_node('mpc_planner', anonymous=False)

        # Solver parameters
        self.freq = rospy.get_param('~planner_freq', rospy.get_param('~controller_freq', 30))
        params = {
            'N':                 rospy.get_param('~N', 6),
            'dT':                rospy.get_param('~dT', 0.1),
            'vehicle_L':         rospy.get_param('~vehicle_L', 0.36),
            'max_speed':         rospy.get_param('~max_speed', 10.0),
            'min_speed':         rospy.get_param('~min_speed', 0.0),
            'max_steering':      rospy.get_param('~max_steering', 0.4),
            'max_progress':      rospy.get_param('~max_progress', 10.0),
            'w_contour':         rospy.get_param('~w_contour', 3.9),
            'w_lag':             rospy.get_param('~w_lag', 1.0),
            'w_progress':        rospy.get_param('~w_progress', 6.0),
            'w_velocity':        rospy.get_param('~w_velocity', 3.0),
            'v_bias_max':        rospy.get_param('~v_bias_max', 1.0),
            'w_dv':              rospy.get_param('~w_dv', 19.0),
            'w_dsteering':       rospy.get_param('~w_dsteering', 28.0),
            'w_dprogress':       rospy.get_param('~w_dprogress', 15.7),
            'boundary_inflation': rospy.get_param('~boundary_inflation', 0.34),
            'ipopt_max_iter':    rospy.get_param('~ipopt_max_iter', 200),
            'ipopt_print_level': rospy.get_param('~ipopt_print_level', 0),
        }
        self.N = params['N']
        self.solver = MPCCSolver(params)

        # Ego state
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0
        self.car_vx = 0.0
        self.pose_received = False

        # Global waypoint cache (populated on first /global_waypoints msg)
        self.global_cached = False
        self.g_s = None
        self.g_x = None
        self.g_y = None
        self.g_z = None              # ### HJ : z_m for 3D marker lift
        self.g_psi = None
        self.g_dleft = None
        self.g_dright = None
        self.g_vx = None

        self._timer = None

        # Publishers
        self.traj_pub = rospy.Publisher('/planner/mpc/trajectory', Path, queue_size=1)
        self.marker_pub = rospy.Publisher('/planner/mpc/markers', MarkerArray, queue_size=1)

        # Subscribers
        rospy.Subscriber('/global_waypoints', WpntArray, self._global_wpnts_cb, queue_size=1)
        rospy.Subscriber('/car_state/pose', PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber('/car_state/odom', Odometry, self._odom_cb, queue_size=1)

        rospy.loginfo("[MPC planner] Waiting for /global_waypoints to build LUTs...")

    def _global_wpnts_cb(self, msg):
        if self.global_cached:
            return
        wpnts = msg.wpnts
        n = len(wpnts)
        if n < 10:
            rospy.logwarn("[MPC planner] Too few global waypoints: %d", n)
            return

        self.g_s = np.array([w.s_m for w in wpnts], dtype=float)
        self.g_x = np.array([w.x_m for w in wpnts], dtype=float)
        self.g_y = np.array([w.y_m for w in wpnts], dtype=float)
        # ### HJ : cache z_m for lifting 2D MPC output to 3D visualization
        self.g_z = np.array([getattr(w, 'z_m', 0.0) for w in wpnts], dtype=float)
        self.g_psi = np.array([w.psi_rad for w in wpnts], dtype=float)
        self.g_dleft = np.array([w.d_left for w in wpnts], dtype=float)
        self.g_dright = np.array([w.d_right for w in wpnts], dtype=float)
        self.g_vx = np.array([w.vx_mps for w in wpnts], dtype=float)

        dx = np.cos(self.g_psi)
        dy = np.sin(self.g_psi)
        self.solver.build_luts(self.g_s, self.g_x, self.g_y, dx, dy,
                               self.g_psi, self.g_dleft, self.g_dright)
        self.global_cached = True
        rospy.loginfo("[MPC planner] LUTs built: %d waypoints, track=%.1fm",
                      n, self.solver.track_length)

        if self._timer is None:
            self._timer = rospy.Timer(rospy.Duration(1.0 / self.freq), self._plan_loop)

    def _pose_cb(self, msg):
        self.car_x = msg.pose.position.x
        self.car_y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, self.car_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose_received = True

    def _odom_cb(self, msg):
        self.car_vx = msg.twist.twist.linear.x

    def _slice_local_ref(self, s_cur):
        """
        Build N+1 local reference from /global_waypoints around s_cur.
        Reference s advances using the raceline's local vx at each step so
        that the slice window matches the speed profile the MPC is expected
        to track (avoids reference compression when the car is slow).
        Returns dict: center_points, left_points, right_points,
                      ref_v, ref_s, ref_dx, ref_dy.
        """
        N = self.N
        dT = self.solver.dT
        tl = self.solver.track_length

        inflation = self.solver.inflation

        center_pts, left_pts, right_pts = [], [], []
        ref_v, ref_s, ref_dx, ref_dy = [], [], [], []
        s_offset = 0.0
        prev_s = None

        target_s = s_cur
        for k in range(N + 1):
            s_wrap = target_s % tl if tl > 0 else target_s
            idx = int(np.argmin(np.abs(self.g_s - s_wrap)))

            x = self.g_x[idx]
            y = self.g_y[idx]
            psi = self.g_psi[idx]
            dl = self.g_dleft[idx]
            dr = self.g_dright[idx]

            normal = np.array([-np.sin(psi), np.cos(psi)])
            center = np.array([x, y])
            center_pts.append(center)
            left_pts.append(center + normal * max(dl - inflation, 0.0))
            right_pts.append(center - normal * max(dr - inflation, 0.0))
            ref_v.append(float(self.g_vx[idx]))
            ref_dx.append(np.cos(psi))
            ref_dy.append(np.sin(psi))

            s_val = float(self.g_s[idx])
            if prev_s is not None and s_val + s_offset < prev_s - 1.0:
                s_offset += max(tl or 100.0, 1.0)
            s_val += s_offset
            ref_s.append(s_val)
            prev_s = s_val

            # Advance the slicing cursor by the local raceline speed.
            # Uses raceline vx at this waypoint so the window spans the
            # distance the car will actually cover in one MPC step.
            local_v = max(float(self.g_vx[idx]), 1.0)
            target_s += local_v * dT

        return {
            'center_points': np.array(center_pts),
            'left_points': np.array(left_pts),
            'right_points': np.array(right_pts),
            'ref_v': np.array(ref_v),
            'ref_s': np.array(ref_s),
            'ref_dx': np.array(ref_dx),
            'ref_dy': np.array(ref_dy),
        }

    def _plan_loop(self, event):
        if not self.solver.ready or not self.pose_received:
            return

        raw_s, _ = self.solver.find_nearest_s(self.car_x, self.car_y)
        s_cur = self.solver.filter_s(raw_s)

        ref_slice = self._slice_local_ref(s_cur)
        initial_state = np.array([self.car_x, self.car_y, self.car_yaw, s_cur])

        t0 = time.time()
        speed, steering, trajectory, success = self.solver.solve(initial_state, ref_slice)
        solve_ms = (time.time() - t0) * 1000.0

        if success and trajectory is not None:
            self._publish_trajectory(trajectory)
            self._publish_markers(trajectory)
        else:
            # Drop stale warm-start so the next solve starts clean and does
            # not inherit an infeasible initial guess.
            self.solver.reset_warm_start()

        rospy.loginfo_throttle(
            1.0,
            "[MPC planner] solve=%.1fms v0=%.2f steer=%.3f ok=%s",
            solve_ms, speed, steering, success,
        )

    def _lift_z(self, x, y):
        """### HJ : approximate z for a 2D MPC point via nearest global waypoint."""
        if self.g_z is None:
            return 0.0
        dists = (self.g_x - x) ** 2 + (self.g_y - y) ** 2
        idx = int(np.argmin(dists))
        return float(self.g_z[idx])

    def _publish_trajectory(self, trajectory):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = 'map'
        for i in range(trajectory.shape[0]):
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = trajectory[i, 0]
            p.pose.position.y = trajectory[i, 1]
            p.pose.position.z = self._lift_z(trajectory[i, 0], trajectory[i, 1])  # ### HJ
            q = quaternion_from_euler(0, 0, trajectory[i, 2])
            p.pose.orientation.x = q[0]
            p.pose.orientation.y = q[1]
            p.pose.orientation.z = q[2]
            p.pose.orientation.w = q[3]
            path.poses.append(p)
        self.traj_pub.publish(path)

    def _publish_markers(self, trajectory):
        now = rospy.Time.now()
        arr = MarkerArray()

        # LINE_STRIP of predicted path
        line = Marker()
        line.header.stamp = now
        line.header.frame_id = 'map'
        line.ns = 'mpc_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.08
        line.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.9)
        line.pose.orientation.w = 1.0
        for i in range(trajectory.shape[0]):
            pt = Point()
            pt.x = float(trajectory[i, 0])
            pt.y = float(trajectory[i, 1])
            pt.z = self._lift_z(trajectory[i, 0], trajectory[i, 1])  # ### HJ
            line.points.append(pt)
        arr.markers.append(line)

        # SPHERE_LIST at each MPC step
        pts = Marker()
        pts.header.stamp = now
        pts.header.frame_id = 'map'
        pts.ns = 'mpc_steps'
        pts.id = 1
        pts.type = Marker.SPHERE_LIST
        pts.action = Marker.ADD
        pts.scale.x = pts.scale.y = pts.scale.z = 0.12
        pts.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.9)
        pts.pose.orientation.w = 1.0
        for i in range(trajectory.shape[0]):
            pt = Point()
            pt.x = float(trajectory[i, 0])
            pt.y = float(trajectory[i, 1])
            pt.z = self._lift_z(trajectory[i, 0], trajectory[i, 1])  # ### HJ
            pts.points.append(pt)
        arr.markers.append(pts)

        self.marker_pub.publish(arr)


if __name__ == '__main__':
    try:
        MPCPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
