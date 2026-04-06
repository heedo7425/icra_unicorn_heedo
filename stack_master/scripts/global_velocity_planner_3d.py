#!/usr/bin/env python3
### HJ : 3D version of global_velocity_planner — uses vel_planner_25d with g_tilde

import rospy
import os
import math
import numpy as np
import json
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64
import trajectory_planning_helpers as tph
from rospkg import RosPack
from f110_msgs.msg import ObstacleArray, OTWpntArray, WpntArray, Wpnt
from vel_planner_25d.vel_planner import calc_vel_profile  ### HJ : 2.5D vel_planner with g_tilde

from copy import deepcopy
import configparser
import shutil

class VelocityPlanner:

    # ===== HJ ADDED: Flag to save modified parameters to CSV files =====

    # SAVE_CONFIG = True  # Set to True to save modified parameters and backup originals

    SAVE_CONFIG = False  # Set to True to save modified parameters and backup originals

    # ===== HJ ADDED END =====

    def __init__(self):
        """
        Initialize the node, subscribe to topics, create publishers and set up member variables.
        """

        self.racecar_version = rospy.get_param("/racecar_version")

        # Velocity Planning
        parser = configparser.ConfigParser()
        self.pars = {}
        if not parser.read(os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, 'racecar_f110.ini')):
            raise ValueError('Specified config file does not exist or is empty!')
        self.pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
        self.pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))
        ggv_path = os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "ggv.csv")
        ax_max_path = os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "ax_max_machines.csv")
        b_ax_max_path = os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "b_ax_max_machines.csv")
        self.ggv, self.ax_max_machines = tph.import_veh_dyn_info.\
            import_veh_dyn_info(ggv_import_path=ggv_path,
                                ax_max_machines_import_path=ax_max_path)
            
        _, self.b_ax_max_machines = tph.import_veh_dyn_info.\
            import_veh_dyn_info(ggv_import_path=ggv_path,
                                ax_max_machines_import_path=b_ax_max_path)

                                
        self.v_max = self.pars["veh_params"]["v_max"]
        self.drag_coeff = self.pars["veh_params"]["dragcoeff"]
        self.m_veh = self.pars["veh_params"]["mass"]
        self.filt_window = self.pars["vel_calc_opts"]["vel_profile_conv_filt_window"]
        self.dyn_model_exp = self.pars["vel_calc_opts"]["dyn_model_exp"] 

        # self.v_max = 12.0
        # self.ax_max_motor = 7
        # self.ax_max_brake = 7
        # self.dyn_model_exp = 1.1

        # self.a_y_max = 5.5
        # self.a_x_max = 7.0
#---------------------------------------

        self.v_max = 12.0
        self.ax_max_motor = 5
        self.ax_max_brake = 7
        self.dyn_model_exp = 1.0

        self.a_y_max = 4.5
        self.a_x_max = 5
#---------------------------------------
        # self.v_max = 15.0
        # self.ax_max_motor = 12.0
        # self.ax_max_brake = 9.0
        # self.dyn_model_exp = 1.4

        # self.a_y_max = 8.0
        # self.a_x_max = 13.0
        #aggresive
#---------------------------------------

        # self.v_max = 12.0
        # self.ax_max_motor = 7.0
        # self.ax_max_brake = 9.0
        # self.dyn_model_exp = 1.4

        # self.a_y_max = 6.5
        # self.a_x_max = 9.3

#---------------------------------------

        self.ggv[:,1] = self.a_x_max
        self.ggv[:,2] = self.a_y_max
        self.ax_max_machines[:,1] = self.ax_max_motor
        self.b_ax_max_machines[:,1] = self.ax_max_brake

        # ===== HJ ADDED: Save modified parameters to CSV if flag is enabled =====
        if self.SAVE_CONFIG:
            self._save_modified_config(ggv_path, ax_max_path, b_ax_max_path)
        # ===== HJ ADDED END =====

        ### HJ : read cog_z for g_tilde calculation
        self.h_cog = self.pars["veh_params"].get("cog_z", 0.074)

        # Subscribe to the topics
        self.glb_wpnts_pub = rospy.Publisher('/global_waypoints', WpntArray, queue_size=10)
        self.smart_static_wpnts_pub = rospy.Publisher('/planner/avoidance/smart_static_otwpnts', OTWpntArray, queue_size=10)

        rospy.Subscriber("/global_waypoints", WpntArray, self.wpnts_callback)
        rospy.Subscriber("/planner/avoidance/smart_static_otwpnts", OTWpntArray, self.smart_static_wpnts_callback)

    def _save_modified_config(self, ggv_path, ax_max_path, b_ax_max_path):
        """
        Save modified ggv and ax_max_machines parameters to CSV files.
        Backup original files to 'original/' folder (only if not already backed up).
        """
        # Get directory where CSV files are located
        veh_dyn_dir = os.path.dirname(ggv_path)
        original_dir = os.path.join(veh_dyn_dir, 'original')

        # Create original/ directory if it doesn't exist
        if not os.path.exists(original_dir):
            os.makedirs(original_dir)
            rospy.loginfo(f"[VelocityPlanner] Created backup directory: {original_dir}")

        # List of files to backup and save: (current_path, filename)
        files_to_process = [
            (ggv_path, 'ggv.csv'),
            (ax_max_path, 'ax_max_machines.csv'),
            (b_ax_max_path, 'b_ax_max_machines.csv')
        ]

        # Backup originals (only if not already backed up)
        for file_path, filename in files_to_process:
            backup_path = os.path.join(original_dir, filename)

            # Check if backup file exists AND has content
            backup_exists = os.path.exists(backup_path) and os.path.getsize(backup_path) > 0

            # Only backup if original doesn't exist or is empty (preserve true original)
            if not backup_exists:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, backup_path)
                    rospy.loginfo(f"[VelocityPlanner] Backed up original: {filename} -> original/{filename}")
                else:
                    rospy.logwarn(f"[VelocityPlanner] Original file not found: {file_path}")
            else:
                rospy.loginfo(f"[VelocityPlanner] Original already backed up: {filename} (skipping backup)")

        # Save modified ggv
        try:
            np.savetxt(ggv_path, self.ggv, delimiter=',', fmt='%.6f',
                      header='v_mps,ax_max_mps2,ay_max_mps2')
            rospy.loginfo(f"[VelocityPlanner] Saved modified ggv.csv: ax_max={self.a_x_max}, ay_max={self.a_y_max}")
        except Exception as e:
            rospy.logerr(f"[VelocityPlanner] Failed to save ggv.csv: {e}")

        # Save modified ax_max_machines
        try:
            np.savetxt(ax_max_path, self.ax_max_machines, delimiter=',', fmt='%.6f',
                      header='vx_mps,ax_max_machines_mps2')
            rospy.loginfo(f"[VelocityPlanner] Saved modified ax_max_machines.csv: ax_max_motor={self.ax_max_motor}")
        except Exception as e:
            rospy.logerr(f"[VelocityPlanner] Failed to save ax_max_machines.csv: {e}")

        # Save modified b_ax_max_machines
        try:
            np.savetxt(b_ax_max_path, self.b_ax_max_machines, delimiter=',', fmt='%.6f',
                      header='vx_mps,ax_max_machines_mps2')
            rospy.loginfo(f"[VelocityPlanner] Saved modified b_ax_max_machines.csv: ax_max_brake={self.ax_max_brake}")
        except Exception as e:
            rospy.logerr(f"[VelocityPlanner] Failed to save b_ax_max_machines.csv: {e}")

    ### HJ : build track_3d_params from waypoints — same calculation as trajectory_optimizer.py
    def _build_track_3d_params(self, wpnts):
        """
        Extract slope, compute omega via Jacobian, build track_3d_params dict.
        Same as trajectory_optimizer.py but from wpnt mu_rad instead of CSV.
        """
        n = len(wpnts)
        mu = np.array([wp.mu_rad for wp in wpnts])
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        s = np.array([wp.s_m for wp in wpnts])

        # check if 3D data exists (mu_rad != 0 for at least some points)
        if np.all(np.abs(mu) < 1e-8):
            return None, mu  # flat track, no 3D params needed

        # slope = mu (already in waypoints)
        slope = mu

        # dmu/ds from slope
        dmu_ds = np.gradient(slope, s)

        # phi: from waypoints z we could estimate, but for no-bank tracks phi ≈ 0
        # if needed, interpolate from CSV — for now use 0
        phi = np.zeros(n)

        # omega via full Euler->body Jacobian J(mu, phi)
        # J = [[1, 0, -sin(mu)], [0, cos(phi), cos(mu)*sin(phi)], [0, -sin(phi), cos(mu)*cos(phi)]]
        # omega = J . [dphi/ds, dmu/ds, dtheta/ds]
        # dtheta/ds = kappa, dphi/ds ≈ 0
        dphi_ds = np.gradient(phi, s)
        dtheta_ds = kappa

        omega_x = dphi_ds - np.sin(mu) * dtheta_ds
        omega_y = np.cos(phi) * dmu_ds + np.cos(mu) * np.sin(phi) * dtheta_ds
        omega_z = -np.sin(phi) * dmu_ds + np.cos(mu) * np.cos(phi) * dtheta_ds

        # d_omega
        d_omega_x = np.gradient(omega_x, s)
        d_omega_y = np.gradient(omega_y, s)
        d_omega_z = np.gradient(omega_z, s)

        track_3d_params = {
            'mu': mu,
            'phi': phi,
            'omega_x': omega_x,
            'omega_y': omega_y,
            'omega_z': omega_z,
            'd_omega_x': d_omega_x,
            'd_omega_y': d_omega_y,
            'd_omega_z': d_omega_z,
            'dmu_ds': dmu_ds,
            'h': self.h_cog,
        }

        return track_3d_params, slope

    def wpnts_callback(self, msg):
        wpnts = msg.wpnts

        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        el_lengths = 0.1 * np.ones(len(kappa))

        ### HJ : build 3D track params from waypoint mu_rad
        track_3d_params, slope = self._build_track_3d_params(wpnts)

        vx_profile = calc_vel_profile(ggv=self.ggv,
                                      ax_max_machines=self.ax_max_machines,
                                      b_ax_max_machines=self.b_ax_max_machines,
                                      v_max=self.v_max,
                                      kappa=kappa,
                                      el_lengths=el_lengths,
                                      closed=True,
                                      filt_window=self.filt_window,
                                      dyn_model_exp=self.dyn_model_exp,
                                      drag_coeff=self.drag_coeff,
                                      m_veh=self.m_veh,
                                      slope=slope,
                                      track_3d_params=track_3d_params)

        for i in range(len(vx_profile)):
            wpnts[i].vx_mps = vx_profile[i]

        vx_profile_opt_cl = np.append(vx_profile, vx_profile[0])
        ax_profile = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                         el_lengths=el_lengths,
                                                         eq_length_output=False)

        for i in range(len(ax_profile)):
            wpnts[i].ax_mps2 = ax_profile[i]

        msg.wpnts = wpnts
        print("NEW 3D Vel Profile Pub")
        self.glb_wpnts_pub.publish(msg)

    def smart_static_wpnts_callback(self, msg):
        """Callback for smart_static_otwpnts topic - calculate velocity profile with 3D correction"""
        wpnts = msg.wpnts

        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        el_lengths = 0.1 * np.ones(len(kappa))

        ### HJ : build 3D track params from waypoint mu_rad
        track_3d_params, slope = self._build_track_3d_params(wpnts)

        vx_profile = calc_vel_profile(ggv=self.ggv,
                                      ax_max_machines=self.ax_max_machines,
                                      b_ax_max_machines=self.b_ax_max_machines,
                                      v_max=self.v_max,
                                      kappa=kappa,
                                      el_lengths=el_lengths,
                                      closed=True,
                                      filt_window=self.filt_window,
                                      dyn_model_exp=self.dyn_model_exp,
                                      drag_coeff=self.drag_coeff,
                                      m_veh=self.m_veh,
                                      slope=slope,
                                      track_3d_params=track_3d_params)

        for i in range(len(vx_profile)):
            wpnts[i].vx_mps = vx_profile[i]

        vx_profile_opt_cl = np.append(vx_profile, vx_profile[0])
        ax_profile = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                         el_lengths=el_lengths,
                                                         eq_length_output=False)

        for i in range(len(ax_profile)):
            wpnts[i].ax_mps2 = ax_profile[i]

        msg.wpnts = wpnts
        msg.header.stamp = rospy.Time.now()
        print("NEW 3D Smart Static Vel Profile Pub")
        self.smart_static_wpnts_pub.publish(msg)




if __name__ == '__main__':
    rospy.init_node("global_velplanner_3d")
    simple_mux = VelocityPlanner()
    rospy.spin()
