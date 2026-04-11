#!/usr/bin/env python3

import logging

import yaml
import rospkg
import os
import copy

import numpy as np
from steering_lookup.lookup_steer_angle import LookupSteerAngle

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ===== HJ ADDED: Global flag for differential trailing control for static sector obstacles =====

DISABLE_TRAILING_CONTROL_FOR_STATIC_SECTOR = False

# Speed multiplier for static sector obstacles when trailing control is disabled
# 1.0 = use global_speed as-is (no deceleration)
# 0.8 = use 80% of global_speed (slight deceleration)
# Values < 1.0 allow tuning deceleration amount
STATIC_SECTOR_SPEED_MULTIPLIER = 1.0
# ===== HJ ADDED END =====

class Controller:
    """This class implements a MAP controller for autonomous driving.
    Input and output topics are managed by the controller manager
    """
 
    def __init__(self,
                t_clip_min,
                t_clip_max,
                m_l1,
                q_l1,
 
                curvature_factor,
                
                KP,
                KI,
                KD,
                heading_error_thres,
                steer_gain_for_speed,
 
                future_constant,
 
                speed_lookahead,
                lat_err_coeff,
                acc_scaler_for_steer,
                dec_scaler_for_steer,
                start_scale_speed,
                end_scale_speed,
                downscale_factor,
                speed_lookahead_for_steer,
 
                trailing_gap,
                trailing_vel_gain,
                trailing_p_gain,
                trailing_i_gain,
                trailing_d_gain,
                blind_trailing_speed,
 
                loop_rate,
                LUT_name,
                wheelbase,   
 
                speed_factor_for_lat_err,
                speed_factor_for_curvature,
                ctrl_algo,
 
                speed_diff_thres,
                start_speed,
                start_curvature_factor,
 
                AEB_thres,
 
                converter,
 
                logger_info = logging.info,
                logger_warn = logging.warn,
            ):

        # Parameters from manager
        self.t_clip_min = t_clip_min
        self.t_clip_max = t_clip_max
        self.m_l1 = m_l1
        self.q_l1 = q_l1
        self.speed_lookahead = speed_lookahead
        self.lat_err_coeff = lat_err_coeff
        self.acc_scaler_for_steer = acc_scaler_for_steer
        self.dec_scaler_for_steer = dec_scaler_for_steer
        self.start_scale_speed = start_scale_speed
        self.end_scale_speed = end_scale_speed
        self.downscale_factor = downscale_factor
        self.speed_lookahead_for_steer = speed_lookahead_for_steer
 
        self.predict_pub = rospy.Publisher("/controller_prediction/markers", MarkerArray, queue_size=10)
 
        # L1 dist calc param
        self.curvature_factor = curvature_factor
 
        self.speed_factor_for_lat_err = speed_factor_for_lat_err
        self.speed_factor_for_curvature = speed_factor_for_curvature
 
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.heading_error_thres = heading_error_thres
        self.steer_gain_for_speed = steer_gain_for_speed
 
        self.future_constant = future_constant
 
        self.trailing_gap = trailing_gap
        self.trailing_vel_gain = trailing_vel_gain
        self.trailing_p_gain = trailing_p_gain
        self.trailing_i_gain = trailing_i_gain
        self.trailing_d_gain = trailing_d_gain
        self.blind_trailing_speed = blind_trailing_speed
 
        self.loop_rate = loop_rate
        self.LUT_name = LUT_name
        self.AEB_thres = AEB_thres
        self.converter = converter
 
        # Parameters in the controller
        self.curr_steering_angle = 0
        self.idx_nearest_waypoint = None # index of nearest waypoint to car
        self.track_length = None
        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0
        self.trailing_command = 2
        self.speed_command = None
        self.curvature_waypoints = 0
        self.current_steer_command = 0
        self.yaw_rate = 0
                
        self.logger_info = logger_info
        self.logger_warn = logger_warn
 
        self.ctrl_algo = ctrl_algo
 
        self.speed_diff_thres = speed_diff_thres
        self.start_speed = start_speed
        self.start_curvature_factor = start_curvature_factor
 
        self.steer_lookup = LookupSteerAngle(self.LUT_name, logger_info)
        self.wheelbase = wheelbase
        
        self.start_mode = False
        self.future_lat_err = 0.0
        self.future_lat_e_norm = 0.0
        self.lat_acc = 0.0
        self.boost_mode = False

        ### HJ : lateral correction params (updated via dyn_reconfigure)
        lat_mode_int = rospy.get_param('L1_controller/lat_correction_mode', 0)
        self.lat_correction_mode = ['none', 'stanley', 'predictive'][lat_mode_int]
        self.lat_K_stanley = rospy.get_param('L1_controller/lat_K_stanley', 1.5)
        self.lat_pred_horizon = rospy.get_param('L1_controller/lat_pred_horizon', 0.3)
        self.lat_pred_alpha = rospy.get_param('L1_controller/lat_pred_alpha', 0.3)
        rospy.loginfo(f"[Controller] Lateral correction mode: {self.lat_correction_mode}")
        self.speed_ff_gain_accel = rospy.get_param('L1_controller/speed_ff_gain_accel', 0.0)
        self.speed_ff_gain_brake = rospy.get_param('L1_controller/speed_ff_gain_brake', 0.0)
        self.ff_accel_lookahead = rospy.get_param('L1_controller/ff_accel_lookahead', 0.0)
        self.ff_brake_lookahead = rospy.get_param('L1_controller/ff_brake_lookahead', 0.0)

        ### HJ : friction-ellipse accel limiter params
        self.accel_limiter_enabled = rospy.get_param('L1_controller/accel_limiter_enabled', True)
        self.accel_lim_ax_max = rospy.get_param('L1_controller/accel_lim_ax_max', 5.0)
        self.accel_lim_ay_max = rospy.get_param('L1_controller/accel_lim_ay_max', 4.5)
        ### HJ : end

        ### HJ : yaw rate feedback (oversteer/understeer compensation)
        self.K_yr = rospy.get_param('L1_controller/K_yr', 0.0)
        ### HJ : end

        ### HJ : GP steering correction
        self.gp_steer_enabled = False
        self.gp_steer_model = None
        self._load_gp_model()
        self._gp_reload_counter = 0
        ### HJ : end
        self.future_position_z = 0.0  ### HJ : z from track spline for future position

    def main_loop(self, state, position_in_map, waypoint_array_in_map, speed_now, opponent, position_in_map_frenet, acc_now, track_length):
        ### HJ : GP hot-reload check (~every 2s at 40Hz)
        self._gp_reload_counter += 1
        if self._gp_reload_counter >= 80:
            self._gp_reload_counter = 0
            self._try_reload_gp()
        ### HJ : end

        # Updating parameters from manager
        self.state = state
        self.position_in_map = position_in_map
 
        #-------------------------------Future Position-----------------------------
        self.future_position = np.zeros((1,3))
        #-------------------------------Future Position-----------------------------
 
 
        self.waypoint_array_in_map = waypoint_array_in_map
        self.speed_now = speed_now
        self.opponent = opponent
        self.position_in_map_frenet = position_in_map_frenet
        self.acc_now = acc_now
        self.track_length = track_length
 
        ## PREPROCESS ##
        # speed vector
        yaw = self.position_in_map[0, 2]
 
        v = [np.cos(yaw)*self.speed_now, np.sin(yaw)*self.speed_now]
        
        #-------------------------------Future Position-----------------------------
 
        self.future_position = self.calc_future_position(self.future_constant)
        
        #-------------------------------Future Position-----------------------------
 
        self.idx_nearest_waypoint = self.nearest_waypoint(self.position_in_map[0, :2], self.waypoint_array_in_map[:, :2])
        
        # if all waypoints are equal set self.idx_nearest_waypoint to 0
        if np.isnan(self.idx_nearest_waypoint):
            self.idx_nearest_waypoint = 0
        
        if len(self.waypoint_array_in_map[self.idx_nearest_waypoint:]) > 2:
            # calculate curvature of global optimizer waypoints
            # ===== HJ MODIFIED: Add safety check for empty slice =====
            curvature_slice = self.waypoint_array_in_map[self.idx_nearest_waypoint+10:self.idx_nearest_waypoint+20,6]
            if len(curvature_slice) > 0:
                self.curvature_waypoints = np.mean(abs(curvature_slice))
            else:
                # Fallback: use current waypoint curvature
                if len(self.waypoint_array_in_map) > self.idx_nearest_waypoint:
                    self.curvature_waypoints = abs(self.waypoint_array_in_map[self.idx_nearest_waypoint, 6])
                else:
                    self.curvature_waypoints = 0.0
            # ===== HJ MODIFIED END =====
                    
        # calculate future lateral error and future lateral error norm
 
        self.future_lat_e_norm, self.future_lat_err = self.calc_future_lateral_error_norm()
 
        ### LONGITUDINAL CONTROL ###
        
        #-----------------------------------------Future-------------------------------------------
        self.speed_command = self.calc_speed_command(v, self.future_lat_e_norm)
        #-----------------------------------------Future-------------------------------------------
 
        self.speed_command = self.speed_adjust_heading(self.speed_command)
 
        # POSTPROCESS for acceleration/speed decision
 
        if self.speed_command is not None:
            speed = max(self.speed_command, 0)
            acceleration = 0
            jerk = 0
 
        else:
            speed = 0
            jerk = 0
            acceleration = 0                
            self.logger_warn("[Controller] speed was none")
            
        ### LATERAL CONTROL ###
 
        steering_angle = None
        self.future_idx_nearest_waypoint = self.nearest_waypoint(self.future_position[0, :2], self.waypoint_array_in_map[:, :2])
 
        #-----------------------------------------Future-------------------------------------------
        L1_point, L1_distance = self.calc_future_L1_point(self.future_lat_err)
        #-----------------------------------------Future-------------------------------------------
        
        if L1_point.any() is not None:
 
            #-----------------------------------------Future-------------------------------------------
            steering_angle = self.calc_steering_angle_for_future(L1_point, L1_distance, yaw, self.future_lat_e_norm, v)
            #-----------------------------------------Future-------------------------------------------
 
            self.current_steer_command = steering_angle
 
        else:
            raise Exception("L1_point is None")
 
        speed = self.AEB_for_weird_local_wpnt(speed)
 
        return speed, acceleration, jerk, steering_angle, L1_point, L1_distance, self.idx_nearest_waypoint, self.curvature_waypoints, self.future_position

    def AEB_for_weird_local_wpnt(self, speed):
        dists = np.linalg.norm(self.waypoint_array_in_map[:, :2] - self.position_in_map[0, :2], axis=1)
        local_wpnt_dist = np.min(dists)
    
        if local_wpnt_dist >= self.AEB_thres:
            return 2.0
        else :
            return speed

    #-------------------------HJ Emergency Editing--------------------------
    # def AEB_for_weird_local_wpnt(self, speed):
    #     dists = np.linalg.norm(self.waypoint_array_in_map[:, :2] - self.position_in_map[0, :2], axis=1)
    #     local_wpnt_dist = np.min(dists)
    #
    #     if local_wpnt_dist >= self.AEB_thres:
    #         return speed
    #     else :
    #         return speed
    #-------------------------HJ Emergency Editing--------------------------

    def calc_steering_angle_for_future(self, future_L1_point, L1_distance, yaw, furture_lat_e_norm, v):
        """
        The purpose of this function is to calculate the steering angle based on the L1 point, desired lateral acceleration and velocity
 
        Inputs:
            future_L1_point: future_L1_point in frenet coordinates at L1 distance in front of the car
            L1_distance: distance of the L1 point to the car
            yaw: yaw angle of the car
            furture_lat_e_norm: future normed lateral error
            v : future speed vector
 
        Returns:
            steering_angle: calculated steering angle
 
        
        """
        marks = MarkerArray()
        for i in range(1):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.type = mrk.SPHERE
            mrk.scale.x = 0.3
            mrk.scale.y = 0.3
            mrk.scale.z = 0.3
            mrk.color.a = 1.0
            mrk.color.b = 1.0
 
            mrk.id = i
            mrk.pose.position.x = self.future_position[0, 0]
            mrk.pose.position.y = self.future_position[0, 1]
            ### HJ : use spline-interpolated z for 3D marker
            mrk.pose.position.z = self.future_position_z
            mrk.pose.orientation.w = 1
            marks.markers.append(mrk)
 
            
        self.predict_pub.publish(marks)
 
        if (self.state == "TRAILING") and (self.opponent is not None):
            speed_la_for_lu = self.speed_now
        else:
            adv_ts_st = self.speed_lookahead_for_steer
            la_position_steer = [self.future_position[0, 0] + v[0]*adv_ts_st, self.future_position[0, 1] + v[1]*adv_ts_st]
            idx_future_la_steer = self.nearest_waypoint(la_position_steer, self.waypoint_array_in_map[:, :2])
            speed_la_for_lu = self.waypoint_array_in_map[idx_future_la_steer, 3]
            
        speed_for_lu = self.speed_adjust_lat_err(speed_la_for_lu, furture_lat_e_norm)
 
        Future_L1_vector = np.array([future_L1_point[0] - self.future_position[0, 0], future_L1_point[1] - self.future_position[0, 1]])
 
        if np.linalg.norm(Future_L1_vector) == 0:
            self.logger_warn("[Controller] norm of L1 vector was 0, eta is set to 0")
            eta = 0
        else:
            eta = np.arcsin(np.dot([-np.sin(yaw), np.cos(yaw)], Future_L1_vector)/np.linalg.norm(Future_L1_vector))
            
        if self.ctrl_algo == 'MAP':
            if L1_distance == 0 or np.sin(eta) == 0:
                self.lat_acc = 0
                self.logger_warn("[Controller] L1 * np.sin(eta), lat_acc is set to 0")
            else:
                self.lat_acc = 2*speed_for_lu**2 / L1_distance * np.sin(eta)
                
            steering_angle = self.steer_lookup.lookup_steer_angle(self.lat_acc, speed_for_lu)
 
        elif self.ctrl_algo == 'PP':
            steering_angle = np.arctan(2*self.wheelbase*np.sin(eta)/L1_distance)
 
        else :
            rospy.logwarn(f"Wrong control algorithm({self.ctrl_algo}) selected!!")
 
        dt = 1.0 / self.loop_rate  
        
        #-------------------------Steering Scaling-----------------------------
 
        # modifying steer based on heading
 
        steering_angle += self.compute_future_heading_correction(Future_L1_vector, yaw, dt, self.speed_now)

        ### HJ : lateral error correction modes (Stanley / Model-predictive)
        signed_d = self.get_signed_lateral_error()
        steering_angle = self.apply_lateral_correction(steering_angle, signed_d, yaw)
        ### HJ : end

        # modifying steer based on acceleration
        #########################################
        steering_angle = self.acc_scaling(steering_angle)
        #########################################
        
        # modifying steer based on speed
 
        steering_angle = self.speed_steer_scaling(steering_angle, speed_for_lu)
        
        # modifying steer based on velocity
        
        steering_angle *= np.clip(1 + (self.speed_now/10), 1, self.steer_gain_for_speed)
 
        # modifying steer based on lateral error
 
        steering_angle = self.steer_scaling_for_lat_err(steering_angle, self.future_lat_err)

        ### HJ : yaw rate feedback — compensate oversteer/understeer
        if self.K_yr > 0 and abs(self.speed_now) > 0.5:
            expected_yr = self.speed_now * np.tan(steering_angle) / self.wheelbase
            yr_error = expected_yr - self.yaw_rate  # >0: understeer, <0: oversteer
            steering_angle += self.K_yr * yr_error
        ### HJ : end

        ### HJ : GP steering correction
        if self.gp_steer_enabled and self.gp_steer_model is not None:
            steering_angle = self._apply_gp_correction(steering_angle, yaw)
        ### HJ : end

        #-------------------------Steering Scaling-----------------------------

        # limit change of steering angle
        threshold = 0.4
        if abs(steering_angle - self.curr_steering_angle) > threshold:
            self.logger_info(f"steering angle clipped")
        steering_angle = np.clip(steering_angle, self.curr_steering_angle - threshold, self.curr_steering_angle + threshold)
        steering_angle = np.clip(steering_angle,-0.53,0.53)
        
        #-------------------------0329 HJ-----------------------------
        # For HOBAO
        # steering_angle = np.clip(steering_angle,-0.6632,0.6632)
        #-------------------------0329 HJ-----------------------------


        # ===== HJ ADDED: Final NaN check before returning =====
        if np.isnan(steering_angle):
            rospy.logerr_throttle(1.0, "[Controller] NaN in steering_angle, using previous value")
            steering_angle = self.curr_steering_angle if not np.isnan(self.curr_steering_angle) else 0.0
        # ===== HJ ADDED END =====

        self.curr_steering_angle = steering_angle
 
        return steering_angle
 
    def calc_future_L1_point(self, future_lateral_error):
 
        # calculate future L1 guidance
 
        if self.speed_now<2.0:
 
            speed = np.clip(self.speed_command , self.speed_now - 1, self.speed_now + 1)
            speed_scaler = self.m_l1 * speed
 
        else:
 
            speed_scaler = self.m_l1 * self.speed_now
            
        if self.state == "START":
            curvature_scaler = self.start_curvature_factor*self.curvature_waypoints
        else :
            curvature_scaler = self.curvature_factor*self.curvature_waypoints*self.speed_now*self.speed_now
 
        L1_distance = (speed_scaler - curvature_scaler) + self.q_l1
   
        # clip lower bound to avoid ultraswerve when far away from mincurv
        lower_bound = max(self.t_clip_min, np.sqrt(2)*future_lateral_error)
        
        L1_distance = np.clip(L1_distance, lower_bound, self.t_clip_max)
 
        future_L1_point = self.waypoint_at_distance_before_car(L1_distance, self.waypoint_array_in_map, self.future_idx_nearest_waypoint)
 
        return future_L1_point, L1_distance
    
    def calc_speed_command(self, v, lat_e_norm):
        """
        The purpose of this function is to isolate the speed calculation from the main control_loop
        
        Inputs:
            v: speed vector
            lat_e_norm: normed lateral error
            curvature_waypoints: -
        Returns:
            speed_command: calculated and adjusted speed, which can be sent to mux
        """
 
        # lookahead for speed (speed delay incorporation by propagating position)
        adv_ts_sp = self.speed_lookahead
        offset = 2
        la_position = [self.position_in_map[0, 0] + v[0]*adv_ts_sp, self.position_in_map[0, 1] + v[1]*adv_ts_sp]
        idx_la_position = self.nearest_waypoint(la_position, self.waypoint_array_in_map[:, :2])
        idx_la_position = np.clip(idx_la_position + offset, 0, len(self.waypoint_array_in_map) -1)
        global_speed = self.waypoint_array_in_map[idx_la_position, 3]
        cur_speed = self.speed_now
 
        if cur_speed < 0:
            cur_speed = 0
 
        if (self.state == "START"
            and self.boost_mode
            and self.waypoint_array_in_map[0,8] > 0):
            if (global_speed-cur_speed) > 0:
                global_speed = self.start_speed
            elif self.cur_state_speed - cur_speed > 0:
                self.cur_state_speed -= self.speed_diff_thres *(self.cur_state_speed - cur_speed)
                global_speed = self.cur_state_speed
            else:
                self.boost_mode = False
        else:
            self.boost_mode = False
 
        # ===== HJ MODIFIED: Differential trailing control for static sector obstacles =====
        if((self.state == "TRAILING") and (self.opponent is not None)): #Trailing controller
            # Check if opponent is static obstacle in static sector
            is_static_in_static_sector = (len(self.opponent) > 5 and
                                         self.opponent[3] and  # is_static
                                         self.opponent[5])     # in_static_obs_sector

            # Apply trailing control based on flag and obstacle type
            if is_static_in_static_sector and DISABLE_TRAILING_CONTROL_FOR_STATIC_SECTOR:
                # Static obstacle in static sector + flag disabled → NO trailing control
                # Use global speed with tunable multiplier (allows partial deceleration)
                self.trailing_speed = global_speed * STATIC_SECTOR_SPEED_MULTIPLIER
                self.i_gap = 0
                speed_command = global_speed * STATIC_SECTOR_SPEED_MULTIPLIER
            else:
                # Dynamic obstacle OR flag enabled → normal trailing control
                speed_command = self.trailing_controller(global_speed)
        else:
            self.trailing_speed = global_speed
            self.i_gap = 0
            speed_command = global_speed
        # ===== HJ MODIFIED END =====
 
        speed_command = self.speed_adjust_lat_err(speed_command, lat_e_norm)

        ### HJ : acceleration feedforward — independent lookahead & gain for accel/brake
        # accel lookahead: 0 → fall back to speed_lookahead (idx_la_position)
        if self.ff_accel_lookahead > 0:
            la_acc = [self.position_in_map[0, 0] + v[0]*self.ff_accel_lookahead,
                      self.position_in_map[0, 1] + v[1]*self.ff_accel_lookahead]
            idx_ff_accel = self.nearest_waypoint(la_acc, self.waypoint_array_in_map[:, :2])
            idx_ff_accel = np.clip(idx_ff_accel + offset, 0, len(self.waypoint_array_in_map) - 1)
        else:
            idx_ff_accel = idx_la_position

        # brake lookahead: 0 → fall back to speed_lookahead (idx_la_position)
        if self.ff_brake_lookahead > 0:
            la_brk = [self.position_in_map[0, 0] + v[0]*self.ff_brake_lookahead,
                      self.position_in_map[0, 1] + v[1]*self.ff_brake_lookahead]
            idx_ff_brake = self.nearest_waypoint(la_brk, self.waypoint_array_in_map[:, :2])
            idx_ff_brake = np.clip(idx_ff_brake + offset, 0, len(self.waypoint_array_in_map) - 1)
        else:
            idx_ff_brake = idx_la_position

        ax_accel = self.waypoint_array_in_map[idx_ff_accel, 8]
        ax_brake = self.waypoint_array_in_map[idx_ff_brake, 8]

        if ax_accel >= 0 and self.speed_ff_gain_accel > 0:
            speed_command += self.speed_ff_gain_accel * ax_accel
        if ax_brake < 0 and self.speed_ff_gain_brake > 0:
            speed_command += self.speed_ff_gain_brake * ax_brake
        ### HJ : end

        ### HJ : friction-ellipse based accel limiter
        # 가속할 때만 적용: 현재 횡가속도를 고려하여 남은 종가속 여유분 이내로 클리핑
        # 마찰 타원: (ay/ay_max)^2 + (ax/ax_max)^2 <= 1
        # 감속/정속은 건드리지 않음
        if self.accel_limiter_enabled and speed_command > cur_speed:
            kappa = self.curvature_waypoints
            ay = cur_speed ** 2 * kappa
            ay_ratio = min(abs(ay) / self.accel_lim_ay_max, 1.0)
            ax_available = self.accel_lim_ax_max * np.sqrt(max(0.0, 1.0 - ay_ratio ** 2))
            dt = 1.0 / self.loop_rate
            v_max_next = cur_speed + ax_available * dt
            speed_command = min(speed_command, v_max_next)
        ### HJ : end

        return speed_command
    
    def trailing_controller(self, global_speed):
        """
        Adjust the speed of the ego car to trail the opponent at a fixed distance
        Inputs:
            speed_command: velocity of global raceline
            self.opponent: frenet s position and vs velocity of opponent
            self.position_in_map_frenet: frenet s position and vs veloctz of ego car
        Returns:
            trailing_command: reference velocity for trailing
        """
 
        self.gap = (self.opponent[0] - self.position_in_map_frenet[0])%self.track_length # gap to opponent
        self.gap_actual = self.gap
        self.gap_should = self.trailing_vel_gain * self.speed_now + self.trailing_gap
 
        self.gap_error = self.gap_should - self.gap_actual
        self.v_diff =  self.position_in_map_frenet[2] - self.opponent[2]
        self.i_gap = np.clip(self.i_gap + self.gap_error/self.loop_rate, -10, 10)
    
        p_value = self.gap_error * self.trailing_p_gain
        d_value = self.v_diff * self.trailing_d_gain
        i_value = self.i_gap * self.trailing_i_gain
 
        self.trailing_command = np.clip(self.opponent[2] - p_value - i_value - d_value, 0, global_speed)
        if not self.opponent[4] and self.gap_actual > self.gap_should:
            self.trailing_command = max(self.blind_trailing_speed, self.trailing_command)
 
        return self.trailing_command
    
 
    def distance(self, point1, point2):
        return np.linalg.norm(point2 - point1)
 
    def acc_scaling(self, steer):
        """
        Steer scaling based on acceleration
        increase steer when accelerating
        decrease steer when decelerating
 
        Returns:
            steer: scaled steering angle based on acceleration
        """
        
        if self.start_mode:
            return steer
        
        if np.mean(self.acc_now) >= 1:
            steer *= self.acc_scaler_for_steer
        elif np.mean(self.acc_now) <= -3.0:
            if self.state == "START":
                steer *= 0.7
            else:
                steer *= self.dec_scaler_for_steer
                
        return steer
 
    def speed_steer_scaling(self, steer, speed):
        """
        Steer scaling based on speed
        decrease steer when driving fast
 
        Returns:
            steer: scaled steering angle based on speed
        """
        speed_diff = max(0.1,self.end_scale_speed-self.start_scale_speed) # to prevent division by zero
        factor = 1 - np.clip((speed - self.start_scale_speed)/(speed_diff), 0.0, 1.0) * self.downscale_factor
        steer *= factor
        return steer
 
    ### HJ : lateral error correction ========================================

    def get_signed_lateral_error(self):
        """Get signed lateral error (d) from frenet coordinates.
        Positive d = left of raceline, negative d = right of raceline."""
        try:
            _, d = self.converter.get_frenet_3d(
                np.array([self.future_position[0, 0]]),
                np.array([self.future_position[0, 1]]),
                np.array([self.future_position_z]))
            idx = self.nearest_waypoint(self.future_position[0, :2], self.waypoint_array_in_map[:, :2])
            wpnt_d = self.waypoint_array_in_map[idx, 9] if self.waypoint_array_in_map.shape[1] > 9 else 0.0
            return float(d[0] - wpnt_d)
        except Exception:
            return 0.0

    def apply_lateral_correction(self, steering_angle, signed_d, yaw):
        """Apply lateral error correction based on selected mode."""
        if self.lat_correction_mode == 'stanley':
            return self._stanley_correction(steering_angle, signed_d, yaw)
        elif self.lat_correction_mode == 'predictive':
            return self._predictive_correction(steering_angle, signed_d, yaw)
        else:
            return steering_angle  # 'none' — no correction

    def _stanley_correction(self, steering_angle, signed_d, yaw):
        """Stanley crosstrack correction at front axle (current position).
        ### HJ : front-axle = current pos + wheelbase along yaw
        """
        v = max(self.speed_now, 0.5)

        ### HJ : front-axle d from current position (Stanley original)
        fx = self.position_in_map[0, 0] + self.wheelbase * np.cos(yaw)
        fy = self.position_in_map[0, 1] + self.wheelbase * np.sin(yaw)
        try:
            _, d_front = self.converter.get_frenet_3d(
                np.array([fx]), np.array([fy]),
                np.array([self.future_position_z]))
            idx = self.nearest_waypoint(np.array([fx, fy]), self.waypoint_array_in_map[:, :2])
            wpnt_d = self.waypoint_array_in_map[idx, 9] if self.waypoint_array_in_map.shape[1] > 9 else 0.0
            d_front = float(d_front[0] - wpnt_d)
        except Exception:
            d_front = signed_d
        d_use = d_front

        # ### HJ : future_position 기반 d 사용 시 아래 주석 해제
        # d_use = signed_d

        correction = np.arctan(self.lat_K_stanley * (-d_use) / v)
        return steering_angle + correction

    def _predictive_correction(self, steering_angle, signed_d, yaw):
        """Model-predictive lateral correction using bicycle model.
        1. Predict d_future with current steering (PP + heading corr applied)
        2. Compute delta_optimal that makes d_future = 0
        3. Blend: delta = delta + alpha * (delta_optimal - delta)
        """
        v = max(self.speed_now, 0.5)
        T = self.lat_pred_horizon
        L = self.wheelbase
        alpha = self.lat_pred_alpha

        # Heading error relative to raceline
        idx = self.nearest_waypoint(self.future_position[0, :2], self.waypoint_array_in_map[:, :2])
        wpnt_psi = self.waypoint_array_in_map[idx, 4]  # psi_rad
        heading_err = yaw - wpnt_psi
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi

        # Predict d_future with current steering
        delta_clipped = np.clip(steering_angle, -1.0, 1.0)
        yaw_rate_pred = v * np.tan(delta_clipped) / L
        heading_err_future = heading_err + yaw_rate_pred * T
        d_future = signed_d + v * (np.sin(heading_err) + np.sin(heading_err_future)) / 2.0 * T

        # Solve for delta_optimal: d_future = 0
        # Linearized: d + v*sin(he)*T + 0.5*v^2*cos(he)*tan(delta)/L*T^2 = 0
        denom = 0.5 * v**2 * np.cos(heading_err) * T**2
        if abs(denom) < 1e-6:
            return steering_angle

        tan_delta_opt = -(signed_d + v * np.sin(heading_err) * T) * L / denom
        tan_delta_opt = np.clip(tan_delta_opt, -2.0, 2.0)
        delta_optimal = np.arctan(tan_delta_opt)

        # Blend PP result with model prediction
        return steering_angle + alpha * (delta_optimal - steering_angle)

    ### HJ : end lateral error correction ====================================

    ### HJ : GP steering correction ==========================================

    ### HJ : fixed GP model path + hot-reload
    GP_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'gp_residual', 'models', 'gp_model.pkl')

    def _load_gp_model(self):
        """Init GP model tracking. Actual load happens via _try_reload_gp()."""
        self._gp_model_mtime = 0.0
        self.gp_max_correction = rospy.get_param('L1_controller/gp_max_correction', 0.05)
        self.gp_uncertainty_thres = rospy.get_param('L1_controller/gp_uncertainty_thres', 0.1)
        self._try_reload_gp()

    def _try_reload_gp(self):
        """Reload GP model if file changed. Called periodically."""
        try:
            if not os.path.exists(self.GP_MODEL_PATH):
                return
            mtime = os.path.getmtime(self.GP_MODEL_PATH)
            if mtime <= self._gp_model_mtime:
                return
            import pickle
            with open(self.GP_MODEL_PATH, 'rb') as f:
                self.gp_steer_model = pickle.load(f)
            self._gp_model_mtime = mtime
            rospy.loginfo(f"[Controller] GP model hot-reloaded: {self.GP_MODEL_PATH}")
        except Exception as e:
            rospy.logwarn(f"[Controller] GP reload failed: {e}")

    def _apply_gp_correction(self, steering_angle, yaw):
        """Apply GP steering correction with safety guards."""
        try:
            v = max(self.speed_now, 0.5)
            idx = self.nearest_waypoint(self.position_in_map[0, :2], self.waypoint_array_in_map[:, :2])
            kappa = self.waypoint_array_in_map[idx, 6]
            ax = np.mean(self.acc_now) if hasattr(self, 'acc_now') else 0.0
            state = np.array([[v, steering_angle, kappa, self.yaw_rate, ax]])

            pred, sigma = self.gp_steer_model.predict(state)
            delta_gp = float(pred[0])

            # uncertainty guard: fade out if uncertain
            if sigma is not None and float(sigma[0]) > self.gp_uncertainty_thres:
                delta_gp = 0.0

            # clamp: speed-dependent max correction
            max_corr = self.gp_max_correction / (1.0 + 0.1 * v)
            delta_gp = np.clip(delta_gp, -max_corr, max_corr)

            return steering_angle + delta_gp
        except Exception:
            return steering_angle

    ### HJ : end GP steering correction ======================================

    def steer_scaling_for_lat_err(self, steer, lateral_error):

        if self.start_mode:
            return steer

        factor = np.exp(np.log(2)*lateral_error)

        steer *= factor
        return steer

    def calc_future_lateral_error_norm(self):
        """
        Calculates future lateral error

        Returns:
           future lat_e_norm: normalization of the future lateral error
           future lateral_error: future distance from car's position to nearest waypoint
        """
        # ===== HJ ADDED: NaN safety check =====
        if np.any(np.isnan(self.future_position)):
            rospy.logwarn_throttle(1.0, "[Controller] NaN in future_position, returning 0 for lateral error norm")
            return 0.0, 0.0
        # ===== HJ ADDED END =====

        future_position = self.future_position[0, :2]
        idx_future_local_wpnts = self.nearest_waypoint(future_position, self.waypoint_array_in_map[:, :2])
        # ===== HJ MODIFIED: Use signed d values, take abs() of difference =====
        future_local_wpnts_d = self.waypoint_array_in_map[idx_future_local_wpnts,9]  # Keep sign

        ### HJ : use 3D frenet conversion for lateral error (z-aware nearest search)
        try:
            future_potision_s, future_position_d = self.converter.get_frenet_3d(
                np.array([self.future_position[0,0]]),
                np.array([self.future_position[0,1]]),
                np.array([self.future_position_z]))
            future_position_d = future_position_d[0]  # Keep sign
            future_lat_err = abs(future_position_d - future_local_wpnts_d)  # Distance between car and waypoint
        except (ValueError, Exception) as e:
            rospy.logwarn_throttle(1.0, f"[Controller] Frenet conversion failed: {e}, returning 0 for lateral error norm")
            return 0.0, 0.0
        ### HJ : end
        # ===== HJ MODIFIED END =====

        max_lat_e = 1
        min_lat_e = 0.
        lat_e_clip = np.clip(future_lat_err, a_min=min_lat_e, a_max=max_lat_e)
        lat_e_norm = ((lat_e_clip - min_lat_e) / (max_lat_e - min_lat_e))
        return lat_e_norm, future_lat_err
 
    def speed_adjust_lat_err(self, global_speed, lat_e_norm):
        """
        Reduce speed from the global_speed based on the lateral error
        and curvature of the track. lat_e_coeff scales the speed reduction:
        lat_e_coeff = 0: no account for lateral error
        lat_e_coaff = 1: maximum accounting
 
        Returns:
            global_speed: the speed we want to follow
        """
        # scaling down global speed with lateral error and curvature
        lat_e_coeff = self.lat_err_coeff # must be in [0, 1]
        lat_e_norm *= self.speed_factor_for_lat_err
        curv = np.clip(2*(np.mean(self.curvature_waypoints)/0.8) - 2, a_min = 0.0, a_max = 1.0) # 0.8 ca. max curvature mean
        curv *= self.speed_factor_for_curvature
        global_speed *= (1.0 - lat_e_coeff + lat_e_coeff*np.exp(-lat_e_norm*curv))
        return global_speed
    
    def speed_adjust_heading(self, speed_command):
        """
        Reduce speed from the global_speed based on the heading error.
        If the difference between the map heading and the actual heading
        is larger than 10 degrees, the speed gets scaled down linearly up to 0.5x
        
        Returns:
            global_speed: the speed we want to follow
        """
 
        heading = self.position_in_map[0,2]
        map_heading = self.waypoint_array_in_map[self.idx_nearest_waypoint, 7]
        if abs(heading - map_heading) > np.pi:
            heading_error = 2*np.pi - abs(heading- map_heading)
        else:
            heading_error = abs(heading - map_heading)
 
        if heading_error < self.heading_error_thres*np.pi/180: # 10 degrees error is okay
            return speed_command
        elif heading_error < np.pi/2:
            scaler = 1 - 0.5* heading_error/(np.pi/2)
        else:
            scaler = 0.5
        return speed_command * scaler
                
    def compute_future_heading_correction(self, L1_vector, yaw, dt, speed,
                               alpha=0.1, v_threshold=15.0,
                               use_pid=True, use_filter=True):

        target_heading = np.arctan2(L1_vector[1], L1_vector[0])
        heading_error = target_heading - yaw
        # ===== HJ MODIFIED: Add NaN check before modulo operation =====
        if np.isnan(heading_error):
            rospy.logerr_throttle(1.0, "[Controller] NaN in heading_error, setting to 0")
            heading_error = 0.0
        else:
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        # ===== HJ MODIFIED END =====
 
        if use_filter:
            if not hasattr(self, 'filtered_heading_error'):
                self.filtered_heading_error = heading_error
            self.filtered_heading_error = alpha * heading_error + (1 - alpha) * self.filtered_heading_error
            heading_error = self.filtered_heading_error
 
        if speed < v_threshold:
            dynamic_gain = self.KP * (speed / v_threshold)
        else:
            dynamic_gain = self.KP
 
 
        if self.state == "OVERTAKE":
            dynamic_gain *= 0.65
 
        if not hasattr(self, 'heading_error_integral'):
            self.heading_error_integral = 0.0
        if not hasattr(self, 'prev_heading_error'):
            self.prev_heading_error = heading_error
 
        if use_pid:
            self.heading_error_integral += heading_error * dt
            derivative = (heading_error - self.prev_heading_error) / dt if dt > 0 else 0.0
            self.prev_heading_error = heading_error
 
            correction = dynamic_gain * heading_error + self.KI * self.heading_error_integral + self.KD * derivative
        else:
            correction = dynamic_gain * heading_error
 
        return correction
    
    def calc_future_position(self, T):
        """
        Predicts the future vehicle state (position and heading) T seconds ahead
        based on the current vehicle state and updates self.position_in_map[0].
        
        Inputs:
            T: Prediction time (seconds), e.g., 0.25
            
        Assumes the following variables exist in self:
            self.position_in_map : 2D array with the first row containing [x, y, psi]
            self.speed_now       : Current vehicle speed (v)
            self.current_steer_command : Current steering input (delta)
            self.yaw_rate        : Current yaw rate from the IMU (rad/s)
            self.wheelbase       : Vehicle wheelbase (distance between front and rear axles)
        """
 
        x_current = self.position_in_map[0, 0]

        # Extract current state
        x_current = self.position_in_map[0, 0]
        y_current = self.position_in_map[0, 1]
        psi_current = self.position_in_map[0, 2]
        v = self.speed_now
        delta = self.current_steer_command  # Steering input

        # ===== HJ ADDED: Input validation =====
        if np.any(np.isnan([x_current, y_current, psi_current, v, delta])):
            nan_sources = []
            if np.isnan(x_current): nan_sources.append(f"x={x_current}")
            if np.isnan(y_current): nan_sources.append(f"y={y_current}")
            if np.isnan(psi_current): nan_sources.append(f"psi={psi_current}")
            if np.isnan(v): nan_sources.append(f"v={v}")
            if np.isnan(delta): nan_sources.append(f"delta={delta}")
            rospy.logerr_throttle(1.0, f"[Controller] NaN in calc_future_position: {', '.join(nan_sources)}")

            future_position = np.zeros((1, 3))
            future_position[0, 0] = x_current if not np.isnan(x_current) else 0.0
            future_position[0, 1] = y_current if not np.isnan(y_current) else 0.0
            future_position[0, 2] = psi_current if not np.isnan(psi_current) else 0.0
            return future_position
        # ===== HJ ADDED END =====

        # Vehicle geometry parameters.
        # Here, L_f and L_r are assumed to be 52% and 48% of the total wheelbase respectively.
        L_total = self.wheelbase
        L_f = 0.52 * L_total
        L_r = 0.48 * L_total

        # 1. Compute geometric slip angle (basic model)
        # ===== HJ MODIFIED: Clip delta to avoid tan explosion =====
        delta_clipped = np.clip(delta, -np.pi/2 + 0.1, np.pi/2 - 0.1)
        beta_model = np.arctan((L_r / (L_f + L_r)) * np.tan(delta_clipped))
        # ===== HJ MODIFIED END =====

        # ===== HJ MODIFIED: Use model-based prediction only (no IMU fusion) =====
        # 2. Estimate slip angle indirectly using IMU yaw rate data
        # if abs(v) > 2.0:
        #     # If speed is sufficient, estimate slip angle from IMU yaw rate
        #     beta_imu = np.arcsin(np.clip(((L_f + L_r) * self.yaw_rate / v), -1.0, 1.0))
        # else:
        #     beta_imu = beta_model  # Maintain basic model when speed is very low

        # 3. Fuse the geometric and IMU-based slip angles using weighted average
        # lambda_weight = 1.0
        # beta_fused = lambda_weight * beta_model + (1 - lambda_weight) * beta_imu
        beta_fused = beta_model  # Use model-based only

        # 4. Predict future position using the fused slip angle
        future_x = x_current + v * np.cos(psi_current + beta_fused) * T
        future_y = y_current + v * np.sin(psi_current + beta_fused) * T

        ### HJ : estimate future z from track spline (vehicle follows track surface)
        try:
            future_s = self.converter.get_approx_s(np.array([future_x]), np.array([future_y]))
            self.future_position_z = float(self.converter.spline_z(future_s[0]))
        except Exception:
            self.future_position_z = 0.0
        ### HJ : end

        # 5. Predict future heading:
        # Option A: Model-based prediction
        future_psi_model = psi_current + (v / (L_f + L_r)) * np.sin(beta_fused) * T
        # Option B: IMU-based prediction
        # future_psi_imu = psi_current + self.yaw_rate * T
        # Fuse the two heading predictions using a weighted average
        # gamma_weight = 1.0
        # future_psi = gamma_weight * future_psi_model + (1 - gamma_weight) * future_psi_imu
        future_psi = future_psi_model  # Use model-based only
        # ===== HJ MODIFIED END =====
        # Normalize heading to the range [-pi, pi]
        future_psi = np.arctan2(np.sin(future_psi), np.cos(future_psi))
        
        # Update the global state: overwrite self.position_in_map[0] with the future state.
 
        future_position = np.zeros((1,3))
 
        future_position[0,0] = future_x
        future_position[0,1] = future_y
        future_position[0,2] = future_psi
 
        return future_position
        
    def nearest_waypoint(self, position, waypoints):
        """
        Calculates index of nearest waypoint to the car
 
        Returns:
            index of nearest waypoint to the car
        """        
        position_array = np.array([position]*len(waypoints))
        distances_to_position = np.linalg.norm(abs(position_array - waypoints), axis=1)
        return np.argmin(distances_to_position)

    def waypoint_at_distance_before_car(self, distance, waypoints, idx_waypoint_behind_car):
        """
        Calculates the waypoint at a certain frenet distance in front of the car
 
        Returns:
            waypoint as numpy array at a ceratin distance in front of the car
        """
        
        if distance is None:
            distance = self.t_clip_min
        d_distance = distance
 
        ### HJ : use 3D distance for lookahead accumulation on sloped tracks
        waypoints_ahead = waypoints[idx_waypoint_behind_car:]

        ### HJ : x=0, y=1, z=2 — compute 3D segment distances
        if waypoints_ahead.shape[1] > 2:
            deltas = np.diff(waypoints_ahead[:, :3], axis=0)
        else:
            deltas = np.diff(waypoints_ahead[:, :2], axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)

        # Compute cumulative distances
        cum_lengths = np.cumsum(seg_lengths)

        # Find the first index where cumulative distance exceeds lookahead
        idx_offset = min(np.searchsorted(cum_lengths, d_distance), len(waypoints_ahead) - 1)

        ### HJ : return xyz for 3D marker visualization (steering still uses xy)
        return waypoints_ahead[idx_offset, :3]
        ### HJ : end
