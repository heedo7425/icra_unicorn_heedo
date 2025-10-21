#!/usr/bin/env python3
import time
from typing import List, Any, Tuple
import copy
import os
import sys
import csv
import json
import threading
import math
import rospy
import numpy as np
import cv2
import yaml
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import BPoly
from scipy.signal import argrelextrema, savgol_filter
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from dynamic_reconfigure.msg import Config
from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray, BehaviorStrategy
from frenet_converter.frenet_converter import FrenetConverter
import tf.transformations as tf_trans
from grid_filter.grid_filter import GridFilter
import trajectory_planning_helpers as tph
from rospkg import RosPack

# Add GB optimizer path to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gb_optimizer/src'))
from global_racetrajectory_optimization.trajectory_optimizer import trajectory_optimizer

class ObstacleSpliner:
    """
    This class implements a ROS node that performs splining around obstacles.

    It subscribes to the following topics:
        - `/tracking/obstacles`: Subscribes to the obstacle array.
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/global_waypoints`: Subscribes to global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.
    
    The node publishes the following topics:
        - `/planner/avoidance/markers`: Publishes real-time spline markers (pre-fix mode).
        - `/planner/avoidance/otwpnts`: Publishes splined waypoints.
        - `/planner/avoidance/considered_OBS`: Publishes markers for the closest obstacle.
        - `/planner/avoidance/propagated_obs`: Publishes markers for the propagated obstacle.
        - `/planner/avoidance/latency`: Publishes the latency of the spliner node. (only if measuring is enabled)
        - `/planner/avoidance/smart_markers`: Publishes GB optimizer fixed path markers (post-fix mode).
    """

    def __init__(self):
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        # Initialize the node
        self.name = "obs_spliner_node"
        rospy.init_node(self.name)

        # initialize the instance variable
        # self.obs = ObstacleArray()
        self.obs_in_interest = None
        self.gb_wpnts = WpntArray()
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = 0
        self.cur_d = 0
        self.cur_vs = 0
        self.gb_scaled_wpnts = WpntArray()
        self.lookahead = 10  # in meters [m]
        self.last_switch_time = rospy.Time.now()
        self.last_ot_side = ""
        self.from_bag = rospy.get_param("/from_bag", False)
        self.measuring = rospy.get_param("/measure", False)
        
        self.sampling_dist = rospy.get_param("/sampling_dist", 20.0)
        self.spline_scale = rospy.get_param("/spline_scale", 0.8)
        self.post_min_dist = rospy.get_param("/post_min_dist", 1.5)
        self.post_max_dist = rospy.get_param("/post_max_dist", 5.0)
        self.kernel_size = rospy.get_param("/kernel_size", 4)
        
        # ===== HJ ADDED: Dual grid filters =====
        self.map_filter = GridFilter(map_topic="/map", debug=False)  # Original map
        self.map_filter.set_erosion_kernel_size(self.kernel_size)
        self.map_filter_fixed = None  # Modified map (with obstacles), created after fixed path generation
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Reverse mapping parameter (from global_planner) =====
        self.reverse_mapping = rospy.get_param('/global_planner/reverse_mapping', False)
        rospy.loginfo(f"[{self.name}] Reverse mapping: {self.reverse_mapping}")
        # ===== HJ ADDED END =====

        # ===== HJ EDITED START: Smart static avoidance memory system =====
        # IMPORTANT: Initialize BEFORE subscribing to /tracking/obstacles to avoid AttributeError
        # Phase B: Memory structure - (sector_id, obs_id) tuple keys for individual obstacle tracking
        self.static_obs_memory = {}  # {(sector_id, obs_id): {'s_history': [], 'd_history': [], 's_history_fixed': [], 'd_history_fixed': [], 'obs_count': 0, 'interferes': bool, 'interferes_fixed': bool, 'last_seen': rospy.Time}}
        self.fixed_path_generated = False  # Static flag: Has fixed path been generated? (Once True, stays True)
        self.use_fixed_path = False  # Dynamic flag: Currently using fixed path? (Can toggle based on obstacles)
        self.fixed_path_generating = False  # Flag to track if generation is in progress
        self.fixed_path_wpnts = OTWpntArray()
        self.fixed_path_markers = MarkerArray()  # Markers for fixed path visualization
        self.fixed_converter = None  # FrenetConverter for fixed path (created when path is generated)
        self.gb_revert_threshold_m = 1.0  # All obstacles must be >1.0m away (lateral) to revert to GB (noise tolerance)
        # ===== HJ ADDED: Low-rate publishing for fixed path =====
        self.fixed_path_last_pub_time = rospy.Time(0)
        self.fixed_path_pub_rate = 2.0  # seconds (0.5Hz) - static data doesn't need high rate
        # ===== HJ ADDED END =====
        self.min_stable_observations = 5  # 0.25 seconds at 20Hz (enough for static obs with std check)
        self.position_std_threshold = 0.05  # 5cm position stability threshold
        self.min_sectors_with_stable_obs = 2  # Need stable obstacles in 2 DIFFERENT sectors
        self.max_history_length = 100  # Keep recent 100 observations (5 seconds)

        # ===== HJ ADDED: Adaptive memory timeout =====
        self.memory_timeout_sec = 15.0  # Initial: 15 seconds (before fixed path)
        self.memory_timeout_min = 1.0   # Minimum: 1 second (after fixed path stabilizes)
        self.memory_timeout_decay_interval = 3.0  # Decay every 5 seconds after fixed path
        self.memory_timeout_decay_amount = 0.0    # Reduce by 1 second each interval
        self.last_timeout_decay_time = None  # Track last decay (set when fixed path generated)
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Load static obstacle sectors for position-based active check =====
        self.static_obs_sectors = {}  # {sector_id: {'s_start': float, 's_end': float, 'static_obs_section': bool}}
        self.track_length = 0.0  # Will be set from GB waypoints
        self._load_static_obs_sectors()

        # Detection zone monitoring: track if vehicle has passed through zone without obstacles
        self.sector_monitoring = {}  # {sector_id: {'in_zone': bool, 'had_obstacles': bool, 'entered_at_s': float}}

        # Sector-specific interference tracking: remember which sectors had interfering obstacles during path generation
        self.sectors_with_interfering_obs = set()  # Set of sector_ids that had interfering obstacles when fixed path was generated
        # ===== HJ ADDED END =====

        # Phase B: Position-based obstacle merging (in case TTL causes ID changes)
        self.use_position_based_merge = True  # Set False if tracking TTL is long enough
        self.position_merge_threshold = 0.05  # 5cm - obstacles closer than this are considered same

        # Phase A: Callback lightweight variables
        self.latest_obstacles = []
        self.obstacles_timestamp = rospy.Time.now()
        self.obstacles_updated = False

        # Phase 3: GB optimizer configuration
        self.obstacle_shift_s_range = 2.0  # [m] shift waypoints within ±2m of obstacle s position
        self.obstacle_wall_margin = 0.1  # [m] margin when adding obstacle perimeter to wall boundary
        self.obstacle_wall_points = 100  # number of points on obstacle perimeter
        self.gb_optimizer_plot = rospy.get_param(f'/{self.name}/gb_optimizer_plot', False)  # Show plots for debugging

        # Get racecar_version and map_name for locating config files (same as mapping.launch)
        self.racecar_version = rospy.get_param('racecar_version', 'FIESTA1')
        self.map_name = rospy.get_param('/map')  # Map name from /map parameter
        self.map_dir = os.path.join(
            os.path.dirname(__file__),
            '../../../stack_master/maps',
            self.map_name
        )

        # Load GB optimizer parameters (loaded from global_planner_params.yaml in launch file)
        self.safety_width = rospy.get_param('~safety_width', 1.2)  # [m] safety width for GB optimizer
        # ===== HJ EDITED END =====

        # Subscribe to the topics
        # ===== HJ EDITED START: Subscribe to tracking obstacles for smart static avoidance =====
        rospy.Subscriber("/tracking/obstacles", ObstacleArray, self.tracking_obs_cb)
        # ===== HJ EDITED END =====
        rospy.Subscriber("/behavior_strategy", BehaviorStrategy, self.behavior_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.state_frenet_cb)
        rospy.Subscriber("/car_state/odom", Odometry, self.state_cb)
        rospy.Subscriber("/global_waypoints", WpntArray, self.gb_cb)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self.gb_scaled_cb)
        # dyn params sub
        self.evasion_dist = 0.65
        self.obs_traj_tresh = 0.3
        self.spline_bound_mindist = 0.2
        self.n_loc_wpnts = 80
        self.width_car = 0.30

        # ===== HJ EDITED START: Params for obstacle interference check (same as state_machine) =====
        self.gb_ego_width_m = rospy.get_param("state_machine/gb_ego_width_m", 0.3)
        self.lateral_width_gb_m = rospy.get_param("state_machine/lateral_width_gb_m", 0.1)
        # ===== HJ EDITED END =====

        if not self.from_bag:
            rospy.Subscriber("/dynamic_spline_tuner_node/parameter_updates", Config, self.dyn_param_cb)
            # ===== HJ EDITED START: Subscribe to state machine dynamic params =====
            rospy.Subscriber("/dyn_statemachine/parameter_updates", Config, self.state_machine_dyn_param_cb)
            # ===== HJ EDITED END =====

        self.mrks_pub = rospy.Publisher("/planner/avoidance/markers", MarkerArray, queue_size=10)
        self.evasion_pub = rospy.Publisher("/planner/avoidance/otwpnts", OTWpntArray, queue_size=10)
        self.closest_obs_pub = rospy.Publisher("/planner/avoidance/considered_OBS", Marker, queue_size=10)
        self.pub_propagated = rospy.Publisher("/planner/avoidance/propagated_obs", Marker, queue_size=10)
        if self.measuring:
            self.latency_pub = rospy.Publisher("/planner/avoidance/latency", Float32, queue_size=10)

        # Smart static avoidance publishers (fixed path)
        self.fixed_path_pub = rospy.Publisher("/planner/avoidance/smart_static_otwpnts", OTWpntArray, queue_size=10)
        self.fixed_path_mrks_pub = rospy.Publisher("/planner/avoidance/smart_markers", MarkerArray, queue_size=10)
        # ===== HJ ADDED: Publisher for use_fixed_path flag =====
        self.use_fixed_path_pub = rospy.Publisher("/planner/avoidance/smart_static_active", Bool, queue_size=10)
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Publisher for obstacles-only map visualization =====
        from nav_msgs.msg import OccupancyGrid
        self.map_with_obs_pub = rospy.Publisher("/map_with_obs", OccupancyGrid, queue_size=1, latch=True)
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Publisher for do_spline samples visualization =====
        self.temp_spline_markers_pub = rospy.Publisher("/planner/avoidance/temp_do_spline_markers", MarkerArray, queue_size=10)
        # ===== HJ ADDED END =====

        self.converter = self.initialize_converter()

        # Set the rate at which the loop runs
        self.rate = rospy.Rate(20)  # Hz

        # ===== HJ DEBUG: Counter for saving do_spline inputs =====
        self._debug_save_counter = 0
        # ===== HJ DEBUG END =====

        # ===== HJ ADDED: Cache for original wall boundaries =====
        self._original_bound_r = None
        self._original_bound_l = None
        # ===== HJ ADDED END =====

    #############
    # CALLBACKS #
    #############
    # ===== HJ EDITED START: Tracking obstacles callback for smart static avoidance =====
    def tracking_obs_cb(self, data: ObstacleArray):
        """
        Callback for /tracking/obstacles topic.
        Phase A: Lightweight callback - only store data, no computation.
        All processing moved to _process_obstacles() called from main loop.
        """
        # Simply store latest data and set flag
        self.latest_obstacles = data.obstacles
        self.obstacles_timestamp = data.header.stamp
        self.obstacles_updated = True
    # ===== HJ PHASE A: Callback lightened =====

    def behavior_cb(self, data: BehaviorStrategy):
        """
        Callback for /behavior_strategy topic.

        NOTE: Commented out for smart static avoidance mode.
        We use tracking_obs_cb instead to see ALL obstacles.
        Keep this for future reference if needed.
        """
        # ===== HJ COMMENTED OUT: Use tracking_obs_cb for smart static avoidance =====
        # if len(data.overtaking_targets)!= 0:
        #     self.obs_in_interest = data.overtaking_targets[0]
        # else:
        #     self.obs_in_interest = None
        pass


    def state_frenet_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

    def state_cb(self, data: Odometry):
        self.cur_x = data.pose.pose.position.x
        self.cur_y = data.pose.pose.position.y
        quat = data.pose.pose.orientation
        q = [quat.x, quat.y, quat.z, quat.w]
        euler = tf_trans.euler_from_quaternion(q)
        self.cur_yaw = euler[2] 

    # Callback for global waypoint topic
    def gb_cb(self, data: WpntArray):
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m
            # ===== HJ ADDED: Set track length for sector checking =====
            self.track_length = self.gb_max_s
            # ===== HJ ADDED END =====

    # Callback for scaled global waypoint topic
    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    # ===== HJ ADDED: Load static obstacle sectors from ROS params =====
    def _load_static_obs_sectors(self):
        """Load static obstacle sectors from ROS parameters (same as detect.cpp)"""
        self.static_obs_sectors.clear()

        n_sectors = rospy.get_param("/static_obs_map_params/n_sectors", 0)
        if n_sectors == 0:
            rospy.logwarn(f"[{self.name}] Static obs sectors param '/static_obs_map_params/n_sectors' not found, defaulting to 0")
            return

        for i in range(n_sectors):
            sector_key = f"/static_obs_map_params/Static_Obs_sector{i}"

            try:
                s_start = rospy.get_param(f"{sector_key}/s_start")
                s_end = rospy.get_param(f"{sector_key}/s_end")
                static_obs_section = rospy.get_param(f"{sector_key}/static_obs_section")

                self.static_obs_sectors[i] = {
                    's_start': s_start,
                    's_end': s_end,
                    'static_obs_section': static_obs_section
                }

                rospy.loginfo(f"[{self.name}] Loaded sector {i}: s_start={s_start:.2f}, s_end={s_end:.2f}, static_obs_section={static_obs_section}")
            except KeyError as e:
                rospy.logwarn(f"[{self.name}] Failed to load sector {i} parameters: {e}")

        rospy.loginfo(f"[{self.name}] Loaded {len(self.static_obs_sectors)} static obstacle sectors")

    def _is_position_in_sector(self, s_position, sector_data):
        """Check if s_position is within sector bounds (handling wrap-around)

        Args:
            s_position: Current s coordinate to check
            sector_data: Dict with 's_start', 's_end', 'static_obs_section'

        Returns:
            bool: True if position is in sector
        """
        if self.track_length == 0.0:
            return False

        # Normalize s to [0, track_length)
        s_normalized = s_position
        while s_normalized < 0:
            s_normalized += self.track_length
        while s_normalized >= self.track_length:
            s_normalized -= self.track_length

        s_start = sector_data['s_start']
        s_end = sector_data['s_end']

        # Check if position is within sector bounds
        if s_start <= s_end:
            # Normal case: sector doesn't wrap around
            in_sector = (s_normalized >= s_start and s_normalized <= s_end)
        else:
            # Wrap-around case: sector crosses track start/end
            in_sector = (s_normalized >= s_start or s_normalized <= s_end)

        return in_sector
    # ===== HJ ADDED END =====

    # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: Config):
        """
        Notices the change in the parameters and changes spline params
        """
        self.evasion_dist = rospy.get_param("dynamic_spline_tuner_node/evasion_dist", 0.65)
        self.obs_traj_tresh = rospy.get_param("dynamic_spline_tuner_node/obs_traj_tresh", 0.3)
        self.spline_bound_mindist = rospy.get_param("dynamic_spline_tuner_node/spline_bound_mindist", 0.2)

        self.kd_obs_pred = rospy.get_param("dynamic_spline_tuner_node/kd_obs_pred")
        self.fixed_pred_time = rospy.get_param("dynamic_spline_tuner_node/fixed_pred_time")
        self.sampling_dist = rospy.get_param("dynamic_spline_tuner_node/post_sampling_dist")
        self.sampling_dist = rospy.get_param("dynamic_spline_tuner_node/post_sampling_dist")
        self.spline_scale = rospy.get_param("dynamic_spline_tuner_node/spline_scale")
        self.post_min_dist = rospy.get_param("dynamic_spline_tuner_node/post_min_dist")
        self.post_max_dist = rospy.get_param("dynamic_spline_tuner_node/post_max_dist")
        self.kernel_size = rospy.get_param("dynamic_spline_tuner_node/kernel_size")
        
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        rospy.loginfo(
            f"[{self.name}] evasion apex distance: {self.evasion_dist} [m],\n"
            f" obstacle trajectory treshold: {self.obs_traj_tresh} [m]\n"
            f" obstacle prediciton k_d: {self.kd_obs_pred},    obstacle prediciton constant time: {self.fixed_pred_time} [s] "
        )

    # ===== HJ EDITED START: State machine dynamic param callback =====
    def state_machine_dyn_param_cb(self, params: Config):
        """
        Update params from state_machine dynamic reconfigure (same params for consistency)
        """
        self.lateral_width_gb_m = rospy.get_param("dyn_statemachine/lateral_width_gb_m", 0.1)
        rospy.loginfo(f"[{self.name}] Updated lateral_width_gb_m: {self.lateral_width_gb_m}")
    # ===== HJ EDITED END =====

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        # Wait for critical Messages and services
        rospy.loginfo(f"[{self.name}] Waiting for messages and services...")
        rospy.wait_for_message("/global_waypoints", WpntArray)
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.wait_for_message("/car_state/odom", Odometry)
        rospy.wait_for_message("/dynamic_spline_tuner_node/parameter_updates", Config)
        rospy.loginfo(f"[{self.name}] Ready!")

        while not rospy.is_shutdown():
            if self.measuring:
                start = time.perf_counter()

            # ===== HJ PHASE A: Process obstacles in main loop =====
            self._process_obstacles()
            # ===== HJ PHASE A END =====

            # Sample data
            gb_scaled_wpnts = self.gb_scaled_wpnts.wpnts
            wpnts = OTWpntArray()
            mrks = MarkerArray()

            # ===== HJ EDITED START: Smart static avoidance mode switching =====
            # Check if we should generate fixed path (only if not generated AND not generating)
            if not self.fixed_path_generated and not self.fixed_path_generating and self._check_obs_is_ready_for_path_gen():
                rospy.loginfo(f"[{self.name}] ========================================")
                rospy.loginfo(f"[{self.name}] OBSTACLES VERIFIED! Starting fixed path generation in background...")
                rospy.loginfo(f"[{self.name}] ========================================")

                # Get verified obstacles for path generation
                verified_obs = self._get_verified_obstacles()

                rospy.loginfo(f"[{self.name}] Verified interfering obstacles: {len(verified_obs)}")
                for sector_id, obs_id, s, d in verified_obs:
                    rospy.loginfo(f"[{self.name}]   Sector {sector_id}, Obs {obs_id}: s={s:.2f}m, d={d:.2f}m")

                # Set generating flag BEFORE starting thread
                self.fixed_path_generating = True

                # Phase 3: Generate fixed optimized path using GB optimizer in SEPARATE THREAD
                # This allows main loop to continue running at 20Hz for real-time spline avoidance
                generation_thread = threading.Thread(
                    target=self._generate_fixed_path_async,
                    args=(verified_obs,),
                    daemon=True
                )
                generation_thread.start()
                rospy.loginfo(f"[{self.name}] Path generation thread started. Continuing real-time avoidance...")

            # Real-time spline avoidance (always active)
            if self.obs_in_interest is not None:
                # Select reference path based on use_fixed_path flag
                # ===== HJ MODIFIED: Extract .wpnts list from OTWpntArray =====
                if self.use_fixed_path:
                    reference_wpnts = self.fixed_path_wpnts.wpnts  # Extract list from OTWpntArray
                    rospy.loginfo_throttle(5.0, f"[{self.name}] POST-FIX (Fixed mode): Spline avoidance around FIXED path")
                else:
                    reference_wpnts = gb_scaled_wpnts  # Already a list
                    if self.fixed_path_generating:
                        rospy.loginfo_throttle(5.0, f"[{self.name}] PRE-FIX: Generating... Using GB raceline")
                    elif self.fixed_path_generated:
                        rospy.loginfo_throttle(5.0, f"[{self.name}] POST-FIX (GB mode): Using GB raceline")
                    else:
                        rospy.loginfo_throttle(5.0, f"[{self.name}] PRE-FIX: Using GB raceline")
                # ===== HJ MODIFIED END =====

                # ===== HJ DEBUG: Save do_spline inputs for offline analysis =====
                if self.use_fixed_path:
                    self._debug_save_counter += 1
                    if self._debug_save_counter == 50:  # Save only first successful call
                        import pickle
                        debug_data = {
                            'obs': copy.deepcopy(self.obs_in_interest),
                            'gb_wpnts': reference_wpnts[:],  # Fixed path waypoints (copy list)
                            'use_fixed_path': self.use_fixed_path,
                            'cur_x': self.cur_x,
                            'cur_y': self.cur_y,
                            'cur_yaw': self.cur_yaw,  # Add heading!
                            'cur_s': self.cur_s,
                            'cur_d': self.cur_d,
                            'boundary_left_xy': copy.deepcopy(self.fixed_boundary_left_xy) if hasattr(self, 'fixed_boundary_left_xy') else None,
                            'boundary_right_xy': copy.deepcopy(self.fixed_boundary_right_xy) if hasattr(self, 'fixed_boundary_right_xy') else None
                        }
                        debug_path = '/tmp/do_spline_input_fixed.pkl'
                        with open(debug_path, 'wb') as f:
                            pickle.dump(debug_data, f)
                        # rospy.logwarn(f"[{self.name}] Saved do_spline input to {debug_path}")
                # ===== HJ DEBUG END =====

                wpnts, mrks = self.do_spline(obs=copy.deepcopy(self.obs_in_interest), gb_wpnts=reference_wpnts)

                # ===== HJ ADDED: Debug log for do_spline result =====
                # if len(wpnts.wpnts) == 0:
                #     rospy.logwarn_throttle(2.0,
                #         f"[{self.name}] DEBUG: do_spline returned EMPTY waypoints! "
                #         f"Mode: {'FIXED' if self.use_fixed_path else 'GB'}, "
                #         f"Obs s={self.obs_in_interest.s_center:.2f}, d={self.obs_in_interest.d_center:.2f}")
                # else:
                #     rospy.loginfo_throttle(2.0,
                #         f"[{self.name}] DEBUG: do_spline SUCCESS! "
                #         f"Generated {len(wpnts.wpnts)} waypoints, "
                #         f"Mode: {'FIXED' if self.use_fixed_path else 'GB'}")
                # ===== HJ ADDED END =====
            else:
                # No obstacle in interest, delete markers
                del_mrk = Marker()
                del_mrk.header.stamp = rospy.Time.now()
                del_mrk.action = Marker.DELETEALL
                mrks.markers.append(del_mrk)

            # Publish real-time spline wpnts and markers
            if self.measuring:
                end = time.perf_counter()
                self.latency_pub.publish(end - start)
            self.evasion_pub.publish(wpnts)
            self.mrks_pub.publish(mrks)

            # Publish fixed path separately (if generated) at low rate (0.5Hz)
            # ===== HJ MODIFIED: Low-rate publishing for static data =====
            if self.fixed_path_generated:
                time_since_last_pub = (rospy.Time.now() - self.fixed_path_last_pub_time).to_sec()
                if time_since_last_pub > self.fixed_path_pub_rate:
                    # ===== HJ ADDED: Update timestamp before publishing =====
                    self.fixed_path_wpnts.header.stamp = rospy.Time.now()
                    self.fixed_path_markers.markers[0].header.stamp = rospy.Time.now() if len(self.fixed_path_markers.markers) > 0 else rospy.Time.now()
                    # ===== HJ ADDED END =====
                    rospy.loginfo(f"[{self.name}] Publishing FIXED optimized path ({len(self.fixed_path_wpnts.wpnts)} waypoints) at {1.0/self.fixed_path_pub_rate:.1f}Hz")
                    self.fixed_path_pub.publish(self.fixed_path_wpnts)
                    self.fixed_path_mrks_pub.publish(self.fixed_path_markers)
                    self.fixed_path_last_pub_time = rospy.Time.now()
            # ===== HJ MODIFIED END =====

            # ===== HJ ADDED: Publish use_fixed_path flag every iteration =====
            self.use_fixed_path_pub.publish(Bool(data=self.use_fixed_path))
            # ===== HJ ADDED END =====
            # ===== HJ EDITED END =====
            self.rate.sleep()
    
    #########
    # UTILS #
    #########
    # ===== HJ EDITED START: Check if obstacle interferes with global raceline =====
    def _check_obstacle_interference(self, obs: Obstacle) -> bool:
        """
        Check if obstacle interferes with global raceline.
        Uses same logic as state_machine's _check_free() for consistency,
        but with FIXED scaling_factor=1.0 (no distance-based scaling).

        This ensures we detect interference early and generate path once.

        NOTE: Potential issue - obstacles that don't interfere with global raceline
        might interfere with the optimized path we generate. This needs to be monitored.

        Returns True if obstacle interferes (needs avoidance).
        """
        if self.gb_wpnts.wpnts is None or len(self.gb_wpnts.wpnts) == 0:
            return False

        # Get global waypoint at obstacle's s position
        wpnt_dist = self.gb_wpnts.wpnts[1].s_m - self.gb_wpnts.wpnts[0].s_m
        obs_s_idx = int(obs.s_center / wpnt_dist) % self.gb_max_idx
        gb_wpnt = self.gb_wpnts.wpnts[obs_s_idx]

        # Distance from raceline (d=0) to obstacle center
        # For global waypoints, d_m is the lateral offset from centerline
        min_dist = abs(gb_wpnt.d_m - obs.d_center) if hasattr(gb_wpnt, 'd_m') else abs(0 - obs.d_center)

        # Free distance calculation (same as state_machine)
        free_dist = min_dist - obs.size/2 - self.gb_ego_width_m/2

        # Fixed scaling_factor = 1.0 (no distance-based scaling for static obstacles)
        # This ensures consistent interference detection regardless of distance
        scaling_factor = 1.0

        # Check interference (same logic as state_machine)
        if free_dist < self.lateral_width_gb_m * scaling_factor:
            return True  # Interference detected!
        else:
            return False  # No interference
    # ===== HJ EDITED END =====

    # ===== HJ EDITED START: Check if obstacle interferes with fixed path =====
    def _check_interference_fixed_path(self, obs: Obstacle) -> bool:
        """
        Check if obstacle interferes with fixed path (Single Frenet approach).
        Uses FrenetConverter to calculate lateral distance from fixed path.

        This is used to detect NEW obstacles that interfere with the GENERATED fixed path,
        not the original GB raceline. When interference is detected, regeneration is triggered.

        Returns True if obstacle interferes with fixed path.
        """
        # Fixed path not generated yet
        if self.fixed_converter is None:
            return False

        # Convert obstacle position to fixed path Frenet coordinates
        result = self.fixed_converter.get_frenet(np.array([obs.x_m]), np.array([obs.y_m]))
        s_fixed, d_fixed = result[0], result[1]

        # Calculate free distance from fixed path
        free_dist = abs(d_fixed) - obs.size/2 - self.gb_ego_width_m/2

        # Check interference (same threshold as GB raceline check)
        scaling_factor = 1.0  # Fixed scaling
        if free_dist < self.lateral_width_gb_m * scaling_factor:
            return True  # Interference with fixed path!
        else:
            return False  # No interference
    # ===== HJ EDITED END =====

    # ===== HJ PHASE D: Get verified obstacles from memory (interfering only) =====
    def _get_verified_obstacles(self) -> List[Tuple[int, int, float, float]]:
        """
        Phase D: Extract verified INTERFERING obstacles from memory for path generation.

        Returns list of tuples: [(sector_id, obs_id, s_mean, d_mean), ...]
        Only includes obstacles that are:
        1. Stable (verified)
        2. Interfering with global raceline

        SIDE EFFECT: Stores sector IDs with interfering obstacles in self.sectors_with_interfering_obs
        """
        verified_obs = []
        interfering_sectors = set()  # Track which sectors have interfering obstacles

        for (sector_id, obs_id), mem in self.static_obs_memory.items():
            # Skip non-interfering obstacles
            if not mem['interferes']:
                continue

            # Check if this obstacle has enough observations
            if mem['obs_count'] < self.min_stable_observations:
                continue

            # Check position stability
            if len(mem['s_history']) < self.min_stable_observations:
                continue

            s_std = np.std(mem['s_history'])
            d_std = np.std(mem['d_history'])

            # Only include verified (stable) obstacles
            if s_std < self.position_std_threshold and d_std < self.position_std_threshold:
                s_mean = np.mean(mem['s_history'])
                d_mean = np.mean(mem['d_history'])
                verified_obs.append((sector_id, obs_id, s_mean, d_mean))
                interfering_sectors.add(sector_id)  # Remember this sector had interfering obstacles
                rospy.loginfo(f"[{self.name}] Verified INTERFERING obstacle: "
                             f"sector={sector_id}, id={obs_id}, s={s_mean:.2f}, d={d_mean:.2f}")

        # Store which sectors had interfering obstacles (will be used in zone passage monitoring)
        self.sectors_with_interfering_obs = interfering_sectors
        rospy.loginfo(f"[{self.name}] Sectors with interfering obstacles: {interfering_sectors}")

        return verified_obs
    # ===== HJ PHASE D END =====

    # ===== HJ PHASE C: Check if obstacles are ready for path generation (2-stage condition) =====
    def _check_obs_is_ready_for_path_gen(self) -> bool:
        """
        Phase C: 2-stage path generation condition.

        Stage 1 (is_ready): 2+ sectors with stable obstacles (interference irrelevant)
        Stage 2 (should_generate): Has interfering obstacles to optimize for

        Returns True only if BOTH conditions met.
        """
        # Count sectors with verified obstacles
        sectors_with_verified_obs = set()
        interfering_count = 0

        for (sector_id, obs_id), mem in self.static_obs_memory.items():
            # Check if this obstacle is verified (stable)
            if mem['obs_count'] < self.min_stable_observations:
                continue  # Not enough observations yet

            if len(mem['s_history']) < self.min_stable_observations:
                continue  # Not enough history

            s_std = np.std(mem['s_history'])
            d_std = np.std(mem['d_history'])

            # Check if position is verified (stable)
            if s_std < self.position_std_threshold and d_std < self.position_std_threshold:
                sectors_with_verified_obs.add(sector_id)

                if mem['interferes']:
                    interfering_count += 1

                rospy.loginfo_throttle(2.0,
                    f"[{self.name}] Sector {sector_id}, Obs {obs_id} VERIFIED: "
                    f"s_std={s_std:.4f}, d_std={d_std:.4f}, interferes={mem['interferes']}")

        # Stage 1: is_ready
        is_ready = len(sectors_with_verified_obs) >= self.min_sectors_with_stable_obs

        # Stage 2: should_generate
        should_generate = interfering_count > 0

        if is_ready and should_generate:
            rospy.loginfo(f"[{self.name}] READY & SHOULD GENERATE! "
                         f"{len(sectors_with_verified_obs)} sectors, {interfering_count} interfering obstacles")
            return True
        elif is_ready and not should_generate:
            rospy.loginfo_throttle(5.0,
                f"[{self.name}] Ready but no interference "
                f"({len(sectors_with_verified_obs)} sectors, 0 interfering)")
            return False
        else:
            rospy.loginfo_throttle(5.0,
                f"[{self.name}] Waiting: {len(sectors_with_verified_obs)}/{self.min_sectors_with_stable_obs} sectors")
            return False
    # ===== HJ PHASE C END =====

    # ===== HJ PHASE B: Find existing obstacle key by position =====
    def _find_obstacle_key_by_position(self, obs: Obstacle):
        """
        Find existing obstacle in memory by position (in case ID changed due to TTL).
        Returns existing key if nearby obstacle found, otherwise returns new key.

        This handles the case where tracking loses an obstacle temporarily and
        reassigns a different ID when it reappears at the same location.
        """
        if not self.use_position_based_merge:
            return (obs.sector_id, obs.id)  # Just use current ID

        # Search for nearby obstacles in same sector
        for (sector_id, old_id), mem in self.static_obs_memory.items():
            if sector_id != obs.sector_id:
                continue  # Different sector

            # Check if position is very close to last observation
            if len(mem['s_history']) > 0:
                last_s = mem['s_history'][-1]
                last_d = mem['d_history'][-1]

                # Euclidean distance in Frenet space
                dist = np.sqrt((obs.s_center - last_s)**2 + (obs.d_center - last_d)**2)

                if dist < self.position_merge_threshold:
                    # Same obstacle, different ID!
                    rospy.loginfo_throttle(5.0,
                        f"[{self.name}] Merged obstacle: old_id={old_id} → new_id={obs.id} "
                        f"(sector={sector_id}, dist={dist:.3f}m)")
                    return (sector_id, old_id)  # Reuse old key

        # No nearby obstacle found, use new key
        return (obs.sector_id, obs.id)
    # ===== HJ PHASE B END =====

    # ===== HJ PHASE A: Process obstacles function =====
    def _process_obstacles(self):
        """
        Process latest obstacles from callback.
        Called from main loop to avoid callback overload.

        Tasks:
        1. Filter obstacles (in_static_obs_sector AND is_static)
        2. Update memory for ALL filtered obstacles (interference check stored)
        3. Set obs_in_interest for PRE-FIX mode
        """
        if not self.obstacles_updated:
            return

        # ===== DEBUG: Log all obstacles (throttled) =====
        if len(self.latest_obstacles) > 0:
            rospy.loginfo_throttle(2.0, f"[{self.name}] ===== OBSTACLE DEBUG =====")
            for obs in self.latest_obstacles:
                rospy.loginfo_throttle(2.0,
                    f"[{self.name}] OBS: id={obs.id}, "
                    f"sector_id={obs.sector_id}, "
                    f"in_static_obs_sector={obs.in_static_obs_sector}, "
                    f"is_static={obs.is_static}, "
                    f"s={obs.s_center:.2f}m, d={obs.d_center:.2f}m")

        # Filter obstacles into two lists:
        # 1. static_obs_in_sector: For memory tracking (use latest_obstacles)
        # 2. static_obs_for_decision: For GB revert decision (use memory, more stable)
        # 3. all_static_obs: For do_spline (includes obstacles outside sector)

        # ===== HJ MODIFIED: Separate memory update and decision logic =====
        # For memory update: use latest_obstacles (real-time)
        static_obs_in_sector = [obs for obs in self.latest_obstacles
                                if obs.in_static_obs_sector and obs.is_static]

        # For decision: use memory (stable across callbacks)
        obstacles_from_memory = []
        for (sector_id, obs_id), mem in self.static_obs_memory.items():
            if len(mem['s_history']) > 0:
                # Create obstacle object from latest memory data
                obs = Obstacle()
                obs.sector_id = sector_id
                obs.id = obs_id
                obs.s_center = mem['s_history'][-1]  # Latest s
                obs.d_center = mem['d_history'][-1]  # Latest d
                obs.is_static = True
                obs.in_static_obs_sector = True
                obstacles_from_memory.append(obs)

        # ===== HJ ADDED: Debug logging =====
        if self.fixed_path_generated:
            rospy.loginfo_throttle(1.0,
                f"[{self.name}] DEBUG Filter: latest_obstacles={len(self.latest_obstacles)}, "
                f"static_in_sector={len(static_obs_in_sector)}, "
                f"memory_obstacles={len(obstacles_from_memory)}, "
                f"use_fixed_path={self.use_fixed_path}")
        # ===== HJ ADDED END =====

        # For do_spline: use latest_obstacles (real-time avoidance)
        all_static_obs = [obs for obs in self.latest_obstacles
                          if obs.is_static]
        # ===== HJ MODIFIED END =====

        # ===== DEBUG: Log filtering results =====
        rospy.loginfo_throttle(2.0,
            f"[{self.name}] Total: {len(self.latest_obstacles)}, "
            f"In static sector: {len(static_obs_in_sector)}, "
            f"All static: {len(all_static_obs)}")

        # Early return if not initialized
        if self.gb_max_s is None or self.gb_wpnts.wpnts is None or len(self.gb_wpnts.wpnts) == 0:
            self.obstacles_updated = False
            return

        # Phase B: Update memory for obstacles IN SECTOR (not just interfering ones)
        # Use (sector_id, obs_id) tuple keys for individual obstacle tracking
        for obs in static_obs_in_sector:
            # Check interference with GB raceline (for path generation trigger)
            interferes_gb = self._check_obstacle_interference(obs)

            # Check interference with fixed path (for regeneration trigger)
            interferes_fixed = self._check_interference_fixed_path(obs)

            # Find appropriate key (may merge with existing obstacle if position matches)
            key = self._find_obstacle_key_by_position(obs)

            # Initialize obstacle memory if first time
            if key not in self.static_obs_memory:
                self.static_obs_memory[key] = {
                    's_history': [],
                    'd_history': [],
                    's_history_fixed': [],  # Fixed path Frenet s coordinates
                    'd_history_fixed': [],  # Fixed path Frenet d coordinates
                    'size_history': [],  # Store obstacle size for GB optimizer wall modification
                    'obs_count': 0,
                    'interferes': False,  # GB raceline interference
                    'interferes_fixed': False,  # Fixed path interference
                    'last_seen': rospy.Time.now()
                }

            mem = self.static_obs_memory[key]
            mem['s_history'].append(obs.s_center)
            mem['d_history'].append(obs.d_center)
            mem['size_history'].append(obs.size)  # Store size
            mem['obs_count'] += 1
            mem['interferes'] = interferes_gb  # Update GB interference status
            mem['interferes_fixed'] = interferes_fixed  # Update fixed path interference status
            mem['last_seen'] = rospy.Time.now()

            # Store fixed path Frenet coordinates (if fixed converter available)
            if self.fixed_converter is not None:
                result = self.fixed_converter.get_frenet(np.array([obs.x_m]), np.array([obs.y_m]))
                s_fixed, d_fixed = result[0], result[1]
                mem['s_history_fixed'].append(s_fixed)
                mem['d_history_fixed'].append(d_fixed)

            # Keep only recent history
            if len(mem['s_history']) > self.max_history_length:
                mem['s_history'].pop(0)
                mem['d_history'].pop(0)
                mem['size_history'].pop(0)
                if len(mem['s_history_fixed']) > 0:
                    mem['s_history_fixed'].pop(0)
                    mem['d_history_fixed'].pop(0)

        # Phase B: Remove old obstacles (not seen for timeout duration)
        current_time = rospy.Time.now()

        # ===== HJ ADDED: Adaptive timeout decay (after fixed path generated) =====
        if self.last_timeout_decay_time is not None:
            time_since_last_decay = (current_time - self.last_timeout_decay_time).to_sec()
            if time_since_last_decay >= self.memory_timeout_decay_interval:
                # Reduce timeout by 1 second
                old_timeout = self.memory_timeout_sec
                self.memory_timeout_sec = max(self.memory_timeout_min,
                                             self.memory_timeout_sec - self.memory_timeout_decay_amount)
                self.last_timeout_decay_time = current_time
                rospy.logwarn(f"[{self.name}] Memory timeout decay: {old_timeout:.1f}s → {self.memory_timeout_sec:.1f}s")
        # ===== HJ ADDED END =====

        timeout = rospy.Duration(self.memory_timeout_sec)

        # ===== HJ ADDED: Debug memory state =====
        rospy.loginfo_throttle(1.0,
            f"[{self.name}] DEBUG MEMORY: {len(self.static_obs_memory)} obstacles in memory, "
            f"timeout={self.memory_timeout_sec}s")

        keys_to_remove = []
        for key, mem in self.static_obs_memory.items():
            time_diff = (current_time - mem['last_seen']).to_sec()
            rospy.loginfo_throttle(1.0,
                f"[{self.name}] DEBUG MEM OBS: sector={key[0]}, id={key[1]}, "
                f"time_since_last_seen={time_diff:.2f}s, will_remove={time_diff > self.memory_timeout_sec}")

            if (current_time - mem['last_seen']) > timeout:
                keys_to_remove.append(key)
                rospy.logwarn(f"[{self.name}] DEBUG TIMEOUT: Removing obstacle sector={key[0]}, id={key[1]}, "
                             f"time_diff={time_diff:.2f}s, timeout={self.memory_timeout_sec}s")
        # ===== HJ ADDED END =====

        for key in keys_to_remove:
            del self.static_obs_memory[key]
            rospy.loginfo(f"[{self.name}] Removed stale obstacle: sector={key[0]}, id={key[1]}")

        # ===== Phase: Decide use_fixed_path and set obs_in_interest =====
        if self.fixed_path_generated:
            # ===== HJ ADDED: Vehicle position-based active check =====
            # POST-FIX mode: Revert to GB only if vehicle is in sector but no interfering obstacles detected
            # If vehicle is in sector but no interfering obstacles detected, revert to GB
            # (Handles "memory forgot" scenario after vehicle stop/restart)
            if self.use_fixed_path and len(self.static_obs_sectors) > 0:
                vehicle_in_sector = False
                vehicle_sector_id = None

                # Check if vehicle is in any ACTIVE sector's detection zone
                # Detection zone: [3m before sector start, 15% into sector]
                # NEW Logic: Monitor zone passage - only revert if vehicle passes ENTIRE zone without obstacles
                sector_start_margin = 3.0  # meters - start checking 3m BEFORE sector start
                sector_check_threshold = 0.15  # Check up to 15% into sector

                for sector_id, sector_data in self.static_obs_sectors.items():
                    if not sector_data['static_obs_section']:
                        continue

                    # ===== HJ ADDED: Only monitor sectors that had interfering obstacles during path generation =====
                    # Skip sectors that never had interfering obstacles - they shouldn't trigger GB reversion
                    if sector_id not in self.sectors_with_interfering_obs:
                        rospy.loginfo_throttle(5.0,
                            f"[{self.name}] Skipping sector {sector_id} zone monitoring (no interfering obstacles during path generation)")
                        continue
                    # ===== HJ ADDED END =====

                    s_start = sector_data['s_start']
                    s_end = sector_data['s_end']

                    # Calculate sector length
                    if s_start <= s_end:
                        # Normal case: no wrap-around
                        sector_length = s_end - s_start
                    else:
                        # Wrap-around case: sector crosses track boundary
                        sector_length = (self.track_length - s_start) + s_end

                    # Define detection zone boundaries
                    check_start = (s_start - sector_start_margin) % self.track_length
                    check_end = (s_start + sector_length * sector_check_threshold) % self.track_length

                    # Normalize current position
                    cur_s_norm = self.cur_s % self.track_length

                    # Check if vehicle is in detection zone (handle wrap-around)
                    if check_start <= check_end:
                        # Normal case: zone doesn't wrap
                        in_zone = (cur_s_norm >= check_start and cur_s_norm <= check_end)
                    else:
                        # Wrap-around case: zone crosses track boundary
                        in_zone = (cur_s_norm >= check_start or cur_s_norm <= check_end)

                    # Initialize monitoring state for this sector if needed
                    if sector_id not in self.sector_monitoring:
                        self.sector_monitoring[sector_id] = {
                            'in_zone': False,
                            'had_obstacles': False,
                            'entered_at_s': None
                        }

                    monitor = self.sector_monitoring[sector_id]

                    # Check for interfering obstacles IN THIS SPECIFIC SECTOR
                    # Filter to only obstacles in this sector (using sector_id attribute)
                    interfering_obs = [obs for obs in self.latest_obstacles
                                      if obs.in_static_obs_sector
                                      and obs.is_static
                                      and obs.sector_id == sector_id  # Only check THIS sector's obstacles
                                      and self._check_obstacle_interference(obs)]

                    if in_zone:
                        # Vehicle entered or is in zone
                        if not monitor['in_zone']:
                            # Just entered zone
                            monitor['in_zone'] = True
                            monitor['had_obstacles'] = False  # Reset
                            monitor['entered_at_s'] = cur_s_norm
                            rospy.loginfo(
                                f"[{self.name}] Vehicle ENTERED sector {sector_id} detection zone "
                                f"(s={self.cur_s:.2f}m, zone={check_start:.2f}-{check_end:.2f}m, "
                                f"sector had interfering obs during path generation)")

                        # Mark if obstacles detected
                        if interfering_obs:
                            monitor['had_obstacles'] = True
                            rospy.loginfo_throttle(1.0,
                                f"[{self.name}] Sector {sector_id}: {len(interfering_obs)} interfering obstacles detected")

                        vehicle_in_sector = True
                        vehicle_sector_id = sector_id
                        break

                    else:
                        # Vehicle exited zone
                        if monitor['in_zone']:
                            # Just exited - check if we can revert
                            monitor['in_zone'] = False

                            if not monitor['had_obstacles']:
                                # Passed through entire zone without obstacles → revert to GB
                                self.use_fixed_path = False
                                rospy.logwarn(
                                    f"[{self.name}] POST-FIX: Vehicle EXITED sector {sector_id} zone "
                                    f"(s={self.cur_s:.2f}m) WITHOUT obstacles detected during passage → Reverting to GB")
                            else:
                                rospy.loginfo(
                                    f"[{self.name}] Vehicle exited sector {sector_id} zone, "
                                    f"but had obstacles during passage → Keep using fixed path")
            # ===== HJ ADDED END =====
            # ===== HJ MODIFIED END =====

        # Set obs_in_interest from ALL static obstacles (not just in sector)
        # Filter by interference with current reference path (GB or Fixed)
        if self.use_fixed_path and self.fixed_converter is not None:
            # Using fixed path: Check interference and find closest
            interfering_obs_fixed = [obs for obs in all_static_obs
                                    if self._check_interference_fixed_path(obs)]

            if interfering_obs_fixed:
                # Get vehicle position in fixed path Frenet
                result_ego = self.fixed_converter.get_frenet(np.array([self.cur_x]), np.array([self.cur_y]))
                cur_s_fixed = result_ego[0]

                # Find closest using FIXED path s coordinates
                closest_obs = None
                min_dist = float('inf')
                for obs in interfering_obs_fixed:
                    result = self.fixed_converter.get_frenet(np.array([obs.x_m]), np.array([obs.y_m]))
                    s_fixed, d_fixed = result[0], result[1]
                    # Calculate distance in fixed path s (no max_s wrapping for simplicity)
                    dist = abs(s_fixed - cur_s_fixed)
                    if dist < min_dist:
                        min_dist = dist
                        closest_obs = obs
                        closest_s_fixed = s_fixed
                        closest_d_fixed = d_fixed

                # Create obstacle with fixed Frenet coordinates for do_spline
                # ===== HJ MODIFIED: Convert all Frenet coordinates to Fixed frame =====
                obs_fixed = copy.deepcopy(closest_obs)
                obs_fixed.s_center = float(closest_s_fixed)  # Convert to Python float for ROS message
                obs_fixed.d_center = float(closest_d_fixed)  # Convert to Python float for ROS message

                # Convert d_left, d_right to Fixed Frenet (method 2: use size)
                # Assuming circular obstacle, size doesn't change with coordinate system
                obs_fixed.d_left = float(closest_d_fixed) + closest_obs.size / 2.0
                obs_fixed.d_right = float(closest_d_fixed) - closest_obs.size / 2.0

                self.obs_in_interest = obs_fixed
                rospy.loginfo_throttle(2.0,
                    f"[{self.name}] POST-FIX (Fixed mode): {len(interfering_obs_fixed)} obstacles, "
                    f"closest at fixed s={float(closest_s_fixed):.2f}, d={float(closest_d_fixed):.2f}, "
                    f"d_left={obs_fixed.d_left:.2f}, d_right={obs_fixed.d_right:.2f}")
                # ===== HJ MODIFIED END =====
            else:
                self.obs_in_interest = None
        else:
            # Using GB raceline: Check interference and find closest
            interfering_obs_gb = [obs for obs in all_static_obs
                                 if self._check_obstacle_interference(obs)]

            if interfering_obs_gb:
                # Find closest using GB Frenet s
                closest_obs = min(interfering_obs_gb,
                                key=lambda o: (o.s_center - self.cur_s) % self.gb_max_s)
                self.obs_in_interest = closest_obs
                mode_str = "POST-FIX (GB mode)" if self.fixed_path_generated else "PRE-FIX"
                rospy.loginfo_throttle(2.0,
                    f"[{self.name}] {mode_str}: {len(interfering_obs_gb)} obstacles, "
                    f"closest at GB s={closest_obs.s_center:.2f}, d={closest_obs.d_center:.2f}")
            else:
                self.obs_in_interest = None

        self.obstacles_updated = False
    # ===== HJ PHASE A END =====

    def initialize_converter(self) -> FrenetConverter:
        """
        Initialize the FrenetConverter object"""
        rospy.wait_for_message("/global_waypoints", WpntArray)

        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])
        rospy.loginfo(f"[{self.name}] initialized FrenetConverter object")

        return converter

    def _more_space(self, obstacle: Obstacle, gb_wpnts: List[Any], obs_s_idx: int) -> Tuple[str, float]:
        # DEBUG: Check if d_left/d_right are properly set
        wpnt_d_left = gb_wpnts[obs_s_idx].d_left
        wpnt_d_right = gb_wpnts[obs_s_idx].d_right
        # rospy.logwarn_throttle(1.0,
        #     f"[{self.name}] _more_space: wpnt[{obs_s_idx}] d_left={wpnt_d_left:.2f}, d_right={wpnt_d_right:.2f}, "
        #     f"obs d_left={obstacle.d_left:.2f}, d_right={obstacle.d_right:.2f}")

        left_gap = abs(wpnt_d_left - obstacle.d_left)
        right_gap = abs(wpnt_d_right + obstacle.d_right)
        min_space = self.evasion_dist + self.spline_bound_mindist

        if right_gap > min_space and left_gap < min_space:
            # Compute apex distance to the right of the opponent
            d_apex_right = obstacle.d_right - self.evasion_dist
            # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
            if d_apex_right > 0:
                d_apex_right = 0
            return "right", d_apex_right

        elif left_gap > min_space and right_gap < min_space:
            # Compute apex distance to the left of the opponent
            d_apex_left = obstacle.d_left + self.evasion_dist
            # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
            if d_apex_left < 0:
                d_apex_left = 0
            return "left", d_apex_left
        else:
            candidate_d_apex_left = obstacle.d_left + self.evasion_dist
            candidate_d_apex_right = obstacle.d_right - self.evasion_dist

            if abs(candidate_d_apex_left) <= abs(candidate_d_apex_right):
                # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
                if candidate_d_apex_left < 0:
                    candidate_d_apex_left = 0
                return "left", candidate_d_apex_left
            else:
                # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
                if candidate_d_apex_right > 0:
                    candidate_d_apex_right = 0
                return "right", candidate_d_apex_right

    def do_spline(self, obs: Obstacle, gb_wpnts: WpntArray) -> Tuple[WpntArray, MarkerArray]:
        """
        Creates an evasion trajectory for the closest obstacle by splining between pre- and post-apex points.

        This function takes as input the obstacles to be evaded, and a list of global waypoints that describe a reference raceline.
        It only considers obstacles that are within a threshold of the raceline and generates an evasion trajectory for each of these obstacles.
        The evasion trajectory consists of a spline between pre- and post-apex points that are spaced apart from the obstacle.
        The spatial and velocity components of the spline are calculated using the `Spline` class, and the resulting waypoints and markers are returned.

        Args:
        - obstacles (ObstacleArray): An array of obstacle objects to be evaded.
        - gb_wpnts (WpntArray): A list of global waypoints that describe a reference raceline.
        - state (Odometry): The current state of the car.

        Returns:
        - wpnts (WpntArray): An array of waypoints that describe the evasion trajectory to the closest obstacle.
        - mrks (MarkerArray): An array of markers that represent the waypoints in a visualization format.

        """
        # Return wpnts and markers
        mrks = MarkerArray()
        wpnts = OTWpntArray()
        wpnts.header.stamp = rospy.Time.now()
        wpnts.header.frame_id = "map"

        # ===== HJ MODIFIED: Calculate wpnt_dist using appropriate Frenet coordinate system =====
        ref_max_idx = len(gb_wpnts)  # Max index for current reference path (GB or Fixed)

        if self.use_fixed_path and self.fixed_converter is not None:
            # Fixed mode: Convert waypoints to Fixed Frenet to get accurate spacing
            wp0_frenet = self.fixed_converter.get_frenet(
                np.array([gb_wpnts[0].x_m]), np.array([gb_wpnts[0].y_m]))
            wp1_frenet = self.fixed_converter.get_frenet(
                np.array([gb_wpnts[1].x_m]), np.array([gb_wpnts[1].y_m]))
            wpnt_dist = float(wp1_frenet[0] - wp0_frenet[0])

            # Calculate ref_max_s as cumulative path length (not last waypoint's s)
            # For closed loop, accumulate distances between consecutive waypoints
            total_dist = 0.0
            for i in range(len(gb_wpnts)):
                dx = gb_wpnts[i].x_m - gb_wpnts[i-1].x_m
                dy = gb_wpnts[i].y_m - gb_wpnts[i-1].y_m
                total_dist += np.sqrt(dx**2 + dy**2)
            ref_max_s = total_dist

            rospy.loginfo_throttle(2.0,
                f"[{self.name}] DEBUG FIXED wpnt_dist: GB_s_diff={gb_wpnts[1].s_m - gb_wpnts[0].s_m:.4f}m, "
                f"FIXED_s_diff={wpnt_dist:.4f}m, ref_max_s_FIXED={ref_max_s:.2f}m (cumulative)")
        else:
            # GB mode: Use existing s_m from waypoints
            wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m
            ref_max_s = gb_wpnts[-1].s_m
        # ===== HJ MODIFIED END =====

        # CRITICAL: Convert current vehicle position to reference path Frenet coordinates
        if self.use_fixed_path and self.fixed_converter is not None:
            # Fixed path mode: Convert vehicle XY to Fixed Frenet
            result_ego = self.fixed_converter.get_frenet(np.array([self.cur_x]), np.array([self.cur_y]))
            cur_s_ref = float(result_ego[0]) % ref_max_s  # Normalize to [0, ref_max_s)
            cur_d_ref = float(result_ego[1])

            # Also normalize obstacle s to same range
            obs_s_normalized = obs.s_center % ref_max_s

            rospy.loginfo_throttle(2.0,
                f"[{self.name}] DEBUG do_spline START (FIXED): ref_max_idx={ref_max_idx}, ref_max_s={ref_max_s:.2f}m, "
                f"wpnt_dist={wpnt_dist:.4f}m, cur_s_FIXED={cur_s_ref:.2f}m (raw={float(result_ego[0]):.2f}m, GB={self.cur_s:.2f}m), "
                f"obs.s={obs_s_normalized:.2f}m (raw={obs.s_center:.2f}m), obs.d={obs.d_center:.2f}m")

            # Update obs.s_center for this function scope
            obs = copy.deepcopy(obs)
            obs.s_center = obs_s_normalized
        else:
            # GB mode: Use existing GB Frenet coordinates
            cur_s_ref = self.cur_s
            cur_d_ref = self.cur_d
            # rospy.logerr(
            #     f"[{self.name}] DEBUG do_spline START (GB): ref_max_idx={ref_max_idx}, ref_max_s={ref_max_s:.2f}m, "
            #     f"wpnt_dist={wpnt_dist:.4f}m, cur_s_GB={cur_s_ref:.2f}m, "
            #     f"obs.s={obs.s_center:.2f}m, obs.d={obs.d_center:.2f}m")
        # ===== HJ ADDED END =====

        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        # close_obs = self._obs_filtering(obstacles=obstacles)

        # If there are obstacles within the lookahead distance, then we need to generate an evasion trajectory considering the closest one
        if obs.is_static == True:
            # ===== HJ MODIFIED: Use cur_s_ref (reference path coordinates) instead of self.cur_s =====
            # Calculate raw distance (can be negative if obstacle is behind)
            pre_dist_raw = obs.s_center - cur_s_ref

            # Apply modulo for wrap-around
            pre_dist = pre_dist_raw % ref_max_s

            # Check if obstacle is behind (after wrap-around, distance > half track means it's actually behind)
            if pre_dist > ref_max_s / 2:
                # Obstacle is behind us, skip
                # rospy.logwarn(
                #     f"[{self.name}] DEBUG ABORT #1-A: Obstacle behind! "
                #     f"pre_dist_raw={pre_dist_raw:.2f}m, pre_dist={pre_dist:.2f}m, ref_max_s/2={ref_max_s/2:.2f}m")
                wpnts.wpnts = []
                mrks.markers = []
                return wpnts, mrks

            if pre_dist < 0.5:
                # ===== HJ ADDED: Debug log for early return =====
                # rospy.logwarn(
                #     f"[{self.name}] DEBUG ABORT #1-B: Obstacle too close! "
                #     f"pre_dist={pre_dist:.2f}m (threshold=0.5m), cur_s={cur_s_ref:.2f}m, obs.s={obs.s_center:.2f}m")
                # ===== HJ ADDED END =====
                wpnts.wpnts = []
                mrks.markers = []
                return wpnts, mrks

            obs_s_idx = int(obs.s_center / wpnt_dist) % ref_max_idx
            # ===== HJ MODIFIED END =====

            more_space, d_apex = self._more_space(obs, gb_wpnts, obs_s_idx)
            s_list = [obs.s_center]
            d_list = [d_apex]

            # ===== HJ MODIFIED: Use ref_max_s instead of self.gb_max_s =====
            post_dist = min(min(max(pre_dist, self.post_min_dist), self.post_max_dist), ref_max_s / 2)
            # ===== HJ MODIFIED END =====

            num_post_ref = int((post_dist // self.sampling_dist)) + 1

            for i in range(num_post_ref):
                s_list.append(obs.s_center + post_dist * ((i + 1)/ num_post_ref))
                d_list.append((d_apex * (1 - (i + 1)/ num_post_ref)))
                            
            # evasion_s = np.array([self.cur_s +1 ,self.cur_s +2])
            s_array = np.array(s_list)
            d_array = np.array(d_list)

            # ===== HJ MODIFIED: Use ref_max_s and ref_max_idx =====
            s_array = s_array % ref_max_s

            s_idx = np.round((s_array / wpnt_dist)).astype(int) % ref_max_idx
            # ===== HJ MODIFIED END =====
            # evasion2 = int(evasion_s[1] / wpnt_dist) % self.gb_max_idx
            
            # gb_idxs = [evasion1, evasion2]
            
            # Choose the correct side and compute the distance to the apex based on left of right of the obstacle

            # evasion_d = np.array([obs.d_center + (obs.size/2 + 0.5)  , 0])
            # evasion_d = np.array([d_apex  , 0])
            
            # Do frenet conversion via conversion service for spline and create markers and wpnts
            danger_flag = False
            # ===== HJ ADDED: Use appropriate Frenet converter =====
            # rospy.logerr(
            #     f"[{self.name}] DEBUG Frenet→Cartesian: s_array={s_array}, d_array={d_array}, "
            #     f"Converter: {'FIXED' if (self.use_fixed_path and self.fixed_converter) else 'GB'}")

            if self.use_fixed_path and self.fixed_converter is not None:
                resp = self.fixed_converter.get_cartesian(s_array, d_array)
                # rospy.logerr(f"[{self.name}] DEBUG: Using FIXED converter for Frenet→Cartesian")
            else:
                resp = self.converter.get_cartesian(s_array, d_array)
                # rospy.logerr(f"[{self.name}] DEBUG: Using GB converter for Frenet→Cartesian")
            # ===== HJ ADDED END =====

            # ===== HJ ADDED: Debug log for Cartesian conversion results =====
            # rospy.logerr(f"[{self.name}] DEBUG Cartesian points from Frenet conversion:")
            # for i in range(len(s_array)):
            #     rospy.logerr(f"  Point {i}: s={s_array[i]:.2f}, d={d_array[i]:.2f} → x={resp[0,i]:.2f}, y={resp[1,i]:.2f}")
            # ===== HJ ADDED END =====

            points=[[self.cur_x,self.cur_y]]
            tangents=[[np.cos(self.cur_yaw), np.sin(self.cur_yaw)]]

            for i in range(len(s_idx)):
                points.append(resp[:, i])
                tangents.append(np.array([np.cos(gb_wpnts[s_idx[i]].psi_rad), np.sin(gb_wpnts[s_idx[i]].psi_rad)]))

            # points.append([self.cur_x,self.cur_y])
            # tangents.append(np.array([np.cos(self.cur_yaw), np.sin(self.cur_yaw)]))

            
            # points.append(resp[:,0])
            # points.append(resp[:,1])

            # tangents.append(np.array([np.cos(gb_wpnts[evasion1].psi_rad), np.sin(gb_wpnts[evasion1].psi_rad)]))
            # tangents.append(np.array([np.cos(gb_wpnts[evasion2].psi_rad), np.sin(gb_wpnts[evasion2].psi_rad)]))

            # tangents = np.dot(tangents, self.spline_scale*np.eye(len(points)))
            tangents = np.dot(tangents, self.spline_scale*np.eye(2))
            points = np.asarray(points)
            nPoints, dim = points.shape

            # Parametrization parameter s.
            dp = np.diff(points, axis=0)                 # difference between points
            dp = np.linalg.norm(dp, axis=1)              # distance between points
            d = np.cumsum(dp)                            # cumsum along the segments
            d = np.hstack([[0],d])                       # add distance from first point
            l = d[-1]                                    # length of point sequence
            nSamples =  int(l/wpnt_dist)                 # number of samples 
            s,r = np.linspace(0,l,nSamples,retstep=True) # sample parameter and step

            # Bring points and (optional) tangent information into correct format.
            assert(len(points) == len(tangents))
            spline_result = np.empty([nPoints, dim], dtype=object)
            for i,ref in enumerate(points):
                t = tangents[i]
                # Either tangent is None or has the same
                # number of dimensions as the point ref.
                assert(t is None or len(t)==dim)
                fuse = list(zip(ref,t) if t is not None else zip(ref,))
                spline_result[i,:] = fuse

            # Compute splines per dimension separately.
            samples = np.zeros([nSamples, dim])
            for i in range(dim):
                poly = BPoly.from_derivatives(d, spline_result[:,i])
                samples[:,i] = poly(s)

            # if samples.shape[0] < self.n_loc_wpnts:
            n_additional = 100
            # ===== HJ MODIFIED: Use ref_max_idx instead of self.gb_max_idx =====
            xy_additional = np.array([
                (
                    gb_wpnts[(s_idx[-1] + i + 1) % ref_max_idx].x_m,
                    gb_wpnts[(s_idx[-1] + i + 1) % ref_max_idx].y_m
                )
                for i in range(n_additional)
            ])
            # ===== HJ MODIFIED END =====
            samples = np.vstack([samples, xy_additional])

            # ===== HJ MODIFIED: Use appropriate Frenet converter for Cartesian→Frenet =====
            if self.use_fixed_path and self.fixed_converter is not None:
                s_, d_ = self.fixed_converter.get_frenet(samples[:, 0], samples[:, 1])
                # rospy.logerr(f"[{self.name}] DEBUG: Using FIXED converter for Cartesian→Frenet, samples count={samples.shape[0]}")
            else:
                s_, d_ = self.converter.get_frenet(samples[:, 0], samples[:, 1])
                # rospy.logerr(f"[{self.name}] DEBUG: Using GB converter for Cartesian→Frenet, samples count={samples.shape[0]}")
            # ===== HJ MODIFIED END =====

            psi_, kappa_ = tph.calc_head_curv_num.\
                calc_head_curv_num(
                    path=samples,
                    el_lengths=0.1*np.ones(len(samples)-1),
                    is_closed=False
                )

            # ===== HJ ADDED: Convert to GB Frenet for final waypoint output =====
            # IMPORTANT: Even in Fixed mode, output waypoints must have GB Frenet (s, d)
            # because State Machine expects all waypoints to use GB as reference
            s_gb, d_gb = self.converter.get_frenet(samples[:, 0], samples[:, 1])
            # ===== HJ ADDED END =====

            # ===== HJ ADDED: Track bounds check results for marker visualization =====
            bounds_check_results = []
            # ===== HJ ADDED END =====

            danger_flag = False
            for i in range(samples.shape[0]):
                # ===== HJ MODIFIED: Use ref_max_idx (len of current reference path) =====
                gb_wpnt_i = int((s_[i] / wpnt_dist) % ref_max_idx)
                # ===== HJ MODIFIED END =====

                # ===== HJ MODIFIED: Use appropriate map filter =====
                # Use modified map filter when using fixed path, original map otherwise
                if self.use_fixed_path and self.map_filter_fixed is not None:
                    inside = self.map_filter_fixed.is_point_inside(samples[i, 0], samples[i, 1])
                else:
                    inside = self.map_filter.is_point_inside(samples[i, 0], samples[i, 1])
                # ===== HJ MODIFIED END =====

                # ===== HJ ADDED: Track bounds check result =====
                bounds_check_results.append(inside)
                # ===== HJ ADDED END =====

                if not inside:
                    # ===== HJ MODIFIED: Enhanced debug logging =====
                    # rospy.logwarn(
                    #     f"[{self.name}] DEBUG ABORT #2: Point {i}/{samples.shape[0]} OUTSIDE bounds! "
                    #     f"xy=({samples[i, 0]:.2f}, {samples[i, 1]:.2f}), "
                    #     f"Filter: {'OBSTACLES_ONLY' if (self.use_fixed_path and self.map_filter_fixed) else 'ORIGINAL'}")
                    # ===== HJ MODIFIED END =====
                    danger_flag = True
                    break
                outside = True
                # Get V from gb wpnts and go slower if we are going through the inside
                vi = gb_wpnts[gb_wpnt_i].vx_mps if outside else gb_wpnts[gb_wpnt_i].vx_mps * 0.9 # TODO make speed scaling ros param

                # ===== HJ MODIFIED: Use GB Frenet coordinates for waypoint output =====
                wpnts.wpnts.append(
                    self.xyv_to_wpnts(x=samples[i, 0], y=samples[i, 1], s=s_gb[i], d=d_gb[i], v=2, psi=psi_[i] + np.pi/2 , kappa= kappa_[i], wpnts=wpnts)
                )
                # ===== HJ MODIFIED END =====
                mrks.markers.append(self.xyv_to_markers(x=samples[i, 0], y=samples[i, 1], v=vi, mrks=mrks))

            # Fill the rest of OTWpnts

            # ===== HJ ADDED: Publish spline sample markers for debugging =====
            self._publish_spline_samples_markers(samples, bounds_check_results)
            # ===== HJ ADDED END =====

            if danger_flag:
                # ===== HJ ADDED: Debug log for danger flag abort =====
                # rospy.logerr(f"[{self.name}] DEBUG ABORT #3: danger_flag=True, returning empty waypoints")
                # ===== HJ ADDED END =====
                wpnts.wpnts = []
                mrks.markers = []
            else:
                # ===== HJ ADDED: Success log =====
                # rospy.logerr(f"[{self.name}] DEBUG SUCCESS: Generated {len(wpnts.wpnts)} waypoints successfully!")
                # ===== HJ ADDED END =====
                pass  # All checks passed, wpnts already populated
        return wpnts, mrks

    def _obs_filtering(self, obstacles: ObstacleArray) -> List[Obstacle]:
        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        obs_on_traj = [obs for obs in obstacles.obstacles if abs(obs.d_center) < self.obs_traj_tresh]

        # Only use obstacles that within self.lookahead in front of the car
        close_obs = []
        for obs in obs_on_traj:
            obs = self._predict_obs_movement(obs)
            # Handle wraparound
            dist_in_front = (obs.s_center - self.cur_s) % self.gb_max_s
            # dist_in_back = abs(dist_in_front % (-self.gb_max_s)) # distance from ego to obstacle in the back
            if dist_in_front < self.lookahead:
                close_obs.append(obs)
                # Not within lookahead
            else:
                pass
        return close_obs

    def _predict_obs_movement(self, obs: Obstacle, mode: str = "constant") -> Obstacle:
        """
        Predicts the movement of an obstacle based on the current state and mode.
        
        TODO: opponent prediction should be completely isolated for added modularity       

        Args:
            obs (Obstacle): The obstacle to predict the movement for.
            mode (str, optional): The mode for predicting the movement. Defaults to "constant".

        Returns:
            Obstacle: The updated obstacle with the predicted movement.
        """
        # propagate opponent by time dependent on distance
        if (obs.s_center - self.cur_s) % self.gb_max_s < 10:  # TODO make param
            if mode == "adaptive":
                # distance in s coordinate
                cur_s = self.cur_s
                ot_distance = (obs.s_center - cur_s) % self.gb_max_s
                rel_speed = np.clip(self.gb_scaled_wpnts.wpnts[int(cur_s * 10)].vx_mps - obs.vs, 0.1, 10)
                ot_time_distance = np.clip(ot_distance / rel_speed, 0, 5) * 0.5

                delta_s = ot_time_distance * obs.vs
                delta_d = ot_time_distance * obs.vd
                delta_d = -(obs.d_center + delta_d) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            elif mode == "adaptive_velheuristic":
                opponent_scaler = 0.7
                cur_s = self.cur_s
                ego_speed = self.gb_scaled_wpnts.wpnts[int(cur_s * 10)].vx_mps

                # distance in s coordinate
                ot_distance = (obs.s_center - cur_s) % self.gb_max_s
                rel_speed = (1 - opponent_scaler) * ego_speed
                ot_time_distance = np.clip(ot_distance / rel_speed, 0, 5)

                delta_s = ot_time_distance * opponent_scaler * ego_speed
                delta_d = -(obs.d_center) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            # propagate opponent by constant time
            elif mode == "constant":
                delta_s = self.fixed_pred_time * obs.vs
                delta_d = self.fixed_pred_time * obs.vd
                # delta_d = -(obs.d_center+delta_d)*np.exp(-np.abs(self.kd_obs_pred*obs.d_center))

            elif mode == "heuristic":
                # distance in s coordinate
                ot_distance = (obs.s_center - self.cur_s) % self.gb_max_s
                rel_speed = 3
                ot_time_distance = ot_distance / rel_speed

                delta_d = ot_time_distance * obs.vd
                delta_d = -(obs.d_center + delta_d) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            # update
            obs.s_start += delta_s
            obs.s_center += delta_s
            obs.s_end += delta_s
            obs.s_start %= self.gb_max_s
            obs.s_center %= self.gb_max_s
            obs.s_end %= self.gb_max_s

            obs.d_left += delta_d
            obs.d_center += delta_d
            obs.d_right += delta_d

            resp = self.converter.get_cartesian([obs.s_center], [obs.d_center])

            marker = self.xy_to_point(resp[0], resp[1], opponent=True)
            self.pub_propagated.publish(marker)

        return obs
    
    def _check_ot_side_possible(self, more_space) -> bool:
        if abs(self.cur_d) > 0.25 and more_space != self.last_ot_side: # TODO make rosparam for cur_d threshold
            rospy.loginfo(f"[{self.name}]: Can't switch sides, because we are not on the raceline")
            return False
        return True

    ######################
    # VIZ + MSG WRAPPING #
    ######################
    def xyv_to_markers(self, x:float, y:float, v:float, mrks: MarkerArray, is_fixed_path: bool = False) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = rospy.Time.now()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = v / self.gb_vmax
        mrk.color.a = 1.0

        # ===== HJ MODIFIED: Different colors for fixed path vs real-time spline =====
        if is_fixed_path:
            # Fixed path: Green
            mrk.color.r = 0.0
            mrk.color.g = 0.5
            mrk.color.b = 1.0
        else:
            # Real-time spline: Purple/Pink
            mrk.color.r = 0.75
            mrk.color.g = 0.75 if self.from_bag else 0.0
            mrk.color.b = 0.75
        # ===== HJ MODIFIED END =====

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = x
        mrk.pose.position.y = y
        mrk.pose.position.z = v / self.gb_vmax / 2
        mrk.pose.orientation.w = 1

        return mrk

    def xy_to_point(self, x: float, y: float, opponent=True) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = rospy.Time.now()
        mrk.type = mrk.SPHERE
        mrk.scale.x = 0.5
        mrk.scale.y = 0.5
        mrk.scale.z = 0.5
        mrk.color.a = 0.8
        mrk.color.b = 0.65
        mrk.color.r = 1 if opponent else 0
        mrk.color.g = 0.65

        mrk.pose.position.x = x
        mrk.pose.position.y = y
        mrk.pose.position.z = 0.01
        mrk.pose.orientation.w = 1

        return mrk

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, psi: float, kappa: float ,wpnts: WpntArray) -> Wpnt:
        wpnt = Wpnt()
        wpnt.id = len(wpnts.wpnts)
        wpnt.x_m = x
        wpnt.y_m = y
        wpnt.s_m = s
        wpnt.d_m = d
        wpnt.vx_mps = v
        wpnt.psi_rad = psi
        wpnt.kappa_radpm = kappa
        

        return wpnt

    ###########################
    # PHASE 3: GB OPTIMIZER   #
    ###########################
    def _generate_fixed_path_async(self, verified_obs: List[Tuple[int, int, float, float]]):
        """
        Async wrapper for _generate_fixed_path. Runs in separate thread.

        This wrapper:
        1. Calls _generate_fixed_path (blocking GB optimizer)
        2. Updates self.fixed_path_generated flag
        3. Clears self.fixed_path_generating flag

        Args:
            verified_obs: List of (sector_id, obs_id, s, d) tuples for stable interfering obstacles
        """
        try:
            rospy.loginfo(f"[{self.name}] [THREAD] Starting GB optimizer path generation...")
            success = self._generate_fixed_path(verified_obs)

            if success:
                rospy.loginfo(f"[{self.name}] [THREAD] Fixed path generated successfully!")
                self.fixed_path_generated = True  # Static flag: generation completed
                self.use_fixed_path = True  # Dynamic flag: start using fixed path
                # ===== HJ ADDED: Publish flag change =====
                self.use_fixed_path_pub.publish(Bool(data=True))
                # ===== HJ ADDED END =====
                # ===== HJ ADDED: Start adaptive timeout decay =====
                self.last_timeout_decay_time = rospy.Time.now()
                rospy.loginfo(f"[{self.name}] Started adaptive memory timeout decay (initial={self.memory_timeout_sec}s)")
                # ===== HJ ADDED END =====
            else:
                rospy.logerr(f"[{self.name}] [THREAD] Fixed path generation FAILED")
                # Don't set generated flag, will retry next iteration

        except Exception as e:
            rospy.logerr(f"[{self.name}] [THREAD] Exception in path generation thread: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Always clear generating flag when done (success or fail)
            self.fixed_path_generating = False
            rospy.loginfo(f"[{self.name}] [THREAD] Path generation thread finished.")

    def _generate_fixed_path(self, verified_obs: List[Tuple[int, int, float, float]]) -> bool:
        """
        Phase 3: Generate fixed optimized path using GB optimizer.

        Uses same approach as mapping.launch + global_planner_node.py:
        - Creates CSV in current working directory (usually ~/.ros/ or ~/catkin_ws/)
        - Reads racecar_f110.ini from stack_master/config/{racecar_version}
        - Calls trajectory_optimizer with mincurv_iqp

        Args:
            verified_obs: List of (sector_id, obs_id, s, d) tuples for stable interfering obstacles

        Returns:
            True if path generation successful, False otherwise
        """
        # Temporary CSV filename (relative path, created in current working directory)
        csv_filename = 'smart_static_avoidance_temp.csv'

        try:
            rospy.loginfo(f"[{self.name}] Phase 3: Starting GB optimizer path generation...")

            # Retry with progressively reduced safety_width ratios if optimization fails
            # Obstacles are already added to occupancy grid, so only adjust safety_width
            safety_width_ratios = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

            trajectory_opt = None
            last_error = None

            for idx, ratio in enumerate(safety_width_ratios):
                try:
                    safety_width_adjusted = self.safety_width * ratio
                    rospy.loginfo(
                        f"[{self.name}] Attempt {idx+1}/{len(safety_width_ratios)}: "
                        f"safety_width={safety_width_adjusted:.3f}m (ratio={ratio:.2f})"
                    )

                    # Step 1: Prepare reftrack using occupancy grid approach
                    # (Same method as mapping.launch - more accurate than Frenet-based approach)
                    reftrack = self._prepare_reftrack_from_occupancy_grid(verified_obs)
                    if reftrack is None:
                        rospy.logerr(f"[{self.name}] Failed to prepare reftrack from occupancy grid")
                        continue

                    rospy.loginfo(f"[{self.name}] Reftrack prepared: {reftrack.shape[0]} points")

                    # Step 2: Save to temporary CSV in current working directory
                    self._save_reftrack_to_csv(reftrack, csv_filename)

                    # Step 3: Call GB optimizer
                    rospy.loginfo(f"[{self.name}] Calling GB optimizer (mincurv_iqp)...")
                    trajectory_opt, bound_r, bound_l, est_time = self._call_gb_optimizer(
                        csv_filename,
                        safety_width=safety_width_adjusted
                    )
                    rospy.loginfo(
                        f"[{self.name}] GB optimizer SUCCESS! "
                        f"Trajectory: {trajectory_opt.shape[0]} points, Est time: {est_time:.2f}s"
                    )
                    break  # Success! Exit retry loop

                except ValueError as e:
                    if "constraints are inconsistent" in str(e):
                        rospy.logwarn(
                            f"[{self.name}] Attempt {idx+1} failed (constraints inconsistent). "
                            f"Trying next configuration..."
                        )
                        last_error = e
                        continue  # Try next configuration
                    else:
                        raise  # Re-raise if different error

            # Check if all attempts failed
            if trajectory_opt is None:
                rospy.logerr(f"[{self.name}] GB optimizer failed with all {len(safety_width_ratios)} configurations")
                if last_error:
                    raise last_error
                else:
                    return False

            # Step 4: Package into OTWpntArray with visualization markers
            self.fixed_path_wpnts, self.fixed_path_markers = self._package_to_otwpntarray(trajectory_opt)
            rospy.loginfo(f"[{self.name}] Fixed path packaged: {len(self.fixed_path_wpnts.wpnts)} waypoints, {len(self.fixed_path_markers.markers)} markers")

            # ===== HJ ADDED: Remove last waypoint to prevent closed loop issues =====
            if len(self.fixed_path_wpnts.wpnts) > 0:
                original_count = len(self.fixed_path_wpnts.wpnts)
                self.fixed_path_wpnts.wpnts = self.fixed_path_wpnts.wpnts[:-1]
                rospy.loginfo(f"[{self.name}] Removed last waypoint for open loop: {original_count} → {len(self.fixed_path_wpnts.wpnts)} waypoints")
            # ===== HJ ADDED END =====

            # Step 4.5: Create FrenetConverter for fixed path (for interference checking)
            x_array = np.array([w.x_m for w in self.fixed_path_wpnts.wpnts])
            y_array = np.array([w.y_m for w in self.fixed_path_wpnts.wpnts])
            self.fixed_converter = FrenetConverter(x_array, y_array)
            rospy.loginfo(f"[{self.name}] Created FrenetConverter for fixed path ({len(x_array)} points)")

            # ===== HJ ADDED: Calculate d_left, d_right for Fixed path waypoints =====
            # Step 4.5.5: Extract track boundaries and calculate d_left, d_right
            obstacles_only_png_path = os.path.join(self.map_dir, f'{self.map_name}_obstacles_only.png')
            modified_yaml_path = os.path.join(self.map_dir, f'{self.map_name}.yaml')
            if os.path.exists(obstacles_only_png_path):
                self._calculate_fixed_path_boundaries(obstacles_only_png_path, modified_yaml_path, verified_obs)
            else:
                rospy.logwarn(f"[{self.name}] Cannot calculate d_left/d_right: obstacles_only map not found")
            # ===== HJ ADDED END =====

            # Step 4.6: Create GridFilter for obstacles-only map (for do_spline bounds checking)
            # ===== HJ MODIFIED: Use obstacles_only.png instead of modified_result.png =====
            # (paths already defined above)
            if os.path.exists(obstacles_only_png_path):
                self.map_filter_fixed = GridFilter(map_topic=None, debug=False)
                if self.map_filter_fixed.load_from_file(obstacles_only_png_path, modified_yaml_path):
                    self.map_filter_fixed.set_erosion_kernel_size(self.kernel_size)
                    rospy.loginfo(f"[{self.name}] Created GridFilter for obstacles-only map: {obstacles_only_png_path}")

                    # ===== HJ ADDED: Publish obstacles-only map for visualization (ONE-TIME, latch=True) =====
                    self._publish_obstacles_only_map(obstacles_only_png_path, modified_yaml_path)
                    rospy.loginfo(f"[{self.name}] Published obstacles-only map to /map_with_obs (latched)")
                    # ===== HJ ADDED END =====
                else:
                    rospy.logwarn(f"[{self.name}] Failed to load obstacles-only map, will use original map filter")
                    self.map_filter_fixed = None
            else:
                rospy.logwarn(f"[{self.name}] Obstacles-only map not found at {obstacles_only_png_path}, will use original map filter")
                self.map_filter_fixed = None
            # ===== HJ MODIFIED END =====

            # Step 5: Save to JSON file (for future use)
            # Load ORIGINAL wall boundaries (from global_waypoints.json, without obstacles)
            bound_r_original, bound_l_original = self._get_original_wall_boundaries()
            if bound_r_original is not None and bound_l_original is not None:
                self._save_smart_global_waypoints_json(
                    trajectory_wpnts=self.fixed_path_wpnts,
                    trajectory_markers=self.fixed_path_markers,
                    bound_r_original=bound_r_original,
                    bound_l_original=bound_l_original,
                    est_time=est_time
                )
            else:
                rospy.logwarn(f"[{self.name}] Failed to load original boundaries for JSON save")

            # Step 6: Clean up temporary CSV file
            if os.path.exists(csv_filename):
                os.remove(csv_filename)
                rospy.loginfo(f"[{self.name}] Cleaned up temporary CSV file")

            return True

        except Exception as e:
            rospy.logerr(f"[{self.name}] Exception in _generate_fixed_path: {e}")
            import traceback
            traceback.print_exc()

            # Clean up temporary CSV file on error
            if os.path.exists(csv_filename):
                try:
                    os.remove(csv_filename)
                    rospy.loginfo(f"[{self.name}] Cleaned up temporary CSV file after error")
                except:
                    pass

            return False

    def _prepare_reftrack_from_occupancy_grid(
        self,
        verified_obs: List[Tuple[int, int, float, float]]
    ) -> np.ndarray:
        """
        Prepare reftrack using occupancy grid + contour extraction (like mapping.launch).
        This is more accurate than Frenet-based approach.

        Steps:
        1. Load map PNG and YAML
        2. Convert obstacles (s, d) to pixel coordinates
        3. Draw obstacles on occupancy grid (0.5m circle + 0.4m line to nearest wall)
        4. Extract centerline using skeletonize
        5. Extract bounds using watershed
        6. Calculate reftrack distances

        Returns:
            reftrack: [x_m, y_m, w_tr_right_m, w_tr_left_m]
        """
        try:
            # Step 1: Load map PNG and YAML
            map_path = os.path.join(
                os.path.dirname(__file__),
                '../../../stack_master/maps',
                self.map_name,
                f'{self.map_name}.png'
            )
            yaml_path = os.path.join(
                os.path.dirname(__file__),
                '../../../stack_master/maps',
                self.map_name,
                f'{self.map_name}.yaml'
            )

            if not os.path.exists(map_path) or not os.path.exists(yaml_path):
                rospy.logerr(f"[{self.name}] Map files not found: {map_path}, {yaml_path}")
                return None

            # Load map image
            og_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            if og_img is None:
                rospy.logerr(f"[{self.name}] Failed to load map image: {map_path}")
                return None

            # Load map metadata
            with open(yaml_path, 'r') as f:
                map_data = yaml.safe_load(f)

            map_resolution = map_data['resolution']  # m/pixel
            map_origin_x = map_data['origin'][0]     # meters
            map_origin_y = map_data['origin'][1]     # meters

            rospy.loginfo(f"[{self.name}] Loaded map: {og_img.shape}, res={map_resolution}, origin=({map_origin_x}, {map_origin_y})")

            # Create binary mask (white = free space, black = occupied)
            bw = np.where(og_img > 200, 255, 0).astype(np.uint8)

            # Step 2 & 3: Add obstacles to map
            # ===== HJ MODIFIED: Generate two versions - obstacles only + obstacles with wall lines =====
            modified_bw, obstacles_only_bw = self._add_obstacles_to_occupancy_grid(
                bw, verified_obs, map_resolution, map_origin_x, map_origin_y
            )

            # DEBUG: Save modified occupancy grid (with wall lines - for skeletonize)
            debug_path = os.path.join(self.map_dir, f'{self.map_name}_modified_result.png')
            cv2.imwrite(debug_path, modified_bw)
            rospy.loginfo(f"[{self.name}] Saved modified occupancy grid (with wall lines) to {debug_path}")

            # DEBUG: Save obstacles-only occupancy grid (for GridFilter)
            obstacles_only_path = os.path.join(self.map_dir, f'{self.map_name}_obstacles_only.png')
            cv2.imwrite(obstacles_only_path, obstacles_only_bw)
            rospy.loginfo(f"[{self.name}] Saved obstacles-only occupancy grid to {obstacles_only_path}")
            # ===== HJ MODIFIED END =====

            # Step 4: Extract centerline using skeletonize
            skeleton = skeletonize(modified_bw // 255).astype(np.uint8) * 255

            # DEBUG: Save skeleton
            skeleton_path = os.path.join(self.map_dir, f'{self.map_name}_skeleton.png')
            cv2.imwrite(skeleton_path, skeleton)
            rospy.loginfo(f"[{self.name}] Saved skeleton to {skeleton_path}")

            centerline_pixels = self._extract_centerline_from_skeleton(skeleton)

            if centerline_pixels is None:
                rospy.logerr(f"[{self.name}] Failed to extract centerline")
                return None

            # Step 5: Extract bounds using watershed
            bound_r_pixels, bound_l_pixels = self._extract_bounds_from_watershed(
                centerline_pixels, modified_bw
            )

            if bound_r_pixels is None or bound_l_pixels is None:
                rospy.logerr(f"[{self.name}] Failed to extract bounds")
                return None

            # Convert to meters
            map_height = og_img.shape[0]  # Image height in pixels
            centerline_m = self._pixels_to_meters(centerline_pixels, map_resolution, map_origin_x, map_origin_y, map_height)
            bound_r_m = self._pixels_to_meters(bound_r_pixels, map_resolution, map_origin_x, map_origin_y, map_height)
            bound_l_m = self._pixels_to_meters(bound_l_pixels, map_resolution, map_origin_x, map_origin_y, map_height)

            # Step 6: Calculate reftrack distances
            reftrack = self._calculate_reftrack_distances(centerline_m, bound_r_m, bound_l_m)

            rospy.loginfo(f"[{self.name}] Prepared reftrack from occupancy grid: {reftrack.shape[0]} points")
            return reftrack

        except Exception as e:
            rospy.logerr(f"[{self.name}] Failed to prepare reftrack from occupancy grid: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_original_wall_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load original wall boundaries from global_waypoints.json (cached)."""
        # Return cached values if already loaded
        if self._original_bound_r is not None and self._original_bound_l is not None:
            return self._original_bound_r, self._original_bound_l

        try:
            json_path = os.path.join(
                os.path.dirname(__file__),
                '../../../stack_master/maps',
                self.map_name,
                'global_waypoints.json'
            )

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract from markers array by color
            markers = data['trackbounds_markers']['markers']
            bound_right = []  # purple (0.5, 0.0, 0.5)
            bound_left = []   # green (0.5, 1.0, 0.0)

            for m in markers:
                pos = m['pose']['position']
                color = (m['color']['r'], m['color']['g'], m['color']['b'])

                if color == (0.5, 0.0, 0.5):  # Purple = Right
                    bound_right.append([pos['x'], pos['y']])
                elif color == (0.5, 1.0, 0.0):  # Green = Left
                    bound_left.append([pos['x'], pos['y']])

            bound_r = np.array(bound_right)
            bound_l = np.array(bound_left)

            # Cache the results
            self._original_bound_r = bound_r
            self._original_bound_l = bound_l

            rospy.loginfo(f"[{self.name}] Loaded wall boundaries: right={len(bound_r)}, left={len(bound_l)}")
            return bound_r, bound_l

        except Exception as e:
            rospy.logerr(f"[{self.name}] Failed to load wall boundaries: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _save_reftrack_to_csv(self, reftrack: np.ndarray, csv_filename: str):
        """
        Save reftrack array to CSV file for GB optimizer input.

        Args:
            reftrack: [x_m, y_m, w_tr_right_m, w_tr_left_m]
            csv_filename: Relative filename (will be saved in current working directory)
        """
        np.savetxt(
            csv_filename,
            reftrack,
            delimiter=',',
            fmt='%.6f'
        )
        rospy.loginfo(f"[{self.name}] Saved reftrack to {os.path.abspath(csv_filename)}: {reftrack.shape[0]} points")

    def _call_gb_optimizer(
        self,
        csv_filename: str,
        safety_width: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Call GB optimizer trajectory_optimizer with mincurv_iqp.
        Uses same approach as mapping.launch + global_planner_node.py.

        Args:
            csv_filename: Relative filename in current working directory (e.g., 'smart_static_avoidance_temp.csv')
            safety_width: Override safety_width parameter (uses self.safety_width if None)

        Returns:
            trajectory_opt: [s_m, x_m, y_m, psi_rad, kappa, vx_mps, ax_mps2]
            bound_r: Right boundary XY coordinates
            bound_l: Left boundary XY coordinates
            est_time: Estimated lap time (s)
        """
        if safety_width is None:
            safety_width = self.safety_width
        # Get input_path for racecar_f110.ini location (same as global_planner_node.py)
        rospack = RosPack()
        stack_master_path = rospack.get_path('stack_master')
        input_path = os.path.join(stack_master_path, 'config', self.racecar_version)

        # Create temporary directory with modified ini file (tighter convergence)
        import tempfile
        import shutil
        import configparser

        temp_dir = tempfile.mkdtemp(prefix='smart_avoidance_')
        temp_input_path = temp_dir

        try:
            # Copy entire config directory structure (ini + veh_dyn_info/ + devices/, etc.)
            # trajectory_optimizer needs veh_dyn_info/ggv.csv and other files
            shutil.copytree(input_path, temp_input_path, dirs_exist_ok=True)

            # Modify convergence criteria (tighter tolerance for faster exit after min iterations)
            temp_ini = os.path.join(temp_input_path, 'racecar_f110.ini')

            # Read entire ini file as text (can't use configparser - values are JSON)
            with open(temp_ini, 'r') as f:
                ini_content = f.read()

            # Find and replace the entire optim_opts_mincurv section
            import re

            # Log original
            original_match = re.search(r'optim_opts_mincurv=\{[^}]+\}', ini_content, re.DOTALL)
            if original_match:
                rospy.loginfo(f"[{self.name}] BEFORE: {original_match.group()}")

            # Replace the entire optim_opts_mincurv dictionary with tighter settings
            # This is more robust than trying to replace individual values
            replacement = 'optim_opts_mincurv={"width_opt": 0.8,\n                    "iqp_iters_min": 15,\n                    "iqp_curverror_allowed": 0.01}'

            ini_content_modified = re.sub(
                r'optim_opts_mincurv=\{[^}]+\}',
                replacement,
                ini_content,
                flags=re.DOTALL
            )

            # Verify the replacement worked
            modified_match = re.search(r'optim_opts_mincurv=\{[^}]+\}', ini_content_modified, re.DOTALL)
            if modified_match:
                rospy.loginfo(f"[{self.name}] AFTER: {modified_match.group()}")

            # Write modified content back
            with open(temp_ini, 'w') as f:
                f.write(ini_content_modified)

            rospy.loginfo(f"[{self.name}] Wrote modified ini to: {temp_ini}")

        except Exception as e:
            rospy.logwarn(f"[{self.name}] Failed to create temp config: {e}, using original")
            temp_input_path = input_path

        # Extract track_name from filename (without .csv extension)
        track_name = os.path.splitext(csv_filename)[0]

        rospy.loginfo(
            f"[{self.name}] Calling trajectory_optimizer: "
            f"input_path={temp_input_path}, "
            f"track_name={track_name}, "
            f"curv_opt_type=mincurv_iqp, "
            f"safety_width={safety_width}, "
            f"plot={self.gb_optimizer_plot}"
        )

        # Call trajectory_optimizer with modified ini
        trajectory_opt, bound_r, bound_l, est_time = trajectory_optimizer(
            input_path=temp_input_path,
            track_name=track_name,
            curv_opt_type='mincurv_iqp',
            safety_width=safety_width,
            plot=self.gb_optimizer_plot
        )

        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

        rospy.loginfo(
            f"[{self.name}] GB optimizer completed: "
            f"trajectory={trajectory_opt.shape[0]} points, "
            f"bound_r={bound_r.shape[0]}, bound_l={bound_l.shape[0]}, "
            f"est_time={est_time:.2f}s"
        )

        return trajectory_opt, bound_r, bound_l, est_time

    def _package_to_otwpntarray(
        self,
        trajectory_opt: np.ndarray
    ) -> Tuple[OTWpntArray, MarkerArray]:
        """
        Package optimized trajectory into OTWpntArray message with visualization markers.

        IMPORTANT: GB optimizer trajectory uses shifted reference line,
        so we need to convert (x, y) back to ORIGINAL GB raceline Frenet coordinates (s, d).

        Uses Wpnt format (same as xyv_to_wpnts) for consistency with existing code.

        Args:
            trajectory_opt: [s_m, x_m, y_m, psi_rad, kappa, vx_mps, ax_mps2] from GB optimizer

        Returns:
            Tuple of (OTWpntArray with packaged waypoints, MarkerArray for visualization)
        """
        wpnt_array = OTWpntArray()
        wpnt_array.header.stamp = rospy.Time.now()
        wpnt_array.header.frame_id = "map"

        mrks = MarkerArray()

        # Convert trajectory XY to ORIGINAL GB raceline Frenet coordinates
        # self.converter uses original GB raceline as reference
        xy = trajectory_opt[:, 1:3]  # Extract x, y
        s_frenet, d_frenet = self.converter.get_frenet(xy[:, 0], xy[:, 1])

        for i in range(trajectory_opt.shape[0]):
            wpnt = Wpnt()
            wpnt.id = i
            wpnt.s_m = s_frenet[i]  # Original GB raceline s coordinate
            wpnt.x_m = trajectory_opt[i, 1]  # Absolute X
            wpnt.y_m = trajectory_opt[i, 2]  # Absolute Y
            wpnt.d_m = d_frenet[i]  # Original GB raceline d coordinate

            # Convert psi from trajectory_planning_helpers convention (0 = north/y-axis)
            # to ROS convention (0 = east/x-axis) by adding π/2
            # Same conversion as in global_planner_node.py line 1242 (conv_psi function)
            psi_tph = trajectory_opt[i, 3]  # tph output (0 = north)
            psi_ros = psi_tph + np.pi / 2   # Convert to ROS (0 = east)
            if psi_ros > np.pi:
                psi_ros = psi_ros - 2 * np.pi
            wpnt.psi_rad = psi_ros  # Absolute heading in ROS convention

            wpnt.kappa_radpm = trajectory_opt[i, 4]  # Curvature
            wpnt.vx_mps = trajectory_opt[i, 5]  # Velocity

            wpnt_array.wpnts.append(wpnt)

            # ===== HJ MODIFIED: Mark as fixed path for green color =====
            # Create visualization marker (same style as xyv_to_markers)
            mrk = self.xyv_to_markers(
                x=trajectory_opt[i, 1],
                y=trajectory_opt[i, 2],
                v=trajectory_opt[i, 5],
                mrks=mrks,
                is_fixed_path=True  # Green color for fixed path
            )
            mrks.markers.append(mrk)
            # ===== HJ MODIFIED END =====

        rospy.loginfo(f"[{self.name}] Packaged {len(wpnt_array.wpnts)} waypoints and {len(mrks.markers)} markers")
        return wpnt_array, mrks

    def _add_obstacles_to_occupancy_grid(
        self,
        bw: np.ndarray,
        verified_obs: List[Tuple[int, int, float, float]],
        map_resolution: float,
        map_origin_x: float,
        map_origin_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add obstacles to occupancy grid by:
        1. Drawing 0.5m diameter circle at obstacle position
        2. Drawing 0.4m thick line from circle to nearest wall (black pixel)

        Args:
            bw: Binary image (255=free, 0=occupied)
            verified_obs: List of (sector_id, obs_id, s, d)
            map_resolution: meters/pixel
            map_origin_x, map_origin_y: Map origin in meters

        Returns:
            Tuple of (modified_bw, obstacles_only_bw):
            - modified_bw: With obstacles + wall lines (for skeletonize)
            - obstacles_only_bw: With obstacles only (for GridFilter)
        """
        modified_bw = bw.copy()
        # ===== HJ ADDED: Create obstacles-only version for GridFilter =====
        obstacles_only_bw = bw.copy()
        # ===== HJ ADDED END =====
        obstacle_radius_m = 0.25  # 0.5m diameter
        line_thickness_m = 0.4    # 0.4m thick line

        obstacle_radius_px = int(obstacle_radius_m / map_resolution)
        line_thickness_px = max(1, int(line_thickness_m / map_resolution))

        for sector_id, obs_id, obs_s, obs_d in verified_obs:
            # Convert Frenet (s, d) to Cartesian (x, y)
            obs_xy = self.converter.get_cartesian(np.array([obs_s]), np.array([obs_d]))
            obs_x_m = obs_xy[0][0]
            obs_y_m = obs_xy[1][0]

            # Convert meters to pixels
            obs_x_px = int((obs_x_m - map_origin_x) / map_resolution)
            obs_y_px = int((obs_y_m - map_origin_y) / map_resolution)

            # Flip y because image coordinates are top-down
            obs_y_px = modified_bw.shape[0] - obs_y_px

            # Step 1: Draw obstacle circle (black = occupied)
            # ===== HJ MODIFIED: Draw on both versions =====
            cv2.circle(modified_bw, (obs_x_px, obs_y_px), obstacle_radius_px, 0, -1)
            cv2.circle(obstacles_only_bw, (obs_x_px, obs_y_px), obstacle_radius_px, 0, -1)  # Obstacles only
            # ===== HJ MODIFIED END =====

            # Step 2: Find nearest wall pixel (existing black pixel)
            # Create mask of existing walls (before adding this obstacle)
            wall_mask = (bw == 0).astype(np.uint8)

            # Find all wall pixels
            wall_pixels = np.column_stack(np.where(wall_mask > 0))

            if len(wall_pixels) == 0:
                rospy.logwarn(f"[{self.name}] No wall pixels found for obstacle at ({obs_x_m:.2f}, {obs_y_m:.2f})")
                continue

            # Calculate distances to all wall pixels
            distances = np.sqrt((wall_pixels[:, 0] - obs_y_px)**2 + (wall_pixels[:, 1] - obs_x_px)**2)
            nearest_idx = np.argmin(distances)
            nearest_wall_y_px = wall_pixels[nearest_idx, 0]
            nearest_wall_x_px = wall_pixels[nearest_idx, 1]

            # Step 3: Draw thick line from obstacle to nearest wall (ONLY on modified_bw, NOT on obstacles_only_bw)
            # ===== HJ MODIFIED: Line only on modified_bw for skeletonize =====
            cv2.line(
                modified_bw,
                (obs_x_px, obs_y_px),
                (nearest_wall_x_px, nearest_wall_y_px),
                0,  # Black (occupied)
                line_thickness_px
            )
            # ===== HJ MODIFIED END =====

            rospy.loginfo(
                f"[{self.name}] Added obstacle {obs_id} at pixel ({obs_x_px}, {obs_y_px}), "
                f"connected to wall at ({nearest_wall_x_px}, {nearest_wall_y_px}), "
                f"distance={distances[nearest_idx]*map_resolution:.2f}m"
            )

        # ===== HJ MODIFIED: Return both versions =====
        return modified_bw, obstacles_only_bw
        # ===== HJ MODIFIED END =====

    def _extract_centerline_from_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Extract centerline from skeleton using contour detection.
        Based on global_planner_node.py lines 698-741.

        Returns:
            centerline_pixels: [[x1, y1], [x2, y2], ...] in pixel coordinates
        """
        # Get contours from skeleton
        contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Save all closed contours
        closed_contours = []
        for i, elem in enumerate(contours):
            opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
            if not opened:
                closed_contours.append(elem)

        if len(closed_contours) == 0:
            rospy.logerr(f"[{self.name}] No closed contours found in skeleton")
            return None

        rospy.loginfo(f"[{self.name}] Found {len(closed_contours)} closed contours in skeleton")

        # Calculate line length of every contour to get the real centerline
        line_lengths = []
        for cont in closed_contours:
            line_length = 0
            for k, pnt in enumerate(cont):
                line_length += np.sqrt((pnt[0][0] - cont[k - 1][0][0]) ** 2 +
                                     (pnt[0][1] - cont[k - 1][0][1]) ** 2)
            line_lengths.append(line_length)

        # Log all contour lengths for debugging
        for i, length in enumerate(line_lengths):
            rospy.loginfo(f"[{self.name}] Contour {i}: {length:.1f} pixels")

        # Take the shortest line (innermost contour = centerline)
        # NOTE: In mapping, longest is outer wall, shortest is usually centerline
        min_length_index = np.argmin(line_lengths)
        smallest_contour = np.array(closed_contours[min_length_index]).flatten()

        rospy.loginfo(f"[{self.name}] Selected contour {min_length_index} as centerline (length={line_lengths[min_length_index]:.1f} pixels)")

        # Reshape from [x1,y1,x2,y2,...] to [[x1,y1],[x2,y2],...]
        len_reshape = int(len(smallest_contour) / 2)
        centerline = smallest_contour.reshape(len_reshape, 2)

        # Check if centerline is closed (first point should be close to last point)
        first_point = centerline[0]
        last_point = centerline[-1]
        closure_distance = np.sqrt((first_point[0] - last_point[0])**2 + (first_point[1] - last_point[1])**2)

        if closure_distance > 10.0:  # More than 10 pixels apart
            rospy.logwarn(
                f"[{self.name}] WARNING: Centerline may not be closed! "
                f"First-to-last distance: {closure_distance:.1f} pixels"
            )

        # Smooth centerline (like global_planner_node.py lines 771-788)
        centerline_smooth = self._smooth_centerline(centerline)
        centerline_smooth = np.flip(centerline_smooth, axis=0) # Default Direction (Counter-Clock)

        # ===== HJ ADDED: Flip centerline if reverse_mapping is True (from global_planner_node.py line 494) =====
        if self.reverse_mapping:
            centerline_smooth = np.flip(centerline_smooth, axis=0)
            rospy.loginfo(f"[{self.name}] Centerline flipped (reverse_mapping=True)")
        # ===== HJ ADDED END =====

        rospy.loginfo(f"[{self.name}] Extracted centerline: {len(centerline_smooth)} points")
        return centerline_smooth

    def _smooth_centerline(self, centerline: np.ndarray) -> np.ndarray:
        """Smooth centerline with Savitzky-Golay filter (from global_planner_node.py)."""
        centerline_length = len(centerline)

        if centerline_length > 2000:
            filter_length = int(centerline_length / 200) * 10 + 1
        elif centerline_length > 1000:
            filter_length = 81
        elif centerline_length > 500:
            filter_length = 41
        else:
            filter_length = 21

        centerline_smooth = savgol_filter(centerline, filter_length, 3, axis=0)

        # Apply second smoothing for end/start transition
        cen_len = int(len(centerline) / 2)
        centerline2 = np.append(centerline[cen_len:], centerline[0:cen_len], axis=0)
        centerline_smooth2 = savgol_filter(centerline2, filter_length, 3, axis=0)

        # Take points from second centerline for first centerline
        centerline_smooth[0:filter_length] = centerline_smooth2[cen_len:(cen_len + filter_length)]
        centerline_smooth[-filter_length:] = centerline_smooth2[(cen_len - filter_length):cen_len]

        return centerline_smooth

    def _extract_bounds_from_watershed(
        self,
        centerline_pixels: np.ndarray,
        filtered_bw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract track bounds using watershed algorithm.
        Based on global_planner_node.py lines 817-938.

        Returns:
            bound_r_pixels, bound_l_pixels: Boundary points in pixel coordinates
        """
        # Create centerline image
        cent_img = np.zeros((filtered_bw.shape[0], filtered_bw.shape[1]), dtype=np.uint8)
        cv2.drawContours(cent_img, [centerline_pixels.astype(int)], 0, 255, 2, cv2.LINE_8)

        # Create markers for watershed
        _, cent_markers = cv2.connectedComponents(cent_img)

        # Apply watershed algorithm
        dist_transform = cv2.distanceTransform(filtered_bw, cv2.DIST_L2, 5)
        labels = watershed(-dist_transform, cent_markers, mask=filtered_bw)

        closed_contours = []

        for label in np.unique(labels):
            if label == 0:
                continue

            # Create mask for this label
            mask = np.zeros(filtered_bw.shape, dtype="uint8")
            mask[labels == label] = 255

            # Find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            # Save all closed contours
            for i, cont in enumerate(contours):
                opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
                if not opened:
                    closed_contours.append(cont)

        # Must have exactly 2 closed contours (outer and inner bounds)
        if len(closed_contours) != 2:
            rospy.logerr(f"[{self.name}] Expected 2 track bounds, found {len(closed_contours)}")
            return None, None

        # Longest contour is outer boundary, shortest is inner
        bound_long = max(closed_contours, key=len)
        bound_long = np.array(bound_long).flatten()
        len_reshape = int(len(bound_long) / 2)
        bound_long = bound_long.reshape(len_reshape, 2)

        bound_short = min(closed_contours, key=len)
        bound_short = np.array(bound_short).flatten()
        len_reshape = int(len(bound_short) / 2)
        bound_short = bound_short.reshape(len_reshape, 2)

        rospy.loginfo(f"[{self.name}] Extracted bounds: long={len(bound_long)}, short={len(bound_short)}")

        # Determine which is right/left based on initial position
        # For now, just return as-is (we'll fix this later if needed)
        return bound_long, bound_short

    def _pixels_to_meters(
        self,
        pixels: np.ndarray,
        map_resolution: float,
        map_origin_x: float,
        map_origin_y: float,
        map_height: int
    ) -> np.ndarray:
        """Convert pixel coordinates to meters."""
        meters = np.zeros(np.shape(pixels))
        meters[:, 0] = pixels[:, 0] * map_resolution + map_origin_x
        # Flip y back (image coordinates are top-down, map coordinates are bottom-up)
        meters[:, 1] = (map_height - pixels[:, 1]) * map_resolution + map_origin_y
        return meters

    def _calculate_reftrack_distances(
        self,
        centerline_m: np.ndarray,
        bound_r_m: np.ndarray,
        bound_l_m: np.ndarray
    ) -> np.ndarray:
        """
        Calculate reftrack [x, y, w_tr_right, w_tr_left] from centerline and bounds.
        Uses XY Euclidean minimum distance (like global_planner_node.py lines 968-990).

        IMPORTANT: Interpolates bounds to 0.1m stepsize before distance calculation.
        """
        # Interpolate bounds to 0.1m stepsize (like global_planner_node.py line 972-975)
        # Add dummy columns for w_tr_right, w_tr_left (required by interp_track)
        from global_racetrajectory_optimization import helper_funcs_glob

        bound_r_tmp = np.column_stack((bound_r_m, np.zeros((bound_r_m.shape[0], 2))))
        bound_l_tmp = np.column_stack((bound_l_m, np.zeros((bound_l_m.shape[0], 2))))

        bound_r_int = helper_funcs_glob.src.interp_track.interp_track(
            reftrack=bound_r_tmp, stepsize_approx=0.1
        )
        bound_l_int = helper_funcs_glob.src.interp_track.interp_track(
            reftrack=bound_l_tmp, stepsize_approx=0.1
        )

        # Calculate distances
        N = centerline_m.shape[0]
        reftrack = np.zeros((N, 4))
        reftrack[:, 0:2] = centerline_m

        for i in range(N):
            wpnt = centerline_m[i]

            # Distance to right boundary (minimum XY Euclidean distance)
            dists_bound_right = np.sqrt(
                np.power(bound_r_int[:, 0] - wpnt[0], 2) +
                np.power(bound_r_int[:, 1] - wpnt[1], 2)
            )
            reftrack[i, 2] = np.amin(dists_bound_right)

            # Distance to left boundary
            dists_bound_left = np.sqrt(
                np.power(bound_l_int[:, 0] - wpnt[0], 2) +
                np.power(bound_l_int[:, 1] - wpnt[1], 2)
            )
            reftrack[i, 3] = np.amin(dists_bound_left)

        rospy.loginfo(
            f"[{self.name}] Calculated reftrack distances: "
            f"avg_right={np.mean(reftrack[:, 2]):.2f}m, avg_left={np.mean(reftrack[:, 3]):.2f}m"
        )

        return reftrack

    def _save_smart_global_waypoints_json(
        self,
        trajectory_wpnts: OTWpntArray,
        trajectory_markers: MarkerArray,
        bound_r_original: np.ndarray,
        bound_l_original: np.ndarray,
        est_time: float
    ):
        """
        Save optimized trajectory and original track bounds to JSON file.
        Follows the same structure as global_planner_node.py for compatibility.

        Args:
            trajectory_wpnts: OTWpntArray with waypoints (converted from trajectory_opt)
            trajectory_markers: MarkerArray for visualization
            bound_r_original: Original right boundary XY (without obstacles)
            bound_l_original: Original left boundary XY (without obstacles)
            est_time: Estimated lap time
        """
        try:
            from rospy_message_converter import message_converter
            from visualization_msgs.msg import Marker, MarkerArray
            from f110_msgs.msg import WpntArray

            # Convert OTWpntArray to WpntArray (same format as global_planner)
            trajectory_wpnts_array = WpntArray()
            trajectory_wpnts_array.header = trajectory_wpnts.header
            for wpnt in trajectory_wpnts.wpnts:
                trajectory_wpnts_array.wpnts.append(wpnt)

            # Create markers for original track bounds
            trackbounds_markers = MarkerArray()
            marker_id = 0

            # Right bound (purple)
            for i in range(0, len(bound_r_original), 10):  # Downsample for file size
                mk = Marker()
                mk.header.frame_id = "map"
                mk.ns = "bound_right"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = bound_r_original[i, 0]
                mk.pose.position.y = bound_r_original[i, 1]
                mk.pose.position.z = 0.0
                mk.scale.x = mk.scale.y = mk.scale.z = 0.05
                mk.color.r = 0.5
                mk.color.g = 0.0
                mk.color.b = 0.5
                mk.color.a = 0.8
                trackbounds_markers.markers.append(mk)

            # Left bound (green)
            for i in range(0, len(bound_l_original), 10):
                mk = Marker()
                mk.header.frame_id = "map"
                mk.ns = "bound_left"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = bound_l_original[i, 0]
                mk.pose.position.y = bound_l_original[i, 1]
                mk.pose.position.z = 0.0
                mk.scale.x = mk.scale.y = mk.scale.z = 0.05
                mk.color.r = 0.5
                mk.color.g = 1.0
                mk.color.b = 0.0
                mk.color.a = 0.8
                trackbounds_markers.markers.append(mk)

            # Build JSON dictionary (same structure as global_planner_node)
            json_data = {
                'map_info_str': {'data': f'Smart static avoidance path for {self.map_name}'},
                'est_lap_time': {'data': est_time},
                'global_traj_markers_iqp': message_converter.convert_ros_message_to_dictionary(trajectory_markers),
                'global_traj_wpnts_iqp': message_converter.convert_ros_message_to_dictionary(trajectory_wpnts_array),
                'trackbounds_markers': message_converter.convert_ros_message_to_dictionary(trackbounds_markers)
            }

            # Save to file
            json_path = os.path.join(self.map_dir, 'smart_global_waypoints.json')
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            rospy.loginfo(
                f"[{self.name}] Saved smart global waypoints to {json_path}\n"
                f"  - Trajectory waypoints: {len(trajectory_wpnts_array.wpnts)} points\n"
                f"  - Trajectory markers: {len(trajectory_markers.markers)} markers\n"
                f"  - Right bound: {len(bound_r_original)} points\n"
                f"  - Left bound: {len(bound_l_original)} points\n"
                f"  - Est time: {est_time:.2f}s"
            )

        except Exception as e:
            rospy.logwarn(f"[{self.name}] Failed to save JSON: {e}")
            import traceback
            traceback.print_exc()

    # ===== HJ ADDED: Publish obstacles-only map for visualization =====
    def _publish_obstacles_only_map(self, png_path: str, yaml_path: str):
        """
        Load obstacles-only PNG and publish as OccupancyGrid for RViz visualization.

        Args:
            png_path: Path to obstacles-only PNG file
            yaml_path: Path to YAML metadata file
        """
        try:
            import cv2
            import yaml
            from nav_msgs.msg import OccupancyGrid

            # Load PNG image
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                rospy.logerr(f"[{self.name}] Failed to load PNG: {png_path}")
                return

            # Load YAML metadata
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # Create OccupancyGrid message
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = rospy.Time.now()
            grid_msg.header.frame_id = "map"

            # Set map metadata
            grid_msg.info.resolution = yaml_data['resolution']
            grid_msg.info.width = img.shape[1]
            grid_msg.info.height = img.shape[0]
            grid_msg.info.origin.position.x = yaml_data['origin'][0]
            grid_msg.info.origin.position.y = yaml_data['origin'][1]
            grid_msg.info.origin.position.z = 0.0
            grid_msg.info.origin.orientation.w = 1.0

            # Convert image to occupancy grid data
            # PNG: 255=free (white), 0=occupied (black)
            # OccupancyGrid: 0=free, 100=occupied, -1=unknown
            flipped_img = np.flipud(img)  # Flip vertically (image origin is top-left, map origin is bottom-left)
            occupancy_data = np.zeros(img.shape, dtype=np.int8)
            occupancy_data[flipped_img == 0] = 100  # Black pixels -> occupied
            occupancy_data[flipped_img == 255] = 0  # White pixels -> free
            grid_msg.data = occupancy_data.flatten().tolist()

            # Publish (latched)
            self.map_with_obs_pub.publish(grid_msg)
            rospy.loginfo(f"[{self.name}] Published obstacles-only map: {img.shape[1]}x{img.shape[0]} @ {yaml_data['resolution']}m/px")

        except Exception as e:
            rospy.logerr(f"[{self.name}] Failed to publish obstacles-only map: {e}")
            import traceback
            traceback.print_exc()
    # ===== HJ ADDED END =====

    # ===== HJ ADDED: Publish do_spline samples as markers =====
    def _publish_spline_samples_markers(self, samples: np.ndarray, bounds_check_results: list):
        """
        Publish spline sample points as markers for debugging.

        Args:
            samples: Nx2 array of (x, y) coordinates
            bounds_check_results: List of booleans, True if point passed bounds check
        """
        try:
            markers = MarkerArray()

            for i in range(samples.shape[0]):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "spline_samples"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Position
                marker.pose.position.x = samples[i, 0]
                marker.pose.position.y = samples[i, 1]
                marker.pose.position.z = 0.1
                marker.pose.orientation.w = 1.0

                # Size
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1

                # Color: Green if passed bounds check, Red if failed
                if i < len(bounds_check_results):
                    if bounds_check_results[i]:
                        # Green - passed
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                        marker.color.a = 0.8
                    else:
                        # Red - failed
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0
                else:
                    # Blue - not checked yet (additional points)
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    marker.color.a = 0.5

                marker.lifetime = rospy.Duration(1.0)
                markers.markers.append(marker)

            self.temp_spline_markers_pub.publish(markers)
            # rospy.loginfo(f"[{self.name}] Published {len(markers.markers)} spline sample markers")

        except Exception as e:
            rospy.logerr(f"[{self.name}] Failed to publish spline sample markers: {e}")
            import traceback
            traceback.print_exc()
    # ===== HJ ADDED END =====

    def _calculate_fixed_path_boundaries(self, obstacles_only_png_path: str, yaml_path: str, verified_obs: List[Tuple[int, int, float, float]]):
        """
        Calculate d_left and d_right for Fixed path waypoints.

        Uses original wall boundaries from global_waypoints.json and adds obstacle
        circular boundaries to the closer wall.

        Args:
            obstacles_only_png_path: Not used (kept for compatibility)
            yaml_path: Not used (kept for compatibility)
            verified_obs: List of (sector_id, obs_id, s, d) tuples for obstacles
        """
        try:
            rospy.loginfo(f"[{self.name}] _calculate_fixed_path_boundaries CALLED with {len(verified_obs)} obstacles")

            # Get original wall boundaries (cached)
            bound_r_original, bound_l_original = self._get_original_wall_boundaries()

            if bound_r_original is None or bound_l_original is None:
                rospy.logerr(f"[{self.name}] Failed to load original wall boundaries - ABORTING boundary calculation")
                return

            # Start with original wall boundaries
            boundary_left = bound_l_original.copy()
            boundary_right = bound_r_original.copy()

            rospy.loginfo(f"[{self.name}] Loaded original boundaries: LEFT={len(boundary_left)}, RIGHT={len(boundary_right)}")

            # Add obstacle circular boundaries from verified_obs
            if verified_obs is not None and len(verified_obs) > 0:
                for sector_id, obs_id, obs_s, obs_d in verified_obs:
                    # Convert GB Frenet (s, d) to XY
                    obs_xy = self.converter.get_cartesian([obs_s], [obs_d])
                    obs_x = obs_xy[0][0]
                    obs_y = obs_xy[1][0]
                    obs_pos = np.array([obs_x, obs_y])

                    # Use default obstacle size (0.5m diameter)
                    obs_radius = 0.25  # 0.5m / 2
                    offset = 0.1  # 0.1m offset from obstacle edge
                    num_points = 200  # Number of points around circle

                    # Determine which wall is closer to obstacle center
                    dist_to_left = np.min(np.linalg.norm(boundary_left - obs_pos, axis=1))
                    dist_to_right = np.min(np.linalg.norm(boundary_right - obs_pos, axis=1))

                    # Generate circular boundary points around obstacle
                    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
                    circle_radius = obs_radius + offset
                    circle_points = []
                    for angle in angles:
                        x = obs_pos[0] + circle_radius * np.cos(angle)
                        y = obs_pos[1] + circle_radius * np.sin(angle)
                        circle_points.append([x, y])

                    circle_points = np.array(circle_points)

                    # Add to closer wall
                    if dist_to_left <= dist_to_right:
                        boundary_left = np.vstack([boundary_left, circle_points])
                        rospy.loginfo(f"[{self.name}] Added obstacle #{obs_id} (sector={sector_id}) to LEFT boundary (dist={dist_to_left:.2f}m, xy={obs_x:.2f},{obs_y:.2f})")
                    else:
                        boundary_right = np.vstack([boundary_right, circle_points])
                        rospy.loginfo(f"[{self.name}] Added obstacle #{obs_id} (sector={sector_id}) to RIGHT boundary (dist={dist_to_right:.2f}m, xy={obs_x:.2f},{obs_y:.2f})")

            rospy.loginfo(f"[{self.name}] Final boundaries: LEFT={len(boundary_left)} points, RIGHT={len(boundary_right)} points")

            # Calculate d_left, d_right for each Fixed path waypoint
            # Pure minimum distance (GB optimizer style - no normal filtering!)
            self.fixed_boundary_left_xy = []   # List of [x, y] for left boundary
            self.fixed_boundary_right_xy = []  # List of [x, y] for right boundary

            for i, wpnt in enumerate(self.fixed_path_wpnts.wpnts):
                wpnt_pos = np.array([wpnt.x_m, wpnt.y_m])

                # Find minimum distance to LEFT boundary
                dists_left = np.linalg.norm(boundary_left - wpnt_pos, axis=1)
                min_idx_left = np.argmin(dists_left)
                min_dist_left = dists_left[min_idx_left]
                closest_left_xy = boundary_left[min_idx_left]

                # Find minimum distance to RIGHT boundary
                dists_right = np.linalg.norm(boundary_right - wpnt_pos, axis=1)
                min_idx_right = np.argmin(dists_right)
                min_dist_right = dists_right[min_idx_right]
                closest_right_xy = boundary_right[min_idx_right]

                # Set boundary distances
                wpnt.d_left = min_dist_left
                wpnt.d_right = min_dist_right

                # Store closest boundary XY
                self.fixed_boundary_left_xy.append(closest_left_xy.copy())
                self.fixed_boundary_right_xy.append(closest_right_xy.copy())

            rospy.loginfo(f"[{self.name}] Calculated d_left/d_right for {len(self.fixed_path_wpnts.wpnts)} Fixed path waypoints")

            # Debug: Log first few waypoints
            for i in range(min(5, len(self.fixed_path_wpnts.wpnts))):
                wpnt = self.fixed_path_wpnts.wpnts[i]
                rospy.loginfo(f"  Wpnt {i}: xy=({wpnt.x_m:.2f}, {wpnt.y_m:.2f}), d_left={wpnt.d_left:.2f}, d_right={wpnt.d_right:.2f}")

        except Exception as e:
            rospy.logerr(f"[{self.name}] Failed to calculate Fixed path boundaries: {e}")
            import traceback
            traceback.print_exc()
    # ===== HJ ADDED END =====

if __name__ == "__main__":
    spliner = ObstacleSpliner()
    spliner.loop()
