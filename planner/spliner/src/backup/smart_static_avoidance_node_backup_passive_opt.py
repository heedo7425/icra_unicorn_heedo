#!/usr/bin/env python3
import time
from typing import List, Any, Tuple
import copy
import os
import sys
import csv
import json
import threading
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import BPoly
from scipy.signal import argrelextrema
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
        - `/planner/smart_avoidance/markers`: Publishes GB optimizer fixed path markers (post-fix mode).
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
        
        self.map_filter = GridFilter(map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        # ===== HJ EDITED START: Smart static avoidance memory system =====
        # IMPORTANT: Initialize BEFORE subscribing to /tracking/obstacles to avoid AttributeError
        # Phase B: Memory structure - (sector_id, obs_id) tuple keys for individual obstacle tracking
        self.static_obs_memory = {}  # {(sector_id, obs_id): {'s_history': [], 'd_history': [], 'obs_count': 0, 'interferes': bool, 'last_seen': rospy.Time}}
        self.fixed_path_generated = False
        self.fixed_path_generating = False  # Flag to track if generation is in progress
        self.fixed_path_wpnts = OTWpntArray()
        self.fixed_path_markers = MarkerArray()  # Markers for fixed path visualization
        self.min_stable_observations = 5  # 0.25 seconds at 20Hz (enough for static obs with std check)
        self.position_std_threshold = 0.05  # 5cm position stability threshold
        self.min_sectors_with_stable_obs = 2  # Need stable obstacles in 2 DIFFERENT sectors
        self.max_history_length = 100  # Keep recent 100 observations (5 seconds)
        self.memory_timeout_sec = 15.0  # Remove obstacles not seen for 15 seconds

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

        # Phase 3: Debug visualization
        self.opt_input_debug = rospy.get_param(f'/{self.name}/opt_input_debug', True)  # Enable debug marker publishing
        self.debug_data_lock = threading.Lock()  # Thread-safe access to debug data
        self.debug_data_list = []  # List of debug data snapshots: [{reftrack, verified_obs, evasion_scale, safety_width, bound_r_xy, bound_l_xy, bound_r_modified, bound_l_modified, attempt_num}]
        self.debug_marker_publishers = {}  # Dynamic publishers: {topic_name: publisher}

        # Get racecar_version and map_name for locating config files (same as mapping.launch)
        self.racecar_version = rospy.get_param('racecar_version', 'FIESTA1')
        self.map_name = rospy.get_param('/map')  # Map name from /map parameter

        # Load GB optimizer parameters (loaded from global_planner_params.yaml in launch file)
        self.safety_width = rospy.get_param('~safety_width', 0.65)  # [m] safety width for GB optimizer
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

        # Smart static avoidance publishers
        self.smart_mrks_pub = rospy.Publisher("/planner/smart_avoidance/markers", MarkerArray, queue_size=10)


        self.converter = self.initialize_converter()

        # Set the rate at which the loop runs
        self.rate = rospy.Rate(20)  # Hz

        # Start debug marker publishing thread (if enabled)
        if self.opt_input_debug:
            self.debug_thread = threading.Thread(target=self._debug_marker_thread, daemon=True)
            self.debug_thread.start()
            rospy.loginfo(f"[{self.name}] Debug marker publishing thread started (1Hz)")

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

    # Callback for scaled global waypoint topic
    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

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

            # Mode switching: use fixed path if generated, otherwise use real-time spline
            if self.fixed_path_generated:
                # POST-FIX MODE: Use fixed optimized path
                rospy.loginfo_throttle(5.0, f"[{self.name}] Using FIXED optimized path")
                wpnts = self.fixed_path_wpnts
                # Publish smart avoidance markers (separate topic)
                self.smart_mrks_pub.publish(self.fixed_path_markers)
                # Clear real-time spline markers
                del_mrk = Marker()
                del_mrk.header.stamp = rospy.Time.now()
                del_mrk.action = Marker.DELETEALL
                mrks.markers.append(del_mrk)
            else:
                # PRE-FIX MODE: Use real-time spline avoidance (like static_avoidance_planner)
                if self.obs_in_interest is not None:
                    if self.fixed_path_generating:
                        rospy.loginfo_throttle(5.0, f"[{self.name}] GENERATING fixed path in background... Using REAL-TIME spline avoidance")
                    else:
                        rospy.loginfo_throttle(5.0, f"[{self.name}] Using REAL-TIME spline avoidance (pre-fix mode)")
                    wpnts, mrks = self.do_spline(obs=copy.deepcopy(self.obs_in_interest), gb_wpnts=gb_scaled_wpnts)
                else:
                    # No obstacle in interest, delete markers
                    del_mrk = Marker()
                    del_mrk.header.stamp = rospy.Time.now()
                    del_mrk.action = Marker.DELETEALL
                    mrks.markers.append(del_mrk)
            # ===== HJ EDITED END =====

            # Publish wpnts and markers
            if self.measuring:
                end = time.perf_counter()
                self.latency_pub.publish(end - start)
            self.evasion_pub.publish(wpnts)
            self.mrks_pub.publish(mrks)
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

    # ===== HJ PHASE D: Get verified obstacles from memory (interfering only) =====
    def _get_verified_obstacles(self) -> List[Tuple[int, int, float, float]]:
        """
        Phase D: Extract verified INTERFERING obstacles from memory for path generation.

        Returns list of tuples: [(sector_id, obs_id, s_mean, d_mean), ...]
        Only includes obstacles that are:
        1. Stable (verified)
        2. Interfering with global raceline
        """
        verified_obs = []

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
                rospy.loginfo(f"[{self.name}] Verified INTERFERING obstacle: "
                             f"sector={sector_id}, id={obs_id}, s={s_mean:.2f}, d={d_mean:.2f}")

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

        # Filter: in_static_obs_sector AND is_static
        static_obs_list = [obs for obs in self.latest_obstacles
                          if obs.in_static_obs_sector and obs.is_static]

        # ===== DEBUG: Log filtering results =====
        rospy.loginfo_throttle(2.0,
            f"[{self.name}] Total: {len(self.latest_obstacles)}, "
            f"In static sector: {sum(1 for o in self.latest_obstacles if o.in_static_obs_sector)}, "
            f"is_static: {sum(1 for o in self.latest_obstacles if o.is_static)}, "
            f"Final filtered: {len(static_obs_list)}")

        # Early return if not initialized
        if self.gb_max_s is None or self.gb_wpnts.wpnts is None or len(self.gb_wpnts.wpnts) == 0:
            self.obstacles_updated = False
            return

        # Phase B: Update memory for ALL filtered obstacles (not just interfering ones)
        # Use (sector_id, obs_id) tuple keys for individual obstacle tracking
        for obs in static_obs_list:
            # Check interference (but store regardless)
            interferes = self._check_obstacle_interference(obs)

            # Find appropriate key (may merge with existing obstacle if position matches)
            key = self._find_obstacle_key_by_position(obs)

            # Initialize obstacle memory if first time
            if key not in self.static_obs_memory:
                self.static_obs_memory[key] = {
                    's_history': [],
                    'd_history': [],
                    'size_history': [],  # Store obstacle size for GB optimizer wall modification
                    'obs_count': 0,
                    'interferes': False,
                    'last_seen': rospy.Time.now()
                }

            mem = self.static_obs_memory[key]
            mem['s_history'].append(obs.s_center)
            mem['d_history'].append(obs.d_center)
            mem['size_history'].append(obs.size)  # Store size
            mem['obs_count'] += 1
            mem['interferes'] = interferes  # Update interference status
            mem['last_seen'] = rospy.Time.now()

            # Keep only recent history
            if len(mem['s_history']) > self.max_history_length:
                mem['s_history'].pop(0)
                mem['d_history'].pop(0)
                mem['size_history'].pop(0)

        # Phase B: Remove old obstacles (not seen for timeout duration)
        current_time = rospy.Time.now()
        timeout = rospy.Duration(self.memory_timeout_sec)
        keys_to_remove = [key for key, mem in self.static_obs_memory.items()
                          if (current_time - mem['last_seen']) > timeout]
        for key in keys_to_remove:
            del self.static_obs_memory[key]
            rospy.loginfo(f"[{self.name}] Removed stale obstacle: sector={key[0]}, id={key[1]}")

        # Set obs_in_interest for PRE-FIX mode
        if not self.fixed_path_generated and static_obs_list:
            closest_obs = min(static_obs_list,
                            key=lambda o: (o.s_center - self.cur_s) % self.gb_max_s)
            self.obs_in_interest = closest_obs
        elif not self.fixed_path_generated:
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
        left_gap = abs(gb_wpnts[obs_s_idx].d_left - obstacle.d_left)
        right_gap = abs(gb_wpnts[obs_s_idx].d_right + obstacle.d_right)
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
        # Get spacing between wpnts for rough approximations
        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m

        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        # close_obs = self._obs_filtering(obstacles=obstacles)

        # If there are obstacles within the lookahead distance, then we need to generate an evasion trajectory considering the closest one
        if obs.is_static == True:
            pre_dist = (obs.s_center - self.cur_s) % self.gb_max_s
            
            if pre_dist < 0.5 or pre_dist > self.gb_max_s / 2:
                wpnts.wpnts = []
                mrks.markers = []
                return wpnts, mrks
            
            obs_s_idx = int(obs.s_center / wpnt_dist) % self.gb_max_idx

            more_space, d_apex = self._more_space(obs, gb_wpnts, obs_s_idx)
            s_list = [obs.s_center]
            d_list = [d_apex]
            
            post_dist = min(min(max(pre_dist, self.post_min_dist), self.post_max_dist), self.gb_max_s / 2)

            num_post_ref = int((post_dist // self.sampling_dist)) + 1

            for i in range(num_post_ref):
                s_list.append(obs.s_center + post_dist * ((i + 1)/ num_post_ref))
                d_list.append((d_apex * (1 - (i + 1)/ num_post_ref)))
                            
            # evasion_s = np.array([self.cur_s +1 ,self.cur_s +2])
            s_array = np.array(s_list)
            d_array = np.array(d_list)
            
            s_array = s_array % self.gb_max_s
            
            s_idx = np.round((s_array / wpnt_dist)).astype(int) % self.gb_max_idx
            # evasion2 = int(evasion_s[1] / wpnt_dist) % self.gb_max_idx
            
            # gb_idxs = [evasion1, evasion2]
            
            # Choose the correct side and compute the distance to the apex based on left of right of the obstacle

            # evasion_d = np.array([obs.d_center + (obs.size/2 + 0.5)  , 0])
            # evasion_d = np.array([d_apex  , 0])
            
            # Do frenet conversion via conversion service for spline and create markers and wpnts
            danger_flag = False
            resp = self.converter.get_cartesian(s_array, d_array)

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
            xy_additional = np.array([
                (
                    gb_wpnts[(s_idx[-1] + i + 1) % self.gb_max_idx].x_m,
                    gb_wpnts[(s_idx[-1] + i + 1) % self.gb_max_idx].y_m
                )
                for i in range(n_additional)
            ])
            samples = np.vstack([samples, xy_additional])

            s_, d_ = self.converter.get_frenet(samples[:, 0], samples[:, 1])

            psi_, kappa_ = tph.calc_head_curv_num.\
                calc_head_curv_num(
                    path=samples,
                    el_lengths=0.1*np.ones(len(samples)-1),
                    is_closed=False
                )
            
            danger_flag = False
            for i in range(samples.shape[0]):
                gb_wpnt_i = int((s_[i] / wpnt_dist) % self.gb_max_idx)
                
                inside = self.map_filter.is_point_inside(samples[i, 0], samples[i, 1])
                if not inside:
                    rospy.loginfo_throttle_identical(
                        2, f"[{self.name}]: Evasion trajectory too close to TRACKBOUNDS, aborting evasion"
                    )            # if abs(evasion_d[i]) > abs(tb_dist) - self.spline_bound_mindist:
                    danger_flag = True
                    break
                outside = True
                # Get V from gb wpnts and go slower if we are going through the inside
                vi = gb_wpnts[gb_wpnt_i].vx_mps if outside else gb_wpnts[gb_wpnt_i].vx_mps * 0.9 # TODO make speed scaling ros param

                wpnts.wpnts.append(
                    self.xyv_to_wpnts(x=samples[i, 0], y=samples[i, 1], s=s_[i], d=d_[i], v=2, psi=psi_[i] + np.pi/2 , kappa= kappa_[i], wpnts=wpnts)
                )
                mrks.markers.append(self.xyv_to_markers(x=samples[i, 0], y=samples[i, 1], v=vi, mrks=mrks))

            # Fill the rest of OTWpnts

            
            if danger_flag:
                wpnts.wpnts = []
                mrks.markers = []
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
    def xyv_to_markers(self, x:float, y:float, v:float, mrks: MarkerArray) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = rospy.Time.now()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = v / self.gb_vmax
        mrk.color.a = 1.0
        mrk.color.b = 0.75
        mrk.color.r = 0.75
        if self.from_bag:
            mrk.color.g = 0.75

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
                self.fixed_path_generated = True
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

            # Try with decreasing evasion_scale and safety_width if optimization fails
            # evasion_scale: how much to shift raceline away from obstacles
            # safety_width: minimum track width for GB optimizer
            attempt_configs = [
                {'evasion_scale': 1.0, 'safety_width': self.safety_width},  # Full evasion
                {'evasion_scale': 0.8, 'safety_width': self.safety_width},  # Reduced evasion
                {'evasion_scale': 0.6, 'safety_width': self.safety_width},
                {'evasion_scale': 0.5, 'safety_width': self.safety_width},
                {'evasion_scale': 0.5, 'safety_width': 0.8},  # Reduced evasion + smaller safety
                {'evasion_scale': 0.5, 'safety_width': 0.65},
                {'evasion_scale': 0.5, 'safety_width': 0.5},
            ]

            trajectory_opt = None
            last_error = None

            for idx, config in enumerate(attempt_configs):
                try:
                    rospy.loginfo(
                        f"[{self.name}] Attempt {idx+1}/{len(attempt_configs)}: "
                        f"evasion_scale={config['evasion_scale']:.2f}, "
                        f"safety_width={config['safety_width']:.2f}m"
                    )

                    # Step 1: Prepare reftrack from GB waypoints (includes obstacle avoidance)
                    reftrack = self._prepare_reftrack_from_gb_waypoints(verified_obs, config['evasion_scale'])
                    if reftrack is None:
                        rospy.logerr(f"[{self.name}] Failed to prepare reftrack from GB waypoints")
                        continue

                    rospy.loginfo(f"[{self.name}] Reftrack prepared: {reftrack.shape[0]} points")

                    # Step 2: Save to temporary CSV in current working directory
                    self._save_reftrack_to_csv(reftrack, csv_filename)

                    # Step 2.5: Store debug data snapshot (if enabled, lightweight - no blocking)
                    if self.opt_input_debug:
                        try:
                            # Only store minimal data - processing happens in separate thread
                            debug_snapshot = {
                                'reftrack': reftrack.copy(),
                                'verified_obs': verified_obs[:],  # Shallow copy is fine
                                'evasion_scale': config['evasion_scale'],
                                'safety_width': config['safety_width'],
                                'attempt_num': idx+1,
                                'timestamp': rospy.Time.now()
                            }
                            with self.debug_data_lock:
                                self.debug_data_list.append(debug_snapshot)
                                if len(self.debug_data_list) > 10:
                                    self.debug_data_list.pop(0)
                        except:
                            pass  # Never block on debug data

                    # Step 3: Call GB optimizer
                    rospy.loginfo(f"[{self.name}] Calling GB optimizer (mincurv_iqp)...")
                    trajectory_opt, bound_r, bound_l, est_time = self._call_gb_optimizer(
                        csv_filename,
                        safety_width=config['safety_width']
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
                rospy.logerr(f"[{self.name}] GB optimizer failed with all {len(attempt_configs)} configurations")
                if last_error:
                    raise last_error
                else:
                    return False

            # Step 4: Calculate d_right, d_left using ORIGINAL boundaries (for debugging/logging only)
            rospy.loginfo(f"[{self.name}] Calculating d_right, d_left with original boundaries...")
            d_right, d_left = self._calculate_boundary_distances(trajectory_opt, verified_obs)
            rospy.loginfo(f"[{self.name}] Boundary distances calculated")

            # Step 5: Package into OTWpntArray with visualization markers
            self.fixed_path_wpnts, self.fixed_path_markers = self._package_to_otwpntarray(trajectory_opt, d_right, d_left)
            rospy.loginfo(f"[{self.name}] Fixed path packaged: {len(self.fixed_path_wpnts.wpnts)} waypoints, {len(self.fixed_path_markers.markers)} markers")

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

    def _prepare_reftrack_from_gb_waypoints(
        self,
        verified_obs: List[Tuple[int, int, float, float]],
        evasion_scale: float = 1.0
    ) -> np.ndarray:
        """
        Prepare reftrack array from GB waypoints with obstacle avoidance.
        Format: [x_m, y_m, w_tr_right_m, w_tr_left_m]

        Steps:
        1. Load original wall boundaries from JSON
        2. Shift reference line away from obstacles
        3. Add obstacle perimeters as virtual walls
        4. Create temporary FrenetConverter with shifted reference
        5. Calculate w_tr_right, w_tr_left using Frenet conversion
        """
        if len(self.gb_wpnts.wpnts) == 0:
            rospy.logerr(f"[{self.name}] No GB waypoints available!")
            return None

        # Step 1: Load original wall boundaries
        bound_r_xy, bound_l_xy = self._get_original_wall_boundaries()
        if bound_r_xy is None or bound_l_xy is None:
            rospy.logerr(f"[{self.name}] Failed to load wall boundaries!")
            return None

        # Step 2: Shift reference line away from obstacles
        shifted_ref_xy = self._shift_raceline_for_obstacles(verified_obs, bound_r_xy, bound_l_xy, evasion_scale)

        # Step 3: Add obstacle perimeters as virtual walls
        bound_r_xy_modified, bound_l_xy_modified = self._add_obstacles_to_walls(
            bound_r_xy.copy(), bound_l_xy.copy(), verified_obs
        )

        # Step 4: Create temporary FrenetConverter with shifted reference
        temp_converter = self._create_temp_frenet_converter(shifted_ref_xy)

        # Step 5: Calculate w_tr_right, w_tr_left using Frenet conversion
        N = shifted_ref_xy.shape[0]
        reftrack = np.zeros((N, 4))
        reftrack[:, 0:2] = shifted_ref_xy  # x, y

        for i in range(N):
            reftrack[i, 2] = self._get_closest_wall_distance(
                shifted_ref_xy[i], bound_r_xy_modified, temp_converter
            )
            reftrack[i, 3] = self._get_closest_wall_distance(
                shifted_ref_xy[i], bound_l_xy_modified, temp_converter
            )

        rospy.loginfo(f"[{self.name}] Prepared reftrack with {N} points, {len(verified_obs)} obstacles")
        return reftrack

    def _get_original_wall_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load original wall boundaries from global_waypoints.json."""
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

            rospy.loginfo(f"[{self.name}] Loaded wall boundaries: right={len(bound_r)}, left={len(bound_l)}")
            return bound_r, bound_l

        except Exception as e:
            rospy.logerr(f"[{self.name}] Failed to load wall boundaries: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _shift_raceline_for_obstacles(
        self,
        verified_obs: List[Tuple[int, int, float, float]],
        bound_r_xy: np.ndarray,
        bound_l_xy: np.ndarray,
        evasion_scale: float = 1.0
    ) -> np.ndarray:
        """
        Shift GB raceline away from obstacles with safety margin check.

        Strategy:
        1. For each obstacle, determine which wall is closer
        2. Shift in opposite direction with Gaussian evasion
        3. Check if shifted position is safe from opposite wall
        4. Reduce shift ratio if too close to opposite wall
        """
        N = len(self.gb_wpnts.wpnts)
        shifted_s = np.array([wpnt.s_m for wpnt in self.gb_wpnts.wpnts])
        shifted_d = np.array([wpnt.d_m for wpnt in self.gb_wpnts.wpnts])

        # Minimum safety margin to opposite wall (m)
        min_safety_margin = 0.3

        for sector_id, obs_id, obs_s, obs_d in verified_obs:
            # Get obstacle XY position
            obs_xy = self.converter.get_cartesian(np.array([obs_s]), np.array([obs_d]))
            obs_xy = np.array([obs_xy[0][0], obs_xy[1][0]])

            # Determine which wall is closer to obstacle
            dist_to_right = np.min(np.linalg.norm(bound_r_xy - obs_xy, axis=1))
            dist_to_left = np.min(np.linalg.norm(bound_l_xy - obs_xy, axis=1))

            if dist_to_right < dist_to_left:
                shift_direction = 1.0  # Shift left (away from right wall)
                opposite_wall = bound_l_xy
            else:
                shift_direction = -1.0  # Shift right (away from left wall)
                opposite_wall = bound_r_xy

            # Find waypoints in s-range around obstacle
            s_diff = np.abs(shifted_s - obs_s)
            s_diff = np.minimum(s_diff, self.gb_max_s - s_diff)  # Handle wraparound
            in_range = s_diff < self.obstacle_shift_s_range

            if not np.any(in_range):
                continue

            # Apply gradual shift with safety check
            for i in np.where(in_range)[0]:
                s_dist = s_diff[i]
                evasion_factor = np.exp(-(s_dist / self.obstacle_shift_s_range)**2)

                # Calculate desired shift amount (scaled by evasion_scale parameter)
                base_shift = shift_direction * evasion_factor * self.evasion_dist * evasion_scale

                # Try shift with safety ratio reduction
                for ratio in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                    test_d = shifted_d[i] + base_shift * ratio
                    test_xy = self.converter.get_cartesian(
                        np.array([shifted_s[i]]),
                        np.array([test_d])
                    )
                    test_xy = np.array([test_xy[0][0], test_xy[1][0]])

                    # Check distance to opposite wall
                    dist_to_opposite = np.min(np.linalg.norm(opposite_wall - test_xy, axis=1))

                    if dist_to_opposite >= min_safety_margin:
                        shifted_d[i] = test_d
                        break
                else:
                    # No safe shift found, keep original
                    rospy.logwarn(
                        f"[{self.name}] Cannot safely shift waypoint {i} for obstacle at s={obs_s:.2f}"
                    )

        # Convert shifted Frenet back to Cartesian
        shifted_xy = self.converter.get_cartesian(shifted_s, shifted_d)
        shifted_xy = np.column_stack((shifted_xy[0], shifted_xy[1]))

        rospy.loginfo(f"[{self.name}] Shifted raceline for {len(verified_obs)} obstacles")
        return shifted_xy

    def _add_obstacles_to_walls(
        self,
        bound_r_xy: np.ndarray,
        bound_l_xy: np.ndarray,
        verified_obs: List[Tuple[int, int, float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add obstacle perimeters as virtual wall points.

        For each obstacle:
        1. Get obstacle XY position and size
        2. Generate points on perimeter (100 points, radius = size/2 + 0.1m margin)
        3. Determine which wall is closer
        4. Add perimeter points to that wall's boundary
        """
        for sector_id, obs_id, obs_s, obs_d in verified_obs:
            # Get obstacle info
            obs_xy = self.converter.get_cartesian(np.array([obs_s]), np.array([obs_d]))
            obs_xy = np.array([obs_xy[0][0], obs_xy[1][0]])

            # Get obstacle size from memory (use average of all observations)
            key = (sector_id, obs_id)
            obs_size = 0.5  # Default size if not found
            if key in self.static_obs_memory:
                size_history = self.static_obs_memory[key]['size_history']
                if size_history:
                    obs_size = np.mean(size_history)  # Use average size
                    rospy.logdebug(f"[{self.name}] Obstacle {key}: avg_size={obs_size:.3f}m from {len(size_history)} observations")

            # Calculate perimeter radius with margin
            radius = obs_size / 2.0 + self.obstacle_wall_margin

            # Generate perimeter points
            angles = np.linspace(0, 2*np.pi, self.obstacle_wall_points, endpoint=False)
            perimeter_x = obs_xy[0] + radius * np.cos(angles)
            perimeter_y = obs_xy[1] + radius * np.sin(angles)
            perimeter_points = np.column_stack((perimeter_x, perimeter_y))

            # Determine which wall is closer
            dist_to_right = np.min(np.linalg.norm(bound_r_xy - obs_xy, axis=1))
            dist_to_left = np.min(np.linalg.norm(bound_l_xy - obs_xy, axis=1))

            # Add to closer wall
            if dist_to_right < dist_to_left:
                bound_r_xy = np.vstack((bound_r_xy, perimeter_points))
            else:
                bound_l_xy = np.vstack((bound_l_xy, perimeter_points))

        rospy.loginfo(
            f"[{self.name}] Added {len(verified_obs)} obstacles to walls: "
            f"right={len(bound_r_xy)}, left={len(bound_l_xy)}"
        )
        return bound_r_xy, bound_l_xy

    def _create_temp_frenet_converter(self, shifted_ref_xy: np.ndarray):
        """Create temporary FrenetConverter with shifted reference line."""
        from frenet_converter.frenet_converter import FrenetConverter
        return FrenetConverter(shifted_ref_xy[:, 0], shifted_ref_xy[:, 1])

    def _get_closest_wall_distance(
        self,
        ref_point: np.ndarray,
        wall_xy: np.ndarray,
        temp_converter
    ) -> float:
        """
        Calculate normal direction distance from reference point to wall.

        Uses Frenet conversion:
        1. Convert wall points to Frenet (s, d) using shifted reference
        2. Find wall point with same s as reference point
        3. Return |d| value as normal direction distance
        """
        # Convert wall points to Frenet coordinates
        wall_s, wall_d = temp_converter.get_frenet(wall_xy[:, 0], wall_xy[:, 1])

        # Reference point is at d=0 in its own Frenet frame
        # Find closest wall point in s-coordinate
        ref_s = temp_converter.get_frenet(
            np.array([ref_point[0]]),
            np.array([ref_point[1]])
        )[0][0]

        # Find wall point with minimum s-distance
        s_diff = np.abs(wall_s - ref_s)
        closest_idx = np.argmin(s_diff)

        # Return absolute d value as distance
        return np.abs(wall_d[closest_idx])

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

        # Extract track_name from filename (without .csv extension)
        track_name = os.path.splitext(csv_filename)[0]

        rospy.loginfo(
            f"[{self.name}] Calling trajectory_optimizer: "
            f"input_path={input_path}, "
            f"track_name={track_name}, "
            f"curv_opt_type=mincurv_iqp, "
            f"safety_width={safety_width}, "
            f"plot={self.gb_optimizer_plot}"
        )

        # Call trajectory_optimizer (same as mapping.launch does)
        # This will:
        # 1. Read racecar_f110.ini from input_path for parameters:
        #    - stepsize_opts: {"stepsize_prep": 0.05, "stepsize_reg": 0.2, "stepsize_interp_after_opt": 0.1}
        #    - optim_opts_mincurv: {"width_opt": 0.8, "iqp_iters_min": 20, "iqp_curverror_allowed": 1.0}
        # 2. Read track CSV from current working directory (csv_filename)
        # 3. Run mincurv_iqp optimization
        trajectory_opt, bound_r, bound_l, est_time = trajectory_optimizer(
            input_path=input_path,
            track_name=track_name,
            curv_opt_type='mincurv_iqp',
            safety_width=safety_width,
            plot=self.gb_optimizer_plot
        )

        rospy.loginfo(
            f"[{self.name}] GB optimizer completed: "
            f"trajectory={trajectory_opt.shape[0]} points, "
            f"bound_r={bound_r.shape[0]}, bound_l={bound_l.shape[0]}, "
            f"est_time={est_time:.2f}s"
        )

        return trajectory_opt, bound_r, bound_l, est_time

    def _calculate_boundary_distances(
        self,
        trajectory_opt: np.ndarray,
        verified_obs: List[Tuple[int, int, float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate d_right, d_left for optimized trajectory using ORIGINAL wall boundaries.

        This follows global_planner_node.py lines 984-990:
        XY Euclidean minimum distance to walls (NOT Frenet d).

        Args:
            trajectory_opt: [s_m, x_m, y_m, psi_rad, kappa, vx_mps, ax_mps2]
            verified_obs: List of verified obstacles (unused here, kept for consistency)

        Returns:
            d_right: Distance to right boundary for each trajectory point
            d_left: Distance to left boundary for each trajectory point
        """
        # Load original boundaries (without obstacle perimeters)
        bound_r_xy, bound_l_xy = self._get_original_wall_boundaries()

        N = trajectory_opt.shape[0]
        d_right = np.zeros(N)
        d_left = np.zeros(N)

        # Extract XY from trajectory_opt: [s_m, x_m, y_m, psi_rad, kappa, vx_mps, ax_mps2]
        traj_xy = trajectory_opt[:, 1:3]  # x_m, y_m

        for i in range(N):
            # Calculate XY Euclidean distance to right boundary
            dists_bound_right = np.linalg.norm(bound_r_xy - traj_xy[i], axis=1)
            d_right[i] = np.min(dists_bound_right)

            # Calculate XY Euclidean distance to left boundary
            dists_bound_left = np.linalg.norm(bound_l_xy - traj_xy[i], axis=1)
            d_left[i] = np.min(dists_bound_left)

        rospy.loginfo(
            f"[{self.name}] Calculated boundary distances: "
            f"d_right min={d_right.min():.3f}m max={d_right.max():.3f}m, "
            f"d_left min={d_left.min():.3f}m max={d_left.max():.3f}m"
        )

        return d_right, d_left

    def _package_to_otwpntarray(
        self,
        trajectory_opt: np.ndarray,
        d_right: np.ndarray,
        d_left: np.ndarray
    ) -> Tuple[OTWpntArray, MarkerArray]:
        """
        Package optimized trajectory into OTWpntArray message with visualization markers.

        IMPORTANT: GB optimizer trajectory uses shifted reference line,
        so we need to convert (x, y) back to ORIGINAL GB raceline Frenet coordinates (s, d).

        Uses Wpnt format (same as xyv_to_wpnts) for consistency with existing code.

        Args:
            trajectory_opt: [s_m, x_m, y_m, psi_rad, kappa, vx_mps, ax_mps2] from GB optimizer
            d_right: Distance to right boundary (unused, kept for API consistency)
            d_left: Distance to left boundary (unused, kept for API consistency)

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
            wpnt.psi_rad = trajectory_opt[i, 3]  # Absolute heading
            wpnt.kappa_radpm = trajectory_opt[i, 4]  # Curvature
            wpnt.vx_mps = trajectory_opt[i, 5]  # Velocity

            wpnt_array.wpnts.append(wpnt)

            # Create visualization marker (same style as xyv_to_markers)
            mrk = self.xyv_to_markers(
                x=trajectory_opt[i, 1],
                y=trajectory_opt[i, 2],
                v=trajectory_opt[i, 5],
                mrks=mrks
            )
            mrks.markers.append(mrk)

        rospy.loginfo(f"[{self.name}] Packaged {len(wpnt_array.wpnts)} waypoints and {len(mrks.markers)} markers")
        return wpnt_array, mrks

    ###################################
    # DEBUG VISUALIZATION (THREADING) #
    ###################################
    def _debug_marker_thread(self):
        """
        Background thread that publishes debug markers at 1Hz.
        Runs independently from main loop to avoid blocking.
        """
        rate = rospy.Rate(1)  # 1Hz
        rospy.loginfo(f"[{self.name}] Debug marker thread running at 1Hz")

        while not rospy.is_shutdown():
            try:
                # Get debug data snapshots (thread-safe)
                with self.debug_data_lock:
                    debug_data_snapshots = self.debug_data_list.copy()

                # Publish markers for each debug data snapshot
                for debug_data in debug_data_snapshots:
                    self._publish_debug_markers(debug_data)

                rate.sleep()

            except Exception as e:
                rospy.logwarn(f"[{self.name}] Exception in debug marker thread: {e}")
                rate.sleep()

    def _publish_debug_markers(self, debug_data: dict):
        """
        Publish debug markers for one reftrack attempt.
        ALL HEAVY COMPUTATION HAPPENS HERE (in separate thread).

        Creates MarkerArray with:
        - Shifted ref path (orange spheres)
        - Obstacle positions (red cubes)
        - Original walls (black spheres, small, sparse)
        - Modified walls (purple/green spheres with obstacles)
        - Optimizer input boundaries (cyan/magenta spheres, 1:1 matched to reftrack, actual XY coordinates)
        """
        try:
            # ===== HEAVY COMPUTATION IN THREAD (won't block main loop) =====

            # Get wall boundaries
            bound_r_xy, bound_l_xy = self._get_original_wall_boundaries()
            if bound_r_xy is None or bound_l_xy is None:
                return

            # Get modified wall boundaries (with obstacles)
            bound_r_modified, bound_l_modified = self._add_obstacles_to_walls(
                bound_r_xy.copy(), bound_l_xy.copy(), debug_data['verified_obs']
            )

            # Create temporary FrenetConverter for shifted reftrack
            reftrack_xy = debug_data['reftrack'][:, 0:2]
            from frenet_converter.frenet_converter import FrenetConverter
            temp_converter = FrenetConverter(reftrack_xy[:, 0], reftrack_xy[:, 1])

            # ===== CREATE MARKERS =====

            # Create unique topic name based on attempt number and evasion scale
            # Replace . with _ for ROS topic name compatibility
            evasion_str = str(int(debug_data['evasion_scale'] * 100))  # 1.00 -> 100, 0.80 -> 80
            topic_name = f"/{self.name}/opt_input_debug_marker_ref_{debug_data['attempt_num']}_ev{evasion_str}"

            # Create publisher if doesn't exist
            if topic_name not in self.debug_marker_publishers:
                self.debug_marker_publishers[topic_name] = rospy.Publisher(
                    topic_name, MarkerArray, queue_size=1
                )
                rospy.loginfo(f"[{self.name}] Created debug marker publisher: {topic_name}")

            pub = self.debug_marker_publishers[topic_name]

            # Create MarkerArray
            marker_array = MarkerArray()
            marker_id = 0

            # 1. Shifted ref path (orange spheres)
            for i in range(len(debug_data['reftrack'])):
                mk = Marker()
                mk.header.frame_id = "map"
                mk.header.stamp = debug_data['timestamp']
                mk.ns = "reftrack"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = debug_data['reftrack'][i, 0]
                mk.pose.position.y = debug_data['reftrack'][i, 1]
                mk.pose.position.z = 0.1
                mk.scale.x = mk.scale.y = mk.scale.z = 0.05
                mk.color.r = 1.0
                mk.color.g = 0.6
                mk.color.b = 0.0
                mk.color.a = 0.8
                marker_array.markers.append(mk)

            # 2. Obstacle positions (red X, larger)
            for sector_id, obs_id, obs_s, obs_d in debug_data['verified_obs']:
                obs_xy = self.converter.get_cartesian(np.array([obs_s]), np.array([obs_d]))
                mk = Marker()
                mk.header.frame_id = "map"
                mk.header.stamp = debug_data['timestamp']
                mk.ns = "obstacles"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.CUBE
                mk.action = Marker.ADD
                mk.pose.position.x = obs_xy[0][0]
                mk.pose.position.y = obs_xy[1][0]
                mk.pose.position.z = 0.2
                mk.scale.x = mk.scale.y = mk.scale.z = 0.15
                mk.color.r = 1.0
                mk.color.g = 0.0
                mk.color.b = 0.0
                mk.color.a = 1.0
                marker_array.markers.append(mk)

            # 3. Original walls (black, very small, sparse)
            step = max(1, len(bound_r_xy) // 300)  # Downsample
            step = 1
            for i in range(0, len(bound_r_xy), step):
                mk = Marker()
                mk.header.frame_id = "map"
                mk.header.stamp = debug_data['timestamp']
                mk.ns = "wall_orig_right"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = bound_r_xy[i, 0]
                mk.pose.position.y = bound_r_xy[i, 1]
                mk.pose.position.z = 0.0
                mk.scale.x = mk.scale.y = mk.scale.z = 0.02
                mk.color.r = 0.0
                mk.color.g = 0.0
                mk.color.b = 0.0
                mk.color.a = 0.3
                marker_array.markers.append(mk)

            step = max(1, len(bound_l_xy) // 300)
            step = 1
            for i in range(0, len(bound_l_xy), step):
                mk = Marker()
                mk.header.frame_id = "map"
                mk.header.stamp = debug_data['timestamp']
                mk.ns = "wall_orig_left"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = bound_l_xy[i, 0]
                mk.pose.position.y = bound_l_xy[i, 1]
                mk.pose.position.z = 0.0
                mk.scale.x = mk.scale.y = mk.scale.z = 0.02
                mk.color.r = 0.0
                mk.color.g = 0.0
                mk.color.b = 0.0
                mk.color.a = 0.3
                marker_array.markers.append(mk)

            # 4. Modified walls with obstacles (purple/green, small)
            step = max(1, len(bound_r_modified) // 300)
            step = 1
            for i in range(0, len(bound_r_modified), step):
                mk = Marker()
                mk.header.frame_id = "map"
                mk.header.stamp = debug_data['timestamp']
                mk.ns = "wall_modified_right"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = bound_r_modified[i, 0]
                mk.pose.position.y = bound_r_modified[i, 1]
                mk.pose.position.z = 0.05
                mk.scale.x = mk.scale.y = mk.scale.z = 0.03
                mk.color.r = 0.5
                mk.color.g = 0.0
                mk.color.b = 0.5
                mk.color.a = 0.6
                marker_array.markers.append(mk)

            step = max(1, len(bound_l_modified) // 300)
            step = 1
            for i in range(0, len(bound_l_modified), step):
                mk = Marker()
                mk.header.frame_id = "map"
                mk.header.stamp = debug_data['timestamp']
                mk.ns = "wall_modified_left"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = bound_l_modified[i, 0]
                mk.pose.position.y = bound_l_modified[i, 1]
                mk.pose.position.z = 0.05
                mk.scale.x = mk.scale.y = mk.scale.z = 0.03
                mk.color.r = 0.0
                mk.color.g = 1.0
                mk.color.b = 0.0
                mk.color.a = 0.6
                marker_array.markers.append(mk)

            # 5. Optimizer input boundaries (1:1 matched to reftrack points)
            # Calculate actual XY coordinates of boundary points using Frenet transform
            for i in range(len(debug_data['reftrack'])):
                ref_x = debug_data['reftrack'][i, 0]
                ref_y = debug_data['reftrack'][i, 1]
                w_right = debug_data['reftrack'][i, 2]  # w_tr_right_m (distance)
                w_left = debug_data['reftrack'][i, 3]   # w_tr_left_m (distance)

                # Get s coordinate of this reftrack point
                s_coord, _ = temp_converter.get_frenet(
                    np.array([ref_x]), np.array([ref_y])
                )

                # Right boundary: d = +w_right (positive = right side)
                opt_w_r_xy = temp_converter.get_cartesian(s_coord, np.array([-w_right]))

                mk = Marker()
                mk.header.frame_id = "map"
                mk.header.stamp = debug_data['timestamp']
                mk.ns = "opt_bound_right"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = opt_w_r_xy[0][0]
                mk.pose.position.y = opt_w_r_xy[1][0]
                mk.pose.position.z = 0.15
                mk.scale.x = mk.scale.y = mk.scale.z = 0.04
                mk.color.r = 0.0
                mk.color.g = 1.0
                mk.color.b = 1.0
                mk.color.a = 0.8
                marker_array.markers.append(mk)

                # Left boundary: d = -w_left (negative = left side)
                opt_w_l_xy = temp_converter.get_cartesian(s_coord, np.array([w_left]))

                mk = Marker()
                mk.header.frame_id = "map"
                mk.header.stamp = debug_data['timestamp']
                mk.ns = "opt_bound_left"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = opt_w_l_xy[0][0]
                mk.pose.position.y = opt_w_l_xy[1][0]
                mk.pose.position.z = 0.15
                mk.scale.x = mk.scale.y = mk.scale.z = 0.04
                mk.color.r = 1.0
                mk.color.g = 0.0
                mk.color.b = 1.0
                mk.color.a = 0.8
                marker_array.markers.append(mk)

            # Publish
            pub.publish(marker_array)
            rospy.logdebug(f"[{self.name}] Published {len(marker_array.markers)} debug markers to {topic_name}")

        except Exception as e:
            rospy.logwarn(f"[{self.name}] Failed to publish debug markers: {e}")

if __name__ == "__main__":
    spliner = ObstacleSpliner()
    spliner.loop()
