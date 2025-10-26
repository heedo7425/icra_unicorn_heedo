#!/usr/bin/env python3
"""
Smart Static State Helper

Provides Fixed Frenet-based state checking by inheriting StateMachine.
Uses Fixed path (Smart Static) as reference instead of GB raceline.

Usage:
    checker = SmartStaticChecker(state_machine)
    checker.update()  # Extract Fixed Frenet from parent's obstacles
    close = checker._check_close_to_raceline()  # Fixed Frenet based check
"""

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from f110_msgs.msg import ObstacleArray, OTWpntArray
from state_machine_node import StateMachine


class SmartStaticChecker(StateMachine):
    """StateMachine functions but with Fixed Frenet reference

    Inherits all check functions from StateMachine.
    Overrides instance variables (cur_s, cur_d, obstacles) with Fixed Frenet values.
    Subscribes only to Fixed Frenet odom, reuses parent's obstacles and waypoints.
    """

    def __init__(self, parent_state_machine):
        """Initialize with Fixed Frenet reference

        Args:
            parent_state_machine: Main StateMachine instance (GB raceline based)
        """
        self.parent = parent_state_machine

        # ===== HJ MODIFIED: Copy ALL parent attributes, then override specific ones =====
        # Copy all parent's attributes by reference (shallow copy of __dict__)
        # This ensures all inherited methods have access to all needed variables
        self.__dict__.update(parent_state_machine.__dict__)

        # Now override only the Fixed Frenet specific variables
        self.cur_s = 0.0  # Progress along Fixed path (override)
        self.cur_d = 0.0  # Lateral offset from Fixed path (override)
        self.cur_vs = 0.0  # Velocity along Fixed path (override)
        self.cur_vd = 0.0  # Lateral velocity along Fixed path (override)

        # Fixed path has its own obstacles_in_interest (override)
        self.obstacles_in_interest = []  # Will be calculated based on Fixed Frenet s
        self.cur_obstacles_in_interest = []  # Current obstacles in interest

        # ===== HJ ADDED: Fixed path has its own obstacles with Fixed Frenet =====
        self.obstacles = []  # Will be extracted from parent with Fixed Frenet coordinates (override)
        # ===== HJ ADDED END =====

        # Waypoint data - use Smart Static waypoints as "gb_wpnts" (override)
        self.cur_gb_wpnts = parent_state_machine.cur_smart_static_avoidance_wpnts
        self.num_glb_wpnts = 0  # Will be updated in update()

        # ===== HJ ADDED: Fixed Frenet odom subscriber (only this one needed!) =====
        rospy.Subscriber('/car_state/odom_frenet_fixed', Odometry, self._odom_fixed_cb)
        rospy.loginfo("[SmartStaticChecker] Initialized with Fixed Frenet odom subscription")
        # ===== HJ ADDED END =====
        # ===== HJ MODIFIED END =====

        # ===== Background update timer (same Hz as state machine) =====
        # Keeps cur_s, cur_d always up-to-date with zero latency
        self.update_timer = rospy.Timer(
            rospy.Duration(1.0/parent_state_machine.rate_hz),
            self._update_callback
        )

    def _odom_fixed_cb(self, data):
        """Fixed Frenet odom callback

        Updates cur_s, cur_d, cur_vs, cur_vd from Fixed Frenet odom.
        Overrides parent's GB Frenet values.
        """
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x
        self.cur_vd = data.twist.twist.linear.y

    def _update_callback(self, event):
        """Timer callback - automatically updates obstacles in background

        Runs at same rate as state machine (80Hz) to keep Fixed Frenet obstacles current.
        This ensures zero latency when transitions access self.obstacles.
        """
        self.update()

    def update(self):
        """Update Fixed Frenet obstacles from parent

        Called every iteration in timer callback.
        Extracts Fixed Frenet coordinates from parent's obstacles.
        """
        if len(self.parent.cur_smart_static_avoidance_wpnts.list) == 0:
            # No Smart Static path available
            self.num_glb_wpnts = 0
            # ===== HJ MODIFIED: Clear obstacles when no path =====
            self.obstacles_in_interest = []
            self.cur_obstacles_in_interest = []
            self.obstacles = []
            # ===== HJ MODIFIED END =====
            return

        # Update waypoint reference
        self.cur_gb_wpnts = self.parent.cur_smart_static_avoidance_wpnts
        self.num_glb_wpnts = len(self.cur_gb_wpnts.list)

        # ===== HJ MODIFIED: Extract Fixed Frenet coordinates from parent's obstacles =====
        self._extract_fixed_frenet_obstacles()
        # ===== HJ MODIFIED END =====

        # ===== HJ MODIFIED: Calculate obstacles in interest based on Fixed Frenet s =====
        self._update_obstacles_in_interest()
        # ===== HJ MODIFIED END =====

    # ===== HJ MODIFIED: Extract Fixed Frenet coordinates from parent's obstacles =====
    def _extract_fixed_frenet_obstacles(self):
        """Extract Fixed Frenet coordinates from parent's obstacles

        Uses _fixed fields already calculated by perception node.
        Creates copies of obstacles with Fixed Frenet values.
        """
        import copy

        self.obstacles = []
        for obs in self.parent.obstacles:
            # Create a copy to avoid modifying parent's obstacle
            obs_copy = copy.copy(obs)

            # Extract Fixed Frenet coordinates (already calculated by perception)
            obs_copy.s_start = obs.s_start_fixed
            obs_copy.s_end = obs.s_end_fixed
            obs_copy.s_center = obs.s_center_fixed
            obs_copy.d_center = obs.d_center_fixed
            obs_copy.d_right = obs.d_right_fixed
            obs_copy.d_left = obs.d_left_fixed

            # Extract Fixed Frenet velocities
            obs_copy.vs = obs.vs_fixed
            obs_copy.vd = obs.vd_fixed

            # Extract Fixed Frenet variances
            obs_copy.s_var = obs.s_var_fixed
            obs_copy.d_var = obs.d_var_fixed
            obs_copy.vs_var = obs.vs_var_fixed
            obs_copy.vd_var = obs.vd_var_fixed

            self.obstacles.append(obs_copy)
    # ===== HJ MODIFIED END =====

    # ===== HJ MODIFIED: Obstacle filtering based on Fixed Frenet =====
    def _update_obstacles_in_interest(self):
        """Filter obstacles based on Fixed Frenet coordinates

        Only includes obstacles within interest_horizon_m ahead on Fixed path.
        Uses Fixed Frenet coordinates (already extracted from parent's obstacles).
        """
        obstacles_in_interest = []

        # Filter obstacles based on proximity to current Fixed Frenet s position
        # Note: self.obstacles already has Fixed Frenet coordinates
        for obs in self.obstacles:
            gap = (obs.s_start - self.cur_s) % self.track_length
            if gap < self.interest_horizon_m:
                obstacles_in_interest.append(obs)

        self.obstacles_in_interest = obstacles_in_interest
        self.cur_obstacles_in_interest = obstacles_in_interest
    # ===== HJ MODIFIED END =====
