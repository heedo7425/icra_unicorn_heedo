#!/usr/bin/env python3
"""
Smart Static State Helper

Provides Smart Static raceline-based state checking by inheriting StateMachine.
Uses Smart Static path as reference instead of GB raceline.

Usage:
    checker = SmartStaticChecker(state_machine)
    checker.update()  # Calculate Smart Static s, d
    close = checker._check_close_to_raceline()  # Smart Static based check
"""

import numpy as np
import rospy
from state_machine_node import StateMachine
from frenet_converter.frenet_converter import FrenetConverter  # ===== HJ ADDED =====


class SmartStaticChecker(StateMachine):
    """StateMachine functions but with Smart Static raceline as reference

    Inherits all check functions from StateMachine.
    Overrides instance variables (cur_s, cur_d, etc.) to use Smart Static path.
    Does NOT call super().__init__() - manually assigns needed variables.
    """

    def __init__(self, parent_state_machine):
        """Initialize with Smart Static raceline reference

        Args:
            parent_state_machine: Main StateMachine instance (GB raceline based)
        """
        self.parent = parent_state_machine

        # ===== HJ MODIFIED: Copy ALL parent attributes, then override specific ones =====
        # Copy all parent's attributes by reference (shallow copy of __dict__)
        # This ensures all inherited methods have access to all needed variables
        self.__dict__.update(parent_state_machine.__dict__)

        # Now override only the Smart Static specific variables
        self.cur_s = 0.0  # Progress along Smart Static path (override)
        self.cur_d = 0.0  # Lateral offset from Smart Static path (override)
        self.cur_vs = 0.0  # Velocity along Smart Static path (override)

        # Smart Static has its own obstacles_in_interest (override)
        self.obstacles_in_interest = []  # Will be calculated based on Smart Static s
        self.cur_obstacles_in_interest = []  # Current obstacles in interest

        # ===== HJ ADDED: Smart Static has its own obstacles with Smart Static Frenet =====
        self.obstacles = []  # Will be calculated with Smart Static Frenet coordinates (override)
        # ===== HJ ADDED END =====

        # Waypoint data - use Smart Static waypoints as "gb_wpnts" (override)
        self.cur_gb_wpnts = parent_state_machine.cur_smart_static_avoidance_wpnts
        self.num_glb_wpnts = 0  # Will be updated in update()

        # ===== HJ ADDED: FrenetConverter for Smart Static path =====
        self.frenet_converter = None  # Will be created when Smart Static waypoints available
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: For velocity calculation =====
        self.prev_s = 0.0
        self.prev_time = rospy.Time.now()
        # ===== HJ ADDED END =====
        # ===== HJ MODIFIED END =====

        # ===== Background update timer (same Hz as state machine) =====
        # Keeps cur_s, cur_d always up-to-date with zero latency
        self.update_timer = rospy.Timer(
            rospy.Duration(1.0/parent_state_machine.rate_hz),
            self._update_callback
        )

    def _update_callback(self, event):
        """Timer callback - automatically updates cur_s, cur_d in background

        Runs at same rate as state machine (80Hz) to keep Smart Static Frenet coordinates current.
        This ensures zero latency when transitions access cur_s, cur_d.
        """
        self.update()

    def update(self):
        """Update Smart Static based Frenet coordinates (s, d)

        Called every iteration in main loop.
        Calculates cur_s and cur_d based on Smart Static path.
        """
        if len(self.parent.cur_smart_static_avoidance_wpnts.list) == 0:
            # No Smart Static path available
            self.cur_s = 0.0
            self.cur_d = 0.0
            self.num_glb_wpnts = 0
            self.frenet_converter = None
            # ===== HJ MODIFIED: Clear obstacles when no path =====
            self.obstacles_in_interest = []
            self.cur_obstacles_in_interest = []
            # ===== HJ MODIFIED END =====
            return

        # Update waypoint reference
        self.cur_gb_wpnts = self.parent.cur_smart_static_avoidance_wpnts
        self.num_glb_wpnts = len(self.cur_gb_wpnts.list)

        # ===== HJ ADDED: Build FrenetConverter for Smart Static path =====
        # Create FrenetConverter using Smart Static waypoints
        wpnts_x = np.array([wpnt.x_m for wpnt in self.cur_gb_wpnts.list])
        wpnts_y = np.array([wpnt.y_m for wpnt in self.cur_gb_wpnts.list])
        self.frenet_converter = FrenetConverter(wpnts_x, wpnts_y)

        # Recalculate s_m for each waypoint using Smart Static path as reference
        self._recalculate_waypoint_frenet()

        # Convert all obstacles to Smart Static Frenet coordinates
        self._convert_obstacles_to_smart_static_frenet()
        # ===== HJ ADDED END =====

        # Calculate s and d from Smart Static path
        self._calculate_frenet_from_smart_static()

        # ===== HJ MODIFIED: Calculate obstacles in interest based on Smart Static s =====
        self._update_obstacles_in_interest()
        # ===== HJ MODIFIED END =====

    # ===== HJ ADDED: Convert obstacles to Smart Static Frenet =====
    def _convert_obstacles_to_smart_static_frenet(self):
        """Convert all obstacles from GB Frenet to Smart Static Frenet

        Creates copies of obstacles with Smart Static based Frenet coordinates.
        Modifies s_start, s_center, s_end, d_center, vs, vd in place.
        """
        if self.frenet_converter is None:
            self.obstacles = []
            return

        # Get parent's GB-based obstacles
        gb_obstacles = self.parent.obstacles

        # Convert each obstacle to Smart Static Frenet
        import copy
        self.obstacles = []
        for obs in gb_obstacles:
            # Create a copy to avoid modifying parent's obstacle
            obs_copy = copy.copy(obs)

            # Convert obstacle position to Smart Static Frenet
            obs_x = np.array([obs.x_m])
            obs_y = np.array([obs.y_m])
            obs_frenet = self.frenet_converter.get_frenet(obs_x, obs_y)

            # Update Frenet coordinates with Smart Static values
            obs_s_smart = obs_frenet[0][0]
            obs_d_smart = obs_frenet[1][0]

            # Update all s and d values
            obs_copy.s_center = obs_s_smart
            obs_copy.d_center = obs_d_smart
            # Assuming obstacle has width/length, update s_start and s_end
            # (approximation: keep same delta from center)
            if hasattr(obs, 's_start') and hasattr(obs, 's_end'):
                s_delta_start = obs.s_center - obs.s_start
                s_delta_end = obs.s_end - obs.s_center
                obs_copy.s_start = obs_s_smart - s_delta_start
                obs_copy.s_end = obs_s_smart + s_delta_end

            # ===== HJ ADDED: Convert velocity to Smart Static Frenet =====
            # Get obstacle's heading at current position (from FrenetConverter)
            obs_psi = self.frenet_converter.get_heading(obs_x, obs_y)[0]

            # Convert GB Frenet velocity (vs, vd) to Cartesian velocity (vx, vy)
            # First need GB path heading at obstacle position
            # Use parent's frenet_converter or cur_gb_wpnts to get GB heading
            gb_wpnts_x = np.array([wpnt.x_m for wpnt in self.parent.cur_gb_wpnts.list])
            gb_wpnts_y = np.array([wpnt.y_m for wpnt in self.parent.cur_gb_wpnts.list])
            gb_frenet_converter = FrenetConverter(gb_wpnts_x, gb_wpnts_y)
            gb_psi = gb_frenet_converter.get_heading(obs_x, obs_y)[0]

            # Convert GB Frenet (vs, vd) -> Cartesian (vx, vy)
            # vs is along path tangent, vd is perpendicular (left positive)
            vx_cart = obs.vs * np.cos(gb_psi) - obs.vd * np.sin(gb_psi)
            vy_cart = obs.vs * np.sin(gb_psi) + obs.vd * np.cos(gb_psi)

            # Convert Cartesian (vx, vy) -> Smart Static Frenet (vs, vd)
            # vs is projection along Smart Static tangent, vd is projection along normal
            vs_smart = vx_cart * np.cos(obs_psi) + vy_cart * np.sin(obs_psi)
            vd_smart = -vx_cart * np.sin(obs_psi) + vy_cart * np.cos(obs_psi)

            obs_copy.vs = vs_smart
            obs_copy.vd = vd_smart
            # ===== HJ ADDED END =====

            self.obstacles.append(obs_copy)
    # ===== HJ ADDED END =====

    # ===== HJ ADDED: Recalculate waypoint Frenet coordinates =====
    def _recalculate_waypoint_frenet(self):
        """Recalculate s_m for each waypoint based on Smart Static path

        Uses FrenetConverter to get proper s values along Smart Static path.
        Modifies waypoint s_m in place.
        """
        if self.frenet_converter is None or self.num_glb_wpnts == 0:
            return

        # Get all waypoint positions
        wpnts_x = np.array([wpnt.x_m for wpnt in self.cur_gb_wpnts.list])
        wpnts_y = np.array([wpnt.y_m for wpnt in self.cur_gb_wpnts.list])

        # Convert to Frenet coordinates based on Smart Static path
        frenet_coords = self.frenet_converter.get_frenet(wpnts_x, wpnts_y)

        # Update each waypoint's s_m with Smart Static based s
        for i, wpnt in enumerate(self.cur_gb_wpnts.list):
            wpnt.s_m = frenet_coords[0][i]  # s coordinate
            # Note: d_m stays as original (lateral offset from Smart Static path)
    # ===== HJ ADDED END =====

    def _calculate_frenet_from_smart_static(self):
        """Calculate Frenet coordinates (s, d) and velocity relative to Smart Static path

        Uses FrenetConverter for accurate Frenet transformation.
        Calculates cur_vs using time derivative of cur_s.
        """
        # ===== HJ ADDED: Safety check for current_position =====
        if self.current_position is None or self.num_glb_wpnts == 0 or self.frenet_converter is None:
            self.cur_s = 0.0
            self.cur_d = 0.0
            self.cur_vs = 0.0
            return
        # ===== HJ ADDED END =====

        # ===== HJ MODIFIED: Use FrenetConverter =====
        # Get ego position
        ego_x = np.array([self.current_position[0]])
        ego_y = np.array([self.current_position[1]])

        # Convert to Frenet using FrenetConverter
        frenet_coords = self.frenet_converter.get_frenet(ego_x, ego_y)

        new_s = frenet_coords[0][0]  # s coordinate
        self.cur_d = frenet_coords[1][0]  # d coordinate

        # ===== HJ ADDED: Calculate velocity using time derivative =====
        current_time = rospy.Time.now()
        dt = (current_time - self.prev_time).to_sec()

        if dt > 0.0:
            # Handle track wrapping
            ds = new_s - self.prev_s
            # Check for wrap-around (crossing start/finish line)
            if ds > self.track_length / 2:
                ds -= self.track_length
            elif ds < -self.track_length / 2:
                ds += self.track_length

            self.cur_vs = ds / dt
        else:
            self.cur_vs = 0.0

        # Update previous values
        self.prev_s = new_s
        self.prev_time = current_time
        self.cur_s = new_s
        # ===== HJ ADDED END =====
        # ===== HJ MODIFIED END =====

    # ===== HJ ADDED: Obstacle filtering based on Smart Static Frenet =====
    def _update_obstacles_in_interest(self):
        """Filter obstacles based on Smart Static Frenet coordinates

        Only includes obstacles within interest_horizon_m ahead on Smart Static path.
        Uses Smart Static based Frenet coordinates (already converted in _convert_obstacles_to_smart_static_frenet).
        """
        obstacles_in_interest = []

        # Filter obstacles based on proximity to current Smart Static s position
        # Note: self.obstacles already has Smart Static Frenet coordinates
        for obs in self.obstacles:
            gap = (obs.s_start - self.cur_s) % self.track_length
            if gap < self.interest_horizon_m:
                obstacles_in_interest.append(obs)

        self.obstacles_in_interest = obstacles_in_interest
        self.cur_obstacles_in_interest = obstacles_in_interest
    # ===== HJ ADDED END =====
