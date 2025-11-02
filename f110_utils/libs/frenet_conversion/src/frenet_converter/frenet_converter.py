from typing import Union
import numpy as np
from scipy.interpolate import CubicSpline
# ===== HJ ADDED: For occupancy map support =====
import rospy
from nav_msgs.msg import OccupancyGrid
# ===== HJ ADDED END =====

class FrenetConverter:
    def __init__(self, waypoints_x: np.array, waypoints_y: np.array):
        self.waypoints_x = waypoints_x
        self.waypoints_y = waypoints_y
        self.waypoints_s = None
        self.spline_x = None
        self.spline_y = None
        self.raceline_length = None
        self.waypoints_distance_m = 0.1 # [m]
        self.iter_max = 3

        # ===== HJ ADDED: Occupancy map for wall filtering =====
        self.occupancy_grid = None
        self.has_occupancy_map = False
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_width = None
        self.map_height = None
        self.prev_valid_idx = 0
        # ===== HJ ADDED END =====

        self.build_raceline()

    def build_raceline(self):
        self.waypoints_s = [0.0]
        prev_wpnt_x =  self.waypoints_x[0]
        prev_wpnt_y =  self.waypoints_y[0]
        for wpnt_x, wpnt_y in zip(self.waypoints_x[1:], self.waypoints_y[1:]):
            dist = np.linalg.norm([wpnt_x - prev_wpnt_x, wpnt_y - prev_wpnt_y])
            prev_wpnt_x = wpnt_x
            prev_wpnt_y = wpnt_y
            self.waypoints_s.append(self.waypoints_s[-1] + dist)        
        self.spline_x = CubicSpline(self.waypoints_s, self.waypoints_x)
        self.spline_y = CubicSpline(self.waypoints_s, self.waypoints_y)
        self.raceline_length = self.waypoints_s[-1]

    def get_frenet(self, x, y, s=None) -> np.array:
        # Compute Frenet coordinates for a given (x, y) point
        if s is None:
            s = self.get_approx_s(x, y)
            s, d = self.get_frenet_coord(x, y, s)
        else:
            s, d = self.get_frenet_coord(x, y, s)

        return np.array([s, d])

    def get_approx_s(self, x, y) -> float:
        """
        Finds the s-coordinate of the given point by finding the nearest waypoint.
        Uses wall-crossing detection and rotational search when occupancy map is available.
        """
        # get distance with broadcasting multiple arrays
        lenx = len(x)
        dist_x = x - np.tile(self.waypoints_x, (lenx, 1)).T
        dist_y = y - np.tile(self.waypoints_y, (lenx, 1)).T
        nearest_idx = np.argmin(np.linalg.norm([dist_x.T, dist_y.T], axis=0), axis=1)

        # ===== HJ ADDED: Wall-crossing check with rotational search =====
        if not self.has_occupancy_map:
            # Return array to maintain consistent return type
            return nearest_idx * self.waypoints_distance_m

        # Process each point
        result_s = np.zeros(lenx)
        for i in range(lenx):
            idx = nearest_idx[i]
            px, py = x[i], y[i]
            wpx, wpy = self.waypoints_x[idx], self.waypoints_y[idx]

            # Check if nearest waypoint is wall-free
            if not self.is_line_crossing_obstacle(px, py, wpx, wpy):
                self.prev_valid_idx = idx
                result_s[i] = idx * self.waypoints_distance_m
                continue

            # Wall detected - try rotational search
            vec_x = wpx - px
            vec_y = wpy - py

            # Try 90°, 180°, 270° rotations
            found = False
            for angle_deg in [90, 180, 270]:
                angle_rad = np.deg2rad(angle_deg)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)

                # Rotate vector
                rotated_x = vec_x * cos_a - vec_y * sin_a
                rotated_y = vec_x * sin_a + vec_y * cos_a

                # Target point
                target_x = px + rotated_x
                target_y = py + rotated_y

                # Find nearest waypoint to target
                candidate_idx = self.find_nearest_waypoint_to_point(target_x, target_y)
                cand_wpx = self.waypoints_x[candidate_idx]
                cand_wpy = self.waypoints_y[candidate_idx]

                if not self.is_line_crossing_obstacle(px, py, cand_wpx, cand_wpy):
                    # Search in s-direction for better waypoint
                    best_idx = candidate_idx
                    best_dist = (px - cand_wpx)**2 + (py - cand_wpy)**2

                    # Search ALL waypoints in both s directions until wall hit
                    # Forward direction
                    for s_offset in range(1, len(self.waypoints_x)):
                        test_idx = (candidate_idx + s_offset) % len(self.waypoints_x)

                        # Check if wrapped around to starting point
                        if test_idx == candidate_idx:
                            break

                        test_wpx = self.waypoints_x[test_idx]
                        test_wpy = self.waypoints_y[test_idx]

                        if not self.is_line_crossing_obstacle(px, py, test_wpx, test_wpy):
                            dist = (px - test_wpx)**2 + (py - test_wpy)**2
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = test_idx
                        else:
                            break  # Hit wall, stop searching forward

                    # Backward direction
                    for s_offset in range(1, len(self.waypoints_x)):
                        test_idx = (candidate_idx - s_offset) % len(self.waypoints_x)

                        # Check if wrapped around to starting point
                        if test_idx == candidate_idx:
                            break

                        test_wpx = self.waypoints_x[test_idx]
                        test_wpy = self.waypoints_y[test_idx]

                        if not self.is_line_crossing_obstacle(px, py, test_wpx, test_wpy):
                            dist = (px - test_wpx)**2 + (py - test_wpy)**2
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = test_idx
                        else:
                            break  # Hit wall, stop searching backward

                    self.prev_valid_idx = best_idx
                    result_s[i] = best_idx * self.waypoints_distance_m
                    found = True
                    break

            if not found:
                # All directions blocked - use nearest distance (ignoring walls)
                self.prev_valid_idx = idx
                result_s[i] = idx * self.waypoints_distance_m

        return result_s
        # ===== HJ ADDED END =====

    def get_frenet_coord(self, x, y, s, eps_m=0.01) -> float:
        """
        Finds the s-coordinate of the given point, considering the perpendicular
        projection of the point on the track.
        
        Args:
            x (float): x-coordinate of the point
            y (float): y-coordinate of the point
            s (float): estimated s-coordinate of the point
            eps_m (float): maximum error tolerance for the projection. Default is 0.01.
        
        Returns:
            The s-coordinate of the point on the track.
        """
        # Check if point is on the estimated s perpendicular to the track

        _, projection, d = self.check_perpendicular(x, y, s, eps_m)
        for i in range(self.iter_max):
            cand_s = (s + projection)%self.raceline_length
            _, cand_projection, cand_d = self.check_perpendicular(x, y, cand_s, eps_m)
            #print(f"candidate projection: {cand_projection}; projection: {projection}; d: {d} cand_d: {cand_d}")
            cand_projection = np.clip(cand_projection, -self.waypoints_distance_m/(2*self.iter_max), self.waypoints_distance_m/(2*self.iter_max))
            updated_idxs = np.abs(cand_projection) <= np.abs(projection)
            d[updated_idxs] = cand_d[updated_idxs]
            s[updated_idxs] = cand_s[updated_idxs]
            projection[updated_idxs] = cand_projection[updated_idxs]

        return s, d

    def check_perpendicular(self, x, y, s, eps_m=0.01) -> Union[bool, float]:
        # obtain unit vector parallel to the track
        dx_ds, dy_ds = self.get_derivative(s)
        tangent = np.array([dx_ds, dy_ds])
        if np.any(np.isnan(s)):
            raise ValueError("BUB FRENET CONVERTER: S is nan")
        tangent /= np.linalg.norm(tangent, axis=0)

        # obtain vector from the track to the point
        x_vec = x - self.spline_x(s)
        y_vec = y - self.spline_y(s)
        point_to_track = np.array([x_vec, y_vec])
        
        # check if the vectors are perpendicular
        # computes the projection of point_to_track on tangent
        proj = np.einsum('ij,ij->j', tangent, point_to_track)
        perps = np.array([-tangent[1, :], tangent[0, :]])
        d = np.einsum('ij,ij->j', perps, point_to_track)

        # TODO commented out because of computational efficiency
        # eps_m * point_to_track_norm is needed to make it scale invariant 
        # check_perpendicular becomes effectively cos(angle) <= eps_m

        # point_to_track_norm = np.linalg.norm(point_to_track, axis=0)
        # check_perpendicular = np.abs(proj) <= eps_m * point_to_track_norm
        check_perpendicular = None

        return check_perpendicular, proj, d
    
    def get_derivative(self, s) -> np.array:
        """
        Returns the derivative of the point corresponding to s on the chosen line. 
        
        Args: 
            s: parameter which is used to evaluate the spline
            line: argument used to choose the line. Can be 'int', 'mid', 'out'. Default is 'mid'.

        Returns:
            der: dx/ds, dy/ds
        """
        s = s%self.raceline_length

        der = [self.spline_x(s, 1), self.spline_y(s, 1)]
        
        return der
    

    def get_cartesian(self, s: float, d: float) -> np.array:
        """
        Convert Frenet coordinates to Cartesian coordinates
        
        Args:
            s (float): longitudinal coordinate
            d (float): lateral coordinate
            
        Returns:
            np.array: [x, y] Cartesian coordinates
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)
        
        return np.array([x, y])
    

    def get_e_psi(self, x: float, y:float, yaw:float) -> float:
        """
        Calculate E_psi: the heading error between vehicle yaw and track direction.

        Args:
            x (float): vehicle x position
            y (float): vehicle y position
            yaw (float): vehicle yaw angle (radians)

        Returns:
            float: heading error E_psi (in radians, between -pi and pi)
        """
        s = self.get_approx_s(np.array([x]), np.array([y]))[0]
        psi = np.arctan2(*self.get_derivative(s)[::-1])  # dy/ds, dx/ds → arctan2(dy, dx)

        e_psi = yaw - psi
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

        return e_psi

    # ===== HJ ADDED: Occupancy map methods =====
    def set_occupancy_map(self, map_msg: OccupancyGrid):
        """
        Set occupancy map for wall-crossing detection from ROS OccupancyGrid message.

        Args:
            map_msg: OccupancyGrid message from map_server
        """
        if map_msg is None:
            rospy.logerr("[FrenetConverter] Received null map message")
            return

        # Extract map metadata
        self.map_resolution = map_msg.info.resolution
        self.map_origin_x = map_msg.info.origin.position.x
        self.map_origin_y = map_msg.info.origin.position.y
        self.map_width = map_msg.info.width
        self.map_height = map_msg.info.height

        # Convert OccupancyGrid to numpy array
        # OccupancyGrid: -1 (unknown), 0 (free), 100 (occupied)
        # Convert to: 255 (free), 0 (occupied)
        self.occupancy_grid = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

        for y in range(self.map_height):
            for x in range(self.map_width):
                index = x + (self.map_height - 1 - y) * self.map_width  # Flip Y-axis
                cell_value = map_msg.data[index]

                if cell_value == -1 or cell_value > 0:
                    # Unknown or occupied - treat as occupied for safety
                    self.occupancy_grid[y, x] = 0
                else:
                    # Free space
                    self.occupancy_grid[y, x] = 255

        self.has_occupancy_map = True
        rospy.logdebug(f"[FrenetConverter] Occupancy map loaded from ROS: {self.map_width}x{self.map_height} pixels, resolution={self.map_resolution:.3f} m/pixel")
        rospy.logdebug(f"[FrenetConverter] Map origin: ({self.map_origin_x:.2f}, {self.map_origin_y:.2f})")

    def is_line_crossing_obstacle(self, x1, y1, x2, y2):
        """
        Check if line from (x1,y1) to (x2,y2) crosses an obstacle using Bresenham's algorithm.

        Args:
            x1, y1: Start point in world coordinates
            x2, y2: End point in world coordinates

        Returns:
            bool: True if line crosses obstacle, False otherwise
        """
        if not self.has_occupancy_map:
            return False

        # Convert world coordinates to grid coordinates
        px1 = int((x1 - self.map_origin_x) / self.map_resolution)
        py1 = self.map_height - 1 - int((y1 - self.map_origin_y) / self.map_resolution)
        px2 = int((x2 - self.map_origin_x) / self.map_resolution)
        py2 = self.map_height - 1 - int((y2 - self.map_origin_y) / self.map_resolution)

        # Bresenham's line algorithm with max iteration safety
        dx = abs(px2 - px1)
        dy = abs(py2 - py1)
        sx = 1 if px1 < px2 else -1
        sy = 1 if py1 < py2 else -1
        err = dx - dy

        px, py = px1, py1
        max_iterations = max(self.map_width, self.map_height) * 2  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Boundary check
            if px < 0 or px >= self.map_width or py < 0 or py >= self.map_height:
                return True  # Out of bounds = wall

            # Check pixel value (0=occupied, 255=free)
            if self.occupancy_grid[py, px] < 128:  # Threshold for occupied
                return True  # Wall detected

            # Reached end point
            if px == px2 and py == py2:
                break

            # Bresenham step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                px += sx
            if e2 < dx:
                err += dx
                py += sy

        return False  # No wall crossed

    def find_nearest_waypoint_to_point(self, target_x, target_y):
        """
        Find nearest waypoint to a given point.

        Args:
            target_x, target_y: Target point coordinates

        Returns:
            int: Index of nearest waypoint
        """
        dist_squared = (self.waypoints_x - target_x)**2 + (self.waypoints_y - target_y)**2
        return np.argmin(dist_squared)
    # ===== HJ ADDED END =====
