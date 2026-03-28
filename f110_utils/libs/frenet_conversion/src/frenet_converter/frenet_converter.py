from typing import Union
import numpy as np
from scipy.interpolate import CubicSpline

### HJ : 3D frenet converter — added z support for slope-aware s calculation
class FrenetConverter:
    def __init__(self, waypoints_x: np.array, waypoints_y: np.array, waypoints_z: np.array = None):
        self.waypoints_x = waypoints_x
        self.waypoints_y = waypoints_y
        self.waypoints_z = waypoints_z if waypoints_z is not None else np.zeros_like(waypoints_x)
        self.waypoints_s = None
        self.spline_x = None
        self.spline_y = None
        self.spline_z = None
        self.raceline_length = None
        self.waypoints_distance_m = 0.1 # [m]
        self.iter_max = 3
        self.has_z = waypoints_z is not None

        self.build_raceline()

    def build_raceline(self):
        # ### HJ : use 3D distance for s calculation
        self.waypoints_s = [0.0]
        prev_x = self.waypoints_x[0]
        prev_y = self.waypoints_y[0]
        prev_z = self.waypoints_z[0]
        for wx, wy, wz in zip(self.waypoints_x[1:], self.waypoints_y[1:], self.waypoints_z[1:]):
            dist = np.linalg.norm([wx - prev_x, wy - prev_y, wz - prev_z])
            prev_x, prev_y, prev_z = wx, wy, wz
            self.waypoints_s.append(self.waypoints_s[-1] + dist)
        self.spline_x = CubicSpline(self.waypoints_s, self.waypoints_x)
        self.spline_y = CubicSpline(self.waypoints_s, self.waypoints_y)
        self.spline_z = CubicSpline(self.waypoints_s, self.waypoints_z)
        self.raceline_length = self.waypoints_s[-1]
        # ### HJ : end

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
        ### HJ : use 3D distance when z is available
        """
        lenx = len(x)
        dist_x = x - np.tile(self.waypoints_x, (lenx, 1)).T
        dist_y = y - np.tile(self.waypoints_y, (lenx, 1)).T
        dist_2d = np.linalg.norm([dist_x.T, dist_y.T], axis=0)
        return np.argmin(dist_2d, axis=1) * self.waypoints_distance_m

    def get_approx_s_3d(self, x, y, z) -> float:
        """
        ### HJ : 3D nearest waypoint search for overpass/underpass tracks
        """
        lenx = len(x)
        dist_x = x - np.tile(self.waypoints_x, (lenx, 1)).T
        dist_y = y - np.tile(self.waypoints_y, (lenx, 1)).T
        dist_z = z - np.tile(self.waypoints_z, (lenx, 1)).T
        dist_3d = np.linalg.norm([dist_x.T, dist_y.T, dist_z.T], axis=0)
        return np.argmin(dist_3d, axis=1) * self.waypoints_distance_m

    def get_frenet_3d(self, x, y, z, s=None) -> np.array:
        """
        ### HJ : 3D frenet conversion — uses z for nearest search
        """
        if s is None:
            s = self.get_approx_s_3d(x, y, z)
            s, d = self.get_frenet_coord(x, y, s)
        else:
            s, d = self.get_frenet_coord(x, y, s)
        return np.array([s, d])

    def get_frenet_coord(self, x, y, s, eps_m=0.01) -> float:
        """
        Finds the s-coordinate of the given point, considering the perpendicular
        projection of the point on the track.
        """
        _, projection, d = self.check_perpendicular(x, y, s, eps_m)
        for i in range(self.iter_max):
            cand_s = (s + projection)%self.raceline_length
            _, cand_projection, cand_d = self.check_perpendicular(x, y, cand_s, eps_m)
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

        # computes the projection of point_to_track on tangent
        proj = np.einsum('ij,ij->j', tangent, point_to_track)
        perps = np.array([-tangent[1, :], tangent[0, :]])
        d = np.einsum('ij,ij->j', perps, point_to_track)

        check_perpendicular = None

        return check_perpendicular, proj, d

    def get_derivative(self, s) -> np.array:
        """
        Returns the derivative of the point corresponding to s on the chosen line.
        """
        s = s%self.raceline_length
        der = [self.spline_x(s, 1), self.spline_y(s, 1)]
        return der


    def get_cartesian(self, s: float, d: float) -> np.array:
        """
        Convert Frenet coordinates to Cartesian coordinates
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)

        return np.array([x, y])

    ### HJ : 3D cartesian conversion
    def get_cartesian_3d(self, s: float, d: float) -> np.array:
        """
        Convert Frenet coordinates to 3D Cartesian coordinates
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        z = self.spline_z(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)
        return np.array([x, y, z])
    ### HJ : end

    def get_e_psi(self, x: float, y:float, yaw:float) -> float:
        """
        Calculate E_psi: the heading error between vehicle yaw and track direction.
        """
        s = self.get_approx_s(np.array([x]), np.array([y]))[0]
        psi = np.arctan2(*self.get_derivative(s)[::-1])  # dy/ds, dx/ds → arctan2(dy, dx)

        e_psi = yaw - psi
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

        return e_psi
