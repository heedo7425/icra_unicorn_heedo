#! /usr/bin/env python3

import rospy
import rospkg
import numpy as np
import yaml
import matplotlib.pyplot as plt
from f110_msgs.msg import WpntArray, OTWpntArray
from std_msgs.msg import Bool
from dynamic_reconfigure.msg import Config


class VelocityScaler:
    """
    Sector scaler for the velocity of the global waypoints
    """

    def __init__(self, debug_plot: bool = False) -> None:
        self.debug_plot = rospy.get_param("/velocity_scaler/debug_plot", False)
        self.enable_smart_scaling = rospy.get_param("~enable_smart_scaling", False)
        rospy.logwarn(f"[VelocityScaler] enable_smart_scaling parameter: {self.enable_smart_scaling}")

        # sectors params
        self.glb_wpnts_og = None
        self.glb_wpnts_scaled = None
        self.glb_wpnts_sp_og = None
        self.glb_wpnts_sp_scaled = None
        self.update_map = False

        # smart waypoints
        self.smart_wpnts_og = None
        self.smart_wpnts_scaled = None
        self.smart_wpnts_received = False

        # get initial scaling
        pkg_path = rospkg.RosPack().get_path("stack_master")
        map_name = rospy.get_param('/map')
        yaml_file_path = pkg_path + "/maps/" + map_name + "/speed_scaling.yaml" 
        with open(yaml_file_path, "r") as file:
            self.sectors_params = yaml.safe_load(file)
        
        self.n_sectors = self.sectors_params['n_sectors']

        # dyn params sub
        self.glb_wpnts_name = "/global_waypoints"
        rospy.Subscriber("/dyn_sector_speed/parameter_updates", Config, self.dyn_param_cb)
        rospy.Subscriber(self.glb_wpnts_name, WpntArray, self.glb_wpnts_cb)
        rospy.Subscriber(self.glb_wpnts_name+"/shortest_path", WpntArray, self.glb_wpnts_sp_cb)
        rospy.Subscriber("update_map", Bool, self.update_map_cb)

        # new glb_waypoints pub
        self.scaled_points_pub = rospy.Publisher("/global_waypoints_scaled", WpntArray, queue_size=10)
        self.scaled_points_sp_pub = rospy.Publisher("/global_waypoints_scaled/shortest_path", WpntArray, queue_size=10)

        # smart waypoints sub and pub (only if enabled)
        if self.enable_smart_scaling:
            self.smart_sub = rospy.Subscriber("/planner/avoidance/smart_static_otwpnts", OTWpntArray, self.smart_wpnts_cb)
            self.smart_scaled_pub = rospy.Publisher("/planner/avoidance/smart_static_otwpnts_scaled", OTWpntArray, queue_size=10)
            rospy.loginfo("Smart waypoints scaling ENABLED")
        else:
            self.smart_sub = None
            self.smart_scaled_pub = None
            rospy.loginfo("Smart waypoints scaling DISABLED")

    def update_map_cb(self, data:Bool):
        self.update_map = True
        
    def glb_wpnts_cb(self, data:WpntArray):
        """
        Saves the global waypoints
        """
        self.glb_wpnts_og = data

    def glb_wpnts_sp_cb(self, data:WpntArray):
        """
        Saves the global waypoints
        """
        self.glb_wpnts_sp_og = data

    def smart_wpnts_cb(self, data:OTWpntArray):
        """
        Saves smart waypoints whenever received (keeps subscription active for updates).
        Also resets scaled waypoints to trigger re-initialization.
        """
        import copy
        self.smart_wpnts_og = data
        # Immediately create fresh deepcopy for scaled (thread-safe reset)
        self.smart_wpnts_scaled = copy.deepcopy(data)
        if not self.smart_wpnts_received:
            self.smart_wpnts_received = True
            rospy.loginfo(f"Smart waypoints received! {len(data.wpnts)} waypoints, first vel: {data.wpnts[0].vx_mps:.2f} m/s")
        else:
            rospy.loginfo_throttle(5.0, f"Smart waypoints updated! {len(data.wpnts)} waypoints")


    def dyn_param_cb(self, params:Config):
        """
        Notices the change in the parameters and scales the global waypoints
        """
        # get global limit
        self.sectors_params['global_limit'] = params.doubles[0].value

        # update params 
        for i in range(self.n_sectors):
            self.sectors_params[f"Sector{i}"]['scaling'] = np.clip(
                params.doubles[i+1].value, 0, self.sectors_params['global_limit']
            )

        rospy.loginfo(self.sectors_params)

    def xy_to_gb_index(self, x, y):
        """
        Finds the closest global waypoint index for given x, y coordinates.
        Used to determine which sector a point belongs to (sectors use index-based start/end).

        Parameters
        ----------
        x : float
            X coordinate in global frame
        y : float
            Y coordinate in global frame

        Returns
        -------
        idx : int
            Index of the closest global waypoint
        """
        if self.glb_wpnts_og is None:
            return 0

        min_dist = float('inf')
        closest_idx = 0

        for i, wpnt in enumerate(self.glb_wpnts_og.wpnts):
            dist = (wpnt.x_m - x)**2 + (wpnt.y_m - y)**2
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def get_vel_scaling(self, s):
        """
        Gets the dynamically reconfigured velocity scaling for the points.
        Linearly interpolates for points between two sectors

        Parameters
        ----------
        s
            s parameter whose sector we want to find
        """
        hl_change = 10

        if self.n_sectors > 1:
            for i in range(self.n_sectors):
                if i == 0 :
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start']-hl_change, self.sectors_params[f'Sector{i}']['start']+hl_change],
                            fp=[self.sectors_params[f'Sector{self.n_sectors-1}']['scaling'], self.sectors_params[f'Sector{i}']['scaling']]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start'] - hl_change):
                        scaler = self.sectors_params[f"Sector{i}"]['scaling']
                    elif (s >= self.sectors_params[f'Sector{i+1}']['start'] - hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start']):
                        scaler = np.interp(
                        x=s,
                        xp=[self.sectors_params[f'Sector{i+1}']['start']-hl_change, self.sectors_params[f'Sector{i+1}']['start']+hl_change],
                        fp=[self.sectors_params[f'Sector{i}']['scaling'], self.sectors_params[f'Sector{i+1}']['scaling']]
                    )
                elif i != self.n_sectors-1:
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start']-hl_change, self.sectors_params[f'Sector{i}']['start']+hl_change],
                            fp=[self.sectors_params[f'Sector{i-1}']['scaling'], self.sectors_params[f'Sector{i}']['scaling']]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start'] - hl_change):
                        scaler = self.sectors_params[f"Sector{i}"]['scaling']
                    elif (s >= self.sectors_params[f'Sector{i+1}']['start'] - hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start']):
                        scaler = np.interp(
                        x=s,
                        xp=[self.sectors_params[f'Sector{i+1}']['start']-hl_change, self.sectors_params[f'Sector{i+1}']['start']+hl_change],
                        fp=[self.sectors_params[f'Sector{i}']['scaling'], self.sectors_params[f'Sector{i+1}']['scaling']]
                    )
                else:
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start']-hl_change, self.sectors_params[f'Sector{i}']['start']+hl_change],
                            fp=[self.sectors_params[f'Sector{i-1}']['scaling'], self.sectors_params[f'Sector{i}']['scaling']]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i}']['end'] - hl_change):
                        scaler = self.sectors_params[f"Sector{i}"]['scaling']
                    elif (s >= self.sectors_params[f'Sector{i}']['end'] - hl_change):
                        scaler = np.interp(
                        x=s,
                        xp=[self.sectors_params[f'Sector{i}']['end']-hl_change, self.sectors_params[f'Sector{i}']['end']+hl_change],
                        fp=[self.sectors_params[f'Sector{i}']['scaling'], self.sectors_params[f'Sector{0}']['scaling']]
                    )
        elif self.n_sectors == 1:
            scaler = self.sectors_params["Sector0"]['scaling']

        return scaler

    def scale_points(self):
        """
        Scales the global waypoints' velocities
        """
        scaling = []

        if self.glb_wpnts_scaled is None or self.update_map:
            self.glb_wpnts_scaled = self.glb_wpnts_og
            self.glb_wpnts_sp_scaled = self.glb_wpnts_sp_og
            self.update_map = False

        for i, wpnt  in enumerate(self.glb_wpnts_og.wpnts):
            vel_scaling = self.get_vel_scaling(i)
            new_vel = wpnt.vx_mps*vel_scaling
            self.glb_wpnts_scaled.wpnts[i].vx_mps = new_vel
            scaling.append(self.get_vel_scaling(i))

        if self.debug_plot:
            plt.clf()
            plt.plot(scaling)
            plt.legend(['og', 'scaled'])
            plt.ylim(0,1)
            plt.show()

    def scale_smart_waypoints(self):
        """
        Scales the smart waypoints' velocities based on global waypoints' sectors.
        Converts each waypoint's x, y to closest global waypoint index,
        finds the appropriate scaling factor, and applies it to the velocity.
        Preserves all other waypoint attributes (s, d, x, y, order, etc.)
        Updates header timestamp to current time.
        """
        if self.smart_wpnts_og is None:
            return

        # Initialize scaled waypoints on first call
        if self.smart_wpnts_scaled is None:
            import copy
            self.smart_wpnts_scaled = copy.deepcopy(self.smart_wpnts_og)

        # Update header timestamp to current time (ensures this is newer than velocity_planner's message)
        self.smart_wpnts_scaled.header.stamp = rospy.Time.now()

        # Scale each waypoint's velocity
        for i, wpnt in enumerate(self.smart_wpnts_og.wpnts):
            # Convert x, y to closest global waypoint index (sector start/end are index-based)
            gb_idx = self.xy_to_gb_index(wpnt.x_m, wpnt.y_m)

            # Get velocity scaling factor based on global waypoints sector
            vel_scaling = self.get_vel_scaling(gb_idx)

            # Apply scaling to velocity only
            new_vel = wpnt.vx_mps * vel_scaling
            self.smart_wpnts_scaled.wpnts[i].vx_mps = new_vel

        rospy.loginfo_throttle(2.0, f"Smart waypoints scaled! First wpnt vel: {self.smart_wpnts_scaled.wpnts[0].vx_mps:.2f} m/s")


    def loop(self):
        rospy.loginfo("Waiting for global waypoints...")
        rospy.wait_for_message(self.glb_wpnts_name, WpntArray)
        rospy.loginfo("Global waypoints received!")

        # initialise scaled points
        self.scale_points()

        run_rate = rospy.Rate(4)
        while not rospy.is_shutdown():
            # Scale global waypoints
            self.scale_points()
            self.scaled_points_pub.publish(self.glb_wpnts_scaled)

            # Scale and publish smart waypoints if enabled and received
            if self.enable_smart_scaling and self.smart_wpnts_received and self.smart_wpnts_og is not None:
                self.scale_smart_waypoints()
                if self.smart_wpnts_scaled is not None:
                    self.smart_scaled_pub.publish(self.smart_wpnts_scaled)

            run_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("vel_scaler")
    vel_scaler = VelocityScaler()
    vel_scaler.loop()
