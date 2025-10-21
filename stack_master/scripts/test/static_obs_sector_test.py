#!/usr/bin/env python3
import rospy
import rospkg
import yaml
from dynamic_reconfigure.msg import Config
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

class StaticSectorChecker:
    def __init__(self):
        """
        Initializes the node, loads initial sector data from the parameter server,
        and subscribes to dynamic updates and the car's position.
        """
        rospy.init_node('static_sector_checker', anonymous=True)

        # Member variables to hold state
        self.static_sectors_params = {}
        self.n_static_sectors = 0
        self.active_static_zones = [] # List to store zones where static_obs_section is True
        self.cur_s = 0.0 # Current longitudinal position (s-coordinate)

        # 1. Load initial parameters from the ROS Parameter Server
        #    This is the same method used by the state_machine.
        #    NOTE: Your launch file must load 'static_obs_sectors.yaml' into this parameter.
        try:
            self.static_sectors_params = rospy.get_param("/static_obs_map_params")
            self.n_static_sectors = self.static_sectors_params.get("n_sectors", 0)
            rospy.loginfo(f"Successfully loaded initial data for {self.n_static_sectors} sectors.")
        except KeyError:
            rospy.logerr("Parameter '/static_obs_map_params' not found. Please load it in your launch file.")
            rospy.signal_shutdown("Missing required parameter.")
            return

        # 2. Set up the publisher for the check result
        self.check_pub = rospy.Publisher("/static_obs_section_check", Bool, queue_size=1)

        # 3. Subscribe to the dynamic reconfigure topic and the car's Frenet pose
        #    The topic name comes from the node name defined in your .cfg file's generate() call.
        rospy.Subscriber("/dyn_sector_static_obstacle/parameter_updates", Config, self.static_sector_param_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.frenet_pose_cb)

        # Initialize the active zones based on the initial file
        self._update_active_zones_from_params()

        rospy.loginfo("Static Sector Checker is running.")
        rospy.loginfo("Continuously checking car position against active static obstacle sectors.")

    def static_sector_param_cb(self, params: Config):
        """
        Callback for dynamic reconfigure updates. This is the core of the real-time update logic.
        It updates the in-memory dictionary and rebuilds the list of active zones.
        """
        # params.bools is a list of all boolean parameters from the GUI.
        # Assuming the first boolean is 'save_params', the sectors start from index 1.
        # If you change the .cfg file, this indexing might need to change.
        try:
            for i in range(self.n_static_sectors):
                # Update the 'static_obs_section' for each sector in our stored dictionary
                sector_key = f"Static_Obs_sector{i}"
                # The first boolean param (index 0) is 'save_params', so sectors start at index 1
                self.static_sectors_params[sector_key]["static_obs_section"] = params.bools[i+1].value
            
            # After updating all flags, rebuild the list of active zones
            self._update_active_zones_from_params()
            rospy.loginfo_once("Received first dynamic parameter update.")

        except IndexError:
            rospy.logwarn_throttle(5, "IndexError in dynamic parameter callback. Check your .cfg file and parameter order.")

    def _update_active_zones_from_params(self):
        """
        Helper function to clear and repopulate the list of active zones
        based on the current state of self.static_sectors_params.
        """
        self.active_static_zones = []
        for i in range(self.n_static_sectors):
            sector_key = f"Static_Obs_sector{i}"
            if self.static_sectors_params[sector_key]["static_obs_section"]:
                # If the flag is true, add its 's_start' and 's_end' to the active list
                self.active_static_zones.append({
                    's_start': self.static_sectors_params[sector_key]['s_start'],
                    's_end': self.static_sectors_params[sector_key]['s_end']
                })

    def frenet_pose_cb(self, data: Odometry):
        """
        Callback for the car's Frenet position. Updates current 's' value and triggers the check.
        """
        self.cur_s = data.pose.pose.position.x
        self._check_and_publish()

    def _check_and_publish(self):
        """
        Checks if the current 's' position is within any of the active zones and publishes the result.
        """
        is_in_active_zone = False
        for zone in self.active_static_zones:
            # Check if current s-coordinate is between the start and end of an active zone
            if zone['s_start'] <= self.cur_s <= zone['s_end']:
                is_in_active_zone = True
                break # Found a match, no need to check further

        # Publish the result as a boolean message
        self.check_pub.publish(is_in_active_zone)

if __name__ == '__main__':
    try:
        checker = StaticSectorChecker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass