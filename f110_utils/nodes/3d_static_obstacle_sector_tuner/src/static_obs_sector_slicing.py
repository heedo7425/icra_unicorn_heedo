#!/usr/bin/env python3
### HJ : 3D version of static_obs_sector_slicing.py — matplotlib 3D with slider/button
import rospy, rospkg
import yaml, os, subprocess, time
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import MarkerArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

class StaticObstacleSectorSlicer:
    def __init__(self):
        rospy.init_node('static_obs_sector_slicer_node', anonymous=True)

        self.glb_wpnts = None
        self.track_bounds = None

        self.sector_pnts_indices = [0]

        rospy.Subscriber('/global_waypoints', WpntArray, self.glb_wpnts_cb)
        rospy.Subscriber('/trackbounds/markers', MarkerArray, self.bounds_cb)

        self.yaml_dir = rospy.get_param('~save_dir')

    def glb_wpnts_cb(self, data):
        self.glb_wpnts = data

    def bounds_cb(self, data):
        self.track_bounds = data

    def slice_loop(self):
        rospy.loginfo('Waiting for global waypoints...')
        rospy.wait_for_message('/global_waypoints', WpntArray)
        while self.glb_wpnts is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo('Waiting for track bounds...')
        rospy.wait_for_message('/trackbounds/markers', MarkerArray)
        while self.track_bounds is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo('Waypoints and bounds received. Starting 3D GUI.')

        self.sector_gui()
        rospy.loginfo(f'Selected Static Obstacle Sector Indices: {self.sector_pnts_indices}')

        self.sectors_to_yaml()

    def sector_gui(self):
        x = np.array([w.x_m for w in self.glb_wpnts.wpnts])
        y = np.array([w.y_m for w in self.glb_wpnts.wpnts])
        z = np.array([w.z_m for w in self.glb_wpnts.wpnts])
        s = np.array([w.s_m for w in self.glb_wpnts.wpnts])

        bnd_x = np.array([m.pose.position.x for m in self.track_bounds.markers])
        bnd_y = np.array([m.pose.position.y for m in self.track_bounds.markers])
        bnd_z = np.array([m.pose.position.z for m in self.track_bounds.markers])

        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_axes([0.05, 0.25, 0.9, 0.7], projection='3d')
        axslider = fig.add_axes([0.15, 0.15, 0.7, 0.03])
        axselect = fig.add_axes([0.15, 0.08, 0.3, 0.05])
        axfinish = fig.add_axes([0.55, 0.08, 0.3, 0.05])

        def update_map(cur_idx):
            ax1.cla()
            ax1.plot(x, y, z, 'm-', linewidth=0.7)
            ax1.plot(bnd_x, bnd_y, bnd_z, 'g-', linewidth=0.4)
            ax1.scatter(x[cur_idx], y[cur_idx], z[cur_idx], c='red', s=50, zorder=10)
            if len(self.sector_pnts_indices) > 0:
                ax1.scatter(x[self.sector_pnts_indices], y[self.sector_pnts_indices],
                            z[self.sector_pnts_indices], c='green', s=50, zorder=10)
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('y [m]')
            ax1.set_zlabel('z [m]')
            ax1.set_title('Static Obs Sector Slicing (idx=%d, s=%.1fm)' % (cur_idx, s[cur_idx]))
            ax1.view_init(elev=90, azim=-90)

        update_map(0)

        def update_s(val):
            idx = int(slider.val)
            if idx >= len(s):
                idx = len(s) - 1
            update_map(cur_idx=idx)
            self.glob_slider_idx = idx
            fig.canvas.draw_idle()

        def select_s(event):
            self.sector_pnts_indices.append(self.glob_slider_idx)
            update_map(cur_idx=self.glob_slider_idx)
            fig.canvas.draw_idle()

        def finish(event):
            plt.close()
            self.sector_pnts_indices.append(len(s) - 1)
            self.sector_pnts_indices = sorted(list(set(self.sector_pnts_indices)))

        self.glob_slider_idx = 0

        slider = Slider(axslider, 'Waypoint idx', 0, len(s)-1, valinit=0, valfmt='%d')
        slider.on_changed(update_s)

        btn_select = Button(axselect, 'Select Static Obs S')
        btn_select.on_clicked(select_s)

        btn_finish = Button(axfinish, 'Done')
        btn_finish.on_clicked(finish)

        plt.show()

    def sectors_to_yaml(self):
        if len(self.sector_pnts_indices) <= 1:
            rospy.logwarn("No sectors selected. Creating a single sector for the whole track.")
            self.sector_pnts_indices = [0, len(self.glb_wpnts.wpnts) - 1]

        n_sectors = len(self.sector_pnts_indices) - 1
        dict_file = {'n_sectors': n_sectors}

        for i in range(n_sectors):
            start_idx = self.sector_pnts_indices[i]
            end_idx = self.sector_pnts_indices[i + 1]

            s_start = self.glb_wpnts.wpnts[start_idx].s_m
            s_end = self.glb_wpnts.wpnts[end_idx].s_m

            sector_key = f"Static_Obs_sector{i}"
            dict_file[sector_key] = {
                'start': int(start_idx),
                'end': int(end_idx),
                's_start': float(s_start),
                's_end': float(s_end),
                'name': f"sector_{i + 1}",
                'static_obs_section': False
            }

        yaml_path = os.path.join(self.yaml_dir, 'static_obs_sectors.yaml')
        with open(yaml_path, 'w') as file:
            rospy.loginfo(f'Dumping to {yaml_path}: {dict_file}')
            yaml.dump(dict_file, file, default_flow_style=False, sort_keys=False)

        ros_path = rospkg.RosPack().get_path('static_obstacle_sector_tuner_3d')
        cfg_yaml_path = os.path.join(ros_path, 'cfg/static_obs_sectors.yaml')
        with open(cfg_yaml_path, 'w') as file:
            rospy.loginfo(f'Dumping to {cfg_yaml_path}: {dict_file}')
            yaml.dump(dict_file, file, default_flow_style=False, sort_keys=False)

        time.sleep(1)
        rospy.loginfo('Building static_obstacle_sector_tuner_3d...')
        shell_dir = os.path.join(ros_path, 'scripts/finish_sector.sh')
        if os.path.exists(shell_dir):
            subprocess.Popen(shell_dir, shell=True)

if __name__ == "__main__":
    slicer = StaticObstacleSectorSlicer()
    slicer.slice_loop()
