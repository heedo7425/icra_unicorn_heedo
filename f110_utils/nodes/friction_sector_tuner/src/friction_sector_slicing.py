#!/usr/bin/env python3
### HJ : 3D version of sector_slicing.py — matplotlib 3D with slider/button (same UI as 2D)
import rospy, rospkg
import yaml, os, subprocess, time
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import MarkerArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

class SectorSlicer:
    def __init__(self):
        rospy.init_node('sector_node', anonymous=True)

        self.glb_wpnts = None
        self.track_bounds = None

        self.speed_scaling = 1.0
        self.glob_slider_s = 0
        self.sector_pnts = [0]

        rospy.Subscriber('/global_waypoints', WpntArray, self.glb_wpnts_cb)
        rospy.Subscriber('/trackbounds/markers', MarkerArray, self.bounds_cb)

        self.yaml_dir = rospy.get_param('~save_dir')

    def glb_wpnts_cb(self, data):
        self.glb_wpnts = data

    def bounds_cb(self, data):
        self.track_bounds = data

    def slice_loop(self):
        print('Waiting for global waypoints...')
        rospy.wait_for_message('/global_waypoints', WpntArray)
        while self.glb_wpnts is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        print('Waiting for track bounds...')
        rospy.wait_for_message('/trackbounds/markers', MarkerArray)
        while self.track_bounds is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        self.sector_gui()
        print('Selected Sector IDXs:', self.sector_pnts)

        self.sectors_to_yaml()

    def sector_gui(self):
        s = np.array([w.s_m for w in self.glb_wpnts.wpnts])
        x = np.array([w.x_m for w in self.glb_wpnts.wpnts])
        y = np.array([w.y_m for w in self.glb_wpnts.wpnts])
        z = np.array([w.z_m for w in self.glb_wpnts.wpnts])

        bnd_x = np.array([m.pose.position.x for m in self.track_bounds.markers])
        bnd_y = np.array([m.pose.position.y for m in self.track_bounds.markers])
        bnd_z = np.array([m.pose.position.z for m in self.track_bounds.markers])

        ### HJ : 3D plot with slider/button (same UI structure as 2D original)
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_axes([0.05, 0.25, 0.9, 0.7], projection='3d')
        axslider = fig.add_axes([0.15, 0.15, 0.7, 0.03])
        axselect = fig.add_axes([0.15, 0.08, 0.3, 0.05])
        axfinish = fig.add_axes([0.55, 0.08, 0.3, 0.05])

        def update_map(cur_s):
            ax1.cla()
            ax1.plot(x, y, z, color='orange', linewidth=0.7)
            ax1.plot(bnd_x, bnd_y, bnd_z, 'g-', linewidth=0.4)
            ax1.scatter(x[cur_s], y[cur_s], z[cur_s], c='red', s=50, zorder=10)
            if len(self.sector_pnts) > 0:
                ax1.scatter(x[self.sector_pnts], y[self.sector_pnts], z[self.sector_pnts],
                            c='green', s=50, zorder=10)
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('y [m]')
            ax1.set_zlabel('z [m]')
            ax1.set_title('Friction Sector Slicing (idx=%d, s=%.1fm)' % (cur_s, s[cur_s]))
            ax1.view_init(elev=90, azim=-90)

        update_map(0)

        def update_s(val):
            idx = int(slider.val)
            if idx >= len(s):
                idx = len(s) - 1
            self.glob_slider_s = idx
            update_map(cur_s=idx)
            fig.canvas.draw_idle()

        def select_s(event):
            self.sector_pnts.append(self.glob_slider_s)
            update_map(cur_s=self.glob_slider_s)
            fig.canvas.draw_idle()

        def finish(event):
            plt.close()
            self.sector_pnts.append(len(s))
            self.sector_pnts = sorted(list(set(self.sector_pnts)))

        slider = Slider(axslider, 'Waypoint idx', 0, len(s)-1, valinit=0, valfmt='%d')
        slider.on_changed(update_s)

        btn_select = Button(axselect, 'Select S')
        btn_select.on_clicked(select_s)

        btn_finish = Button(axfinish, 'Done')
        btn_finish.on_clicked(finish)

        plt.show()

    def sectors_to_yaml(self):
        if len(self.sector_pnts) == 1:
            self.sector_pnts.append(len(self.glb_wpnts.wpnts))
        n_sectors = len(self.sector_pnts) - 1
        dict_file = {'global_friction': 1.0, 'n_sectors': n_sectors}
        for i in range(0, n_sectors):
            dict_file['Sector' + str(i)] = {
                'start': self.sector_pnts[i] if i == 0 else self.sector_pnts[i] + 1,
                'end': self.sector_pnts[i+1],
                'friction': 1.0,
            }

        yaml_path = os.path.join(self.yaml_dir, 'friction_scaling.yaml')
        with open(yaml_path, 'w') as file:
            print('Dumping to {}: {}'.format(yaml_path, dict_file))
            yaml.dump(dict_file, file, sort_keys=False)

        # Copy to friction_sector_tuner package
        ros_path = rospkg.RosPack().get_path('friction_sector_tuner')
        yaml_path = os.path.join(ros_path, 'cfg/friction_scaling.yaml')
        with open(yaml_path, 'w') as file:
            print('Dumping to {}: {}'.format(yaml_path, dict_file))
            yaml.dump(dict_file, file, sort_keys=False)

        time.sleep(1)
        print('Building friction_sector_tuner...')
        shell_dir = os.path.join(ros_path, 'scripts/finish_sector.sh')
        if os.path.exists(shell_dir):
            subprocess.Popen(shell_dir, shell=True)

if __name__ == "__main__":
    sector_slicer = SectorSlicer()
    sector_slicer.slice_loop()
