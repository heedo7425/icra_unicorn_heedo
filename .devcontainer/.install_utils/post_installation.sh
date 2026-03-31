#! /bin/bash

# extend .bashrc
cat /home/${USER}/catkin_ws/src/race_stack/.devcontainer/.install_utils/bashrc_ext >> ~/.bashrc

### HJ : ensure catkin workspace always extends /opt/ros/noetic
# Without this, _setup_util.py can be generated with a CMAKE_PREFIX_PATH
# that excludes /opt/ros/noetic, causing PYTHONPATH to lose genmsg etc.
source /opt/ros/noetic/setup.bash
cd /home/${USER}/catkin_ws
catkin config --extend /opt/ros/noetic

# install dependencies
pip install ~/catkin_ws/src/race_stack/f110_utils/libs/ccma
# pip install ~/catkin_ws/src/race_stack/planner/graph_based_planner/src/GraphBasedPlanner

# build (includes GLIL: glim, glim_ros - thirdparty libs already in Docker image)
cd /home/${USER}/catkin_ws
catkin build

# source for additional build
source /opt/ros/noetic/setup.bash && source /home/${USER}/catkin_ws/devel/setup.bash

# additional build
cd /home/${USER}/catkin_ws
catkin build

# f1tenth_simulator build
cd /home/${USER}/catkin_ws
catkin build f1tenth_simulator

# particle_filter build
cd /home/${USER}/catkin_ws
catkin build particle_filter
cd /home/${USER}/catkin_ws/src/race_stack/state_estimation/particle_filter_python3/range_libc/pywrapper
chmod +x compile.sh
./compile.sh

# source
source /opt/ros/noetic/setup.bash && source /home/${USER}/catkin_ws/devel/setup.bash

catkin build abp_detection
source /opt/ros/noetic/setup.bash && source /home/${USER}/catkin_ws/devel/setup.bash

# python privileges
find /home/${USER}/catkin_ws -type f -name "*.py" -exec chmod +x {} \;


