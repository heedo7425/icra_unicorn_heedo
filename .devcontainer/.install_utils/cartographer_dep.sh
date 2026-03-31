#! /bin/bash

# merge cartographer ws with ours
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y
