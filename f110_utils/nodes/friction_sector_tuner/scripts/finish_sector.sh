#!/bin/bash

source ~/.bashrc

cd ~/catkin_ws

# Clean and build friction_sector_tuner to force refresh of dynamic reconfigure
catkin clean friction_sector_tuner -y
catkin build friction_sector_tuner
