#!/bin/bash

source ~/.bashrc

cd ~/catkin_ws 

#Clean and build sector tuner to force refresh of dynamic reconfigure
catkin clean static_obstacle_sector_tuner
catkin build static_obstacle_sector_tuner