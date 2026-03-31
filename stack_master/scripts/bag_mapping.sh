#!/bin/bash

# Create bag_mapping directory if it doesn't exist
# Automatically use current user's home directory
USER_HOME="$HOME"
BAG_DIR="$USER_HOME/catkin_ws/src/race_stack/stack_master/bag"
MAPPING_DIR="$BAG_DIR/bag_mapping"

mkdir -p "$MAPPING_DIR"

cd "$MAPPING_DIR"

rosbag record \
    /scan \
    /vesc/sensors/imu/raw \
    /vesc/sensors/core \
    /vesc/sensors/servo_position_command \
    /car_state/odom \
