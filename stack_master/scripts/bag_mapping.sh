#!/bin/bash

# HJ ADD: Create bag_mapping directory if it doesn't exist
BAG_DIR="/home/hj/catkin_ws/src/race_stack/stack_master/bag"
MAPPING_DIR="$BAG_DIR/bag_mapping"

mkdir -p "$MAPPING_DIR"

cd "$MAPPING_DIR"

rosbag record \
    /scan \
    /imu/data \
    /vesc/sensors/core \
    /vesc/sensors/servo_position_command \
    /car_state/odom \
