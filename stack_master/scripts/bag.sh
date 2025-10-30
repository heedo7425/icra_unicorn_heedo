#!/bin/bash

# HJ ADD: Create record directory if it doesn't exist
BAG_DIR="/home/hj/catkin_ws/src/race_stack/stack_master/bag"
RECORD_DIR="$BAG_DIR/record"

mkdir -p "$RECORD_DIR"

cd "$RECORD_DIR"

rosbag record \
    /scan \
    /imu/data \
    /vesc/low_level/ackermann_cmd_mux/output \
    /car_state/odom \
    /car_state/odom \
    /car_state/odom_frenet \
    /behavior_strategy \
