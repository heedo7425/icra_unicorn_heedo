#!/bin/bash
### HJ : host-side wrapper — calls run_on_container.sh via docker exec
set -e

VEHICLE_NAME="${1:-rc_car_10th}"
RESOLUTION="${2:---fast}"

DOCKER_SCRIPT_DIR="/home/unicorn/catkin_ws/src/race_stack/planner/3d_gb_optimizer/fast_ggv_gen"

docker exec icra2026 bash "$DOCKER_SCRIPT_DIR/run_on_container.sh" "$VEHICLE_NAME" "$RESOLUTION"
