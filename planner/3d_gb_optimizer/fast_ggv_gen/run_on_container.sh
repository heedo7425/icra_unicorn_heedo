#!/bin/bash
### HJ : run inside Docker container (no docker exec needed)
set -e

VEHICLE_NAME="${1:-rc_car_10th}"
RESOLUTION="${2:---fast}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source /opt/ros/noetic/setup.bash
source /home/unicorn/catkin_ws/devel/setup.bash

echo "=============================="
echo " Fast GGV Pipeline (container)"
echo " Vehicle: ${VEHICLE_NAME}"
echo " Resolution: ${RESOLUTION}"
echo "=============================="

cd "$SCRIPT_DIR"

# 1. GGV generation
echo ""
echo "[1/3] Generating GGV diagrams..."
python3 fast_gen_gg_diagrams.py --vehicle_name $VEHICLE_NAME $RESOLUTION

# 2. Diamond fitting
echo ""
echo "[2/3] Fitting diamond representation..."
python3 gen_diamond_representation.py --vehicle_name $VEHICLE_NAME

# 3. Plot
echo ""
echo "[3/3] Generating plots..."
MPLBACKEND=Agg python3 plot_gg_diagrams.py --vehicle_name $VEHICLE_NAME

echo ""
echo "=============================="
echo " DONE"
echo " Output: $SCRIPT_DIR/output/$VEHICLE_NAME/"
echo "=============================="
ls -la "$SCRIPT_DIR/output/$VEHICLE_NAME/"*.png 2>/dev/null || echo " (no PNG files found)"
