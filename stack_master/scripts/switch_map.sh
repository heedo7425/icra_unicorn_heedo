#!/bin/bash

# This script transfers the YAML files from the given map in stack_master to the
# overtaking_sector_tuner, sector_tuner, static_obstacle_sector_tuner, and friction_sector_tuner.

cd "${0%/*}"
cd ../..
# This ensures the current working directory is race_stack

MAP_PATH="stack_master/maps"

if [ $# -gt 0 ]; then
    echo "Listing contents of map $1"
    ls $MAP_PATH/$1
    echo -e "\n"

    echo "Copying Overtaking Sector Tuner files..."
    cp $MAP_PATH/$1/ot_sectors.yaml f110_utils/nodes/overtaking_sector_tuner/cfg/ot_sectors.yaml
    echo -e "\n"

    echo "Copying Sector Tuner files..."
    cp $MAP_PATH/$1/speed_scaling.yaml f110_utils/nodes/sector_tuner/cfg/speed_scaling.yaml
    echo -e "\n"

    echo "Copying Static Obstacle Sector Tuner files..."
    cp $MAP_PATH/$1/static_obs_sectors.yaml f110_utils/nodes/static_obstacle_sector_tuner/cfg/static_obs_sectors.yaml
    echo -e "\n"

    echo "Copying Friction Sector Tuner files..."
    cp $MAP_PATH/$1/friction_scaling.yaml f110_utils/nodes/friction_sector_tuner/cfg/friction_scaling.yaml
    echo -e "\n"

    echo "Building ..."
    cd ../..;   # now we are in catkin_ws
    catkin clean sector_tuner overtaking_sector_tuner static_obstacle_sector_tuner friction_sector_tuner &&  \
    catkin build sector_tuner overtaking_sector_tuner static_obstacle_sector_tuner friction_sector_tuner &&  \
    source ~/catkin_ws/devel/setup.bash;

    echo -e "\n"
    echo "Done! Please remember to source in your other terminals."
else
    echo "Please specify a map name for this to work."
    echo "A list of local maps in the stack_master folder:"
    ls -la $MAP_PATH
    echo -e "\n#######################################################################\n"
fi
