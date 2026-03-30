cmake_minimum_required(VERSION 3.0.2)
project(global_line_3d)

find_package(catkin REQUIRED)
catkin_package()

# Make scripts available as ROS nodes
install(PROGRAMS scripts/run_pipeline.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install scripts
install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
  USE_SOURCE_PERMISSIONS
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
