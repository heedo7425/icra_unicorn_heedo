cmake_minimum_required(VERSION 3.0.2)
project(race_day_id)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  nav_msgs
  sensor_msgs
  ackermann_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs std_srvs nav_msgs sensor_msgs ackermann_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  src/race_day_id/orchestrator_node.py
  scripts/inject_synthetic_gt.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
