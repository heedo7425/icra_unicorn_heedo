cmake_minimum_required(VERSION 3.0.2)
project(mpc_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  f110_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  node/mpc_planner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
