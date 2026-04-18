cmake_minimum_required(VERSION 3.0.2)
project(sampling_based_planner_3d)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  f110_msgs
  dynamic_reconfigure
)

### HJ : dynamic_reconfigure for state-aware sampling planner (SamplingCost.cfg)
generate_dynamic_reconfigure_options(
  cfg/SamplingCost.cfg
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    nav_msgs
    geometry_msgs
    visualization_msgs
    f110_msgs
    dynamic_reconfigure
)

catkin_install_python(PROGRAMS
  node/sampling_planner_node.py
  ### HJ : state-aware variant used by 3d_headtohead with sampling_planner_enable:=true
  node/sampling_planner_state_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
