#include <frenet_odom_republisher_node.h>
#include <tf/tf.h>
#include <string> 

namespace frenet_odom_republisher{

FrenetRepublisher::FrenetRepublisher(ros::NodeHandle& nh):
  nh_(nh){
  InitSubscribersPublishers();
  ros::spin();
}

FrenetRepublisher::~FrenetRepublisher() {}

void FrenetRepublisher::InitSubscribersPublishers() {
  global_trajectory_sub_ = nh_.subscribe<f110_msgs::WpntArray>
      ("/global_waypoints", 10, &FrenetRepublisher::GlobalTrajectoryCallback, this);

  fixed_path_trajectory_sub_ = nh_.subscribe<f110_msgs::OTWpntArray>
      ("/planner/avoidance/smart_static_otwpnts", 10, &FrenetRepublisher::FixedPathTrajectoryCallback, this);

  odom_sub_ = nh_.subscribe<nav_msgs::Odometry>
      ("/odom", 10, &FrenetRepublisher::OdomCallback, this);

  frenet_odom_pub_ = nh_.advertise<nav_msgs::Odometry>
      ("/odom_frenet", 1);

  frenet_odom_fixed_pub_ = nh_.advertise<nav_msgs::Odometry>
      ("/odom_frenet_fixed", 1);
}

void FrenetRepublisher::GlobalTrajectoryCallback(
    const f110_msgs::WpntArrayConstPtr &wpt_array){
  wpt_array_ = wpt_array->wpnts;
  bool enable_wrapping = true;
  frenet_converter_.SetGlobalTrajectory(&wpt_array_, enable_wrapping);
  has_global_trajectory_ = true;
}

void FrenetRepublisher::FixedPathTrajectoryCallback(
    const f110_msgs::OTWpntArrayConstPtr &wpt_array){
  fixed_path_wpt_array_ = wpt_array->wpnts;
  bool enable_wrapping = true;
  frenet_converter_fixed_.SetGlobalTrajectory(&fixed_path_wpt_array_, enable_wrapping);
  has_fixed_path_trajectory_ = true;
}

void FrenetRepublisher::OdomCallback(const nav_msgs::OdometryConstPtr &msg){
  // Get quaternion and yaw for both conversions
  tf::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  // Publish GB path based frenet odom
  if (has_global_trajectory_) {
    nav_msgs::Odometry frenet_odom = *msg;
    frenet_converter_.GetFrenetOdometry(msg->pose.pose.position.x,
                                        msg->pose.pose.position.y,
                                        yaw, msg->twist.twist.linear.x,
                                        msg->twist.twist.linear.y,
                                        &frenet_odom.pose.pose.position.x,
                                        &frenet_odom.pose.pose.position.y,
                                        &frenet_odom.twist.twist.linear.x,
                                        &frenet_odom.twist.twist.linear.y,
                                        &closest_wpt_index_);
    // abuse child frame id to pass closest wapoint index
    frenet_odom.child_frame_id = std::to_string(closest_wpt_index_);
    frenet_odom_pub_.publish(frenet_odom);
  }

  // Publish fixed path based frenet odom
  if (has_fixed_path_trajectory_) {
    nav_msgs::Odometry frenet_odom_fixed = *msg;
    frenet_converter_fixed_.GetFrenetOdometry(msg->pose.pose.position.x,
                                              msg->pose.pose.position.y,
                                              yaw, msg->twist.twist.linear.x,
                                              msg->twist.twist.linear.y,
                                              &frenet_odom_fixed.pose.pose.position.x,
                                              &frenet_odom_fixed.pose.pose.position.y,
                                              &frenet_odom_fixed.twist.twist.linear.x,
                                              &frenet_odom_fixed.twist.twist.linear.y,
                                              &closest_wpt_index_fixed_);
    // abuse child frame id to pass closest wapoint index
    frenet_odom_fixed.child_frame_id = std::to_string(closest_wpt_index_fixed_);
    frenet_odom_fixed_pub_.publish(frenet_odom_fixed);
  }
}

}// end namespace frenet_odom_republisher

// launch node
int main(int argc, char** argv)
{
  ros::init(argc, argv, "frenet_odom_republisher");
  ros::NodeHandle nh;

  frenet_odom_republisher::FrenetRepublisher node(nh);
  return 0;
}