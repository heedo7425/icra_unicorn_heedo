#ifndef FRENET_CONVERSION_H_
#define FRENET_CONVERSION_H_

#include <ros/ros.h>
#include <f110_msgs/Wpnt.h>
#include <visualization_msgs/MarkerArray.h>
#include <mutex>
#include <vector>

namespace frenet_conversion{
    
class FrenetConverter {
 public:
  FrenetConverter();
  ~FrenetConverter();

  /**
   * @brief Initializes the frenet converter with the global trajectory
   * 
   * @param wptns pointer to vector containing the waypoints describing the 
   * global trajectory, the last and first waypoints should overlap if the 
   * trajectory is wrapping around
   * 
   */
  void SetGlobalTrajectory(const std::vector<f110_msgs::Wpnt> *wptns,
                           const bool is_closed_contour);

  // ### HJ : set track boundaries for wall-crossing detection
  void SetTrackBounds(const visualization_msgs::MarkerArrayConstPtr& bounds_msg);
  // ### HJ : end

  /**
   * @brief Returns the frenet point corresponding to the given position
   * 
   * @param x input x position
   * @param y input y position
   * @param s returns the frenet s coordinate
   * @param d returns the frenet d coordinate
   * @param idx returns the index of the closest waypoint
   * 
   */
  void GetFrenetPoint(const double x, const double y, double* s, double* d, 
                      int* idx, bool full_search);
  
  /**
   * @brief Returns the global point corresponding to the frenet position
   * 
   * @param s input frenet s coordinate
   * @param d input frenet d coordinate
   * @param x output x position
   * @param y output y position
   * 
   */
  void GetGlobalPoint(const double s, const double d, double* x, double* y);

  /**
   * @brief Get the Closest Index on the global trajectory to the given position
   * 
   * @param x input
   * @param y input
   * @param idx returns the index of the closest waypoint
   */
  void GetClosestIndex(const double x, const double y, int* idx);

  /**
   * @brief Get the Closest Index on the global trajectory to track advancement
   * 
   * @param s input track advancement
   * @param idx returns the index of the closest waypoint
   */
  void GetClosestIndex(const double s, int* idx);

  /**
   * @brief Returns the frenet point and locally projected velocity
   *  corresponding to the given position and velocity
   * 
   * @param x input x position
   * @param y input y position
   * @param theta car heading angle
   * @param v_x input x velocity, in body frame
   * @param v_y input y velocity, in body frame
   * @param s returns the frenet s coordinate
   * @param d returns the frenet d coordinate
   * @param v_s returns the frenet s velocity
   * @param v_d returns the frenet d velocity
   * @param idx returns the index of the closest waypoint
   */
  // ### iy : add z for 3D closest-point search + first_call full search
  void GetFrenetOdometry(const double x, const double y, const double z,
                         const double theta, const double v_x, const double v_y,
                         double* s, double* d, double* v_s, double* v_d, int* idx);
  // ### iy : end

  // ### HJ : force full search from external trigger (e.g. interactive marker)
  void ForceFullSearch() { first_call_ = true; }
  // ### HJ : end

 private:
  /**
   * @brief Calculates the frenet coordinates of the given position
   * 
   * @param x input x position
   * @param y input y position
   * @param s returns the frenet s coordinate
   * @param d returns the frenet d coordinate
   */
  // ### HJ : add z for 3D tangent projection
  void CalcFrenetPoint(const double x, const double y, const double z, double* s, double* d);
  // ### HJ : end

  /**
   * @brief Calculates the global position based on the frenet position
   * 
   * @param s input track advancement
   * @param d input track offset
   * @param x output x position
   * @param y output y position
   */
  void CalcGlobalPoint(const double s, const double d, double* x, double* y);

  /**
   * @brief Projects the velocities into local frenet frame
   * 
   * @param v_x input x velocity, in body frame
   * @param v_y input y velocity, in body frame
   * @param theta car heading angle
   * @param v_s returns the frenet s velocity
   * @param v_d returns the frenet d velocity
   */
  void CalcFrenetVelocity(const double v_x, const double v_y, const double theta,
                          double* v_s, double* v_d);

  /**
   * @brief Updates the closest index of waypoint array to the given position
   * 
   * @param x input x position
   * @param y input y position
   */
  // ### iy : add z for 3D distance in closest-point search
  void UpdateClosestIndex(const double x, const double y, const double z, int* idx, bool full_search);
  // ### iy : end

  /**
   * @brief Updates the closest index of waypoint array to the given 
   * track advancement
   * 
   * @param s 
   */
  void UpdateClosestIndex(const double s);

  // ### HJ : d_height filter — compute normal projection distance
  double CalcHeightOffset(const double x, const double y, const double z, int wpt_idx);
  // ### HJ : z-filtered 2D boundary raycast — check if line crosses track wall
  bool IsLineCrossingBoundary(const double x1, const double y1, const double x2, const double y2, const double z_ref);
  // ### HJ : 2D line segment intersection test
  bool SegmentsIntersect2D(double ax, double ay, double bx, double by,
                           double cx, double cy, double dx, double dy);
  // ### HJ : end

  int closest_idx_;
  std::vector<f110_msgs::Wpnt> wpt_array_;
  bool has_global_trajectory_{false};
  double global_trajectory_length_;
  bool is_closed_contour_;
  std::mutex mutexGlobalTrajectory_;
  // ### iy : full search on first odom call to handle arbitrary start position
  bool first_call_{true};
  // ### iy : end

  // ### HJ : track boundary data for wall-crossing detection
  struct BoundPoint { double x, y, z; };
  std::vector<BoundPoint> left_bounds_;
  std::vector<BoundPoint> right_bounds_;
  bool has_track_bounds_{false};
  double height_filter_threshold_{0.10};  // [m] d_height threshold for layer filtering
  double z_boundary_margin_{0.1};         // [m] z margin for boundary filtering
  // ### HJ : end

};
   
}// end namespace frenet_conversion

#endif /* FRENET_CONVERSION_H_ */