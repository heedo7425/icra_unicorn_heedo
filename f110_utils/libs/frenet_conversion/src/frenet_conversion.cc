#include "frenet_conversion.h"
#include <math.h>
#include <ros/ros.h>

namespace frenet_conversion{

  FrenetConverter::FrenetConverter() {}

  FrenetConverter::~FrenetConverter() {}

  /*** -------------- PUBLIC ----------------- ***/

  void FrenetConverter::SetGlobalTrajectory(
                                    const std::vector<f110_msgs::Wpnt> *wptns,
                                    const bool is_closed_contour) {
    // deep copy wptns
    wpt_array_ = *wptns;
    global_trajectory_length_ = wpt_array_.back().s_m;
    is_closed_contour_ = is_closed_contour;
    // remove the first element -> no negative s
    if (is_closed_contour_) {
      wpt_array_.erase(wpt_array_.begin());
    }
    has_global_trajectory_ = true;
  }

  void FrenetConverter::GetFrenetPoint(const double x, const double y, 
                                       double* s, double* d, int* idx, bool full_search) {

    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }

    UpdateClosestIndex(x, y, 0.0, idx, full_search);

    // calculate frenet point (z=0 for 2D callers)
    CalcFrenetPoint(x, y, 0.0, s, d);
  }

  // ### iy : add z for 3D closest-point search + first_call full search
  // ### HJ : CalcFrenetPoint uses 3D tangent, CalcFrenetVelocity stays 2D (body frame twist)
  void FrenetConverter::GetFrenetOdometry(const double x, const double y,
                                          const double z, const double theta,
                                          const double v_x, const double v_y,
                                          double* s, double* d, double* v_s,
                                          double* v_d, int* idx) {

    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }
    std::unique_lock<std::mutex> lock(mutexGlobalTrajectory_);

    // full search on first call to correctly initialize from any start position
    UpdateClosestIndex(x, y, z, idx, first_call_);
    if (first_call_) first_call_ = false;

    CalcFrenetPoint(x, y, z, s, d);

    CalcFrenetVelocity(v_x, v_y, theta, v_s, v_d);
    lock.unlock();
  }
  // ### HJ : end

  void FrenetConverter::GetGlobalPoint(const double s, const double d, 
                                       double* x, double* y) {

    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }
    std::unique_lock<std::mutex> lock(mutexGlobalTrajectory_);

    UpdateClosestIndex(s);
    
    // calculate frenet point
    CalcGlobalPoint(s, d, x, y);
    lock.unlock();
  }

  void FrenetConverter::GetClosestIndex(const double s, int* idx) {
    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }
    UpdateClosestIndex(s);
    *idx = (closest_idx_ + 1); // account for removing the first element
  }

  void FrenetConverter::GetClosestIndex(const double x, const double y, 
                                         int* idx) {
    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }
    UpdateClosestIndex(x, y, 0.0, idx, false);
  }

  /*** -------------- PRIVATE ----------------- ***/

  // ### HJ : 3D tangent projection for slopes — use adjacent waypoints to compute 3D tangent
  void FrenetConverter::CalcFrenetPoint(const double x, const double y,
                                             const double z, double* s, double* d) {
    double d_x = x - wpt_array_.at(closest_idx_).x_m;
    double d_y = y - wpt_array_.at(closest_idx_).y_m;
    double d_z = z - wpt_array_.at(closest_idx_).z_m;

    // compute 3D tangent from adjacent waypoints
    int next_idx = (closest_idx_ + 1) % wpt_array_.size();
    double tx = wpt_array_.at(next_idx).x_m - wpt_array_.at(closest_idx_).x_m;
    double ty = wpt_array_.at(next_idx).y_m - wpt_array_.at(closest_idx_).y_m;
    double tz = wpt_array_.at(next_idx).z_m - wpt_array_.at(closest_idx_).z_m;
    double t_norm = std::sqrt(tx*tx + ty*ty + tz*tz);
    if (t_norm > 1e-9) { tx /= t_norm; ty /= t_norm; tz /= t_norm; }

    // lateral direction: perpendicular to tangent in xy plane, then normalize
    // (lateral is horizontal — vehicle doesn't slide sideways off the slope)
    double nx = -ty;
    double ny = tx;
    double n_norm_xy = std::sqrt(nx*nx + ny*ny);
    if (n_norm_xy > 1e-9) { nx /= n_norm_xy; ny /= n_norm_xy; }

    // s = projection onto 3D tangent + waypoint s_m
    *s = (d_x * tx + d_y * ty + d_z * tz) + wpt_array_.at(closest_idx_).s_m;
    if (is_closed_contour_) {
      *s = std::fmod(*s, global_trajectory_length_);
    }
    // d = projection onto lateral (horizontal normal)
    *d = d_x * nx + d_y * ny;
  }
  // ### HJ : end

  void FrenetConverter::CalcGlobalPoint(const double s, const double d,
                                        double* x, double* y) {
    double d_s = s - wpt_array_.at(closest_idx_).s_m;
    *x = wpt_array_.at(closest_idx_).x_m 
        + d_s * std::cos(wpt_array_.at(closest_idx_).psi_rad) 
        - d * std::sin(wpt_array_.at(closest_idx_).psi_rad);
    *y = wpt_array_.at(closest_idx_).y_m 
        + d * std::cos(wpt_array_.at(closest_idx_).psi_rad) 
        + d_s * std::sin(wpt_array_.at(closest_idx_).psi_rad);
  }

  // GLIL base_odom twist is in body frame — body linear.x is already on-surface speed
  void FrenetConverter::CalcFrenetVelocity(const double v_x, const double v_y,
                                          const double theta, double* v_s,
                                          double* v_d) {
    double delta_psi = theta - wpt_array_.at(closest_idx_).psi_rad;
    *v_s = v_x * std::cos(delta_psi) - v_y * std::sin(delta_psi);
    *v_d = v_x * std::sin(delta_psi) + v_y * std::cos(delta_psi);
  }

  // ### iy : use 3D distance (dx²+dy²+dz²) to disambiguate bridge/slope overlaps
  // TODO speed up by using intelligent search (binary search)
  void FrenetConverter::UpdateClosestIndex(const double x, const double y,
                                           const double z, int* idx,
                                           bool full_search) {
    // get the closest waypoint using 3D distance
    double min_dist = INFINITY;
    int search_ahead_radius = 20; // TODO don't hardcode this
    int start = (wpt_array_.size() + *idx - 1) % wpt_array_.size();
    int end_idx = (start + search_ahead_radius) % wpt_array_.size();

    // -- proximity search: --
    if (!full_search) {
      if (end_idx < start) {
        for (int i = start; i < (int)wpt_array_.size(); i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                             std::pow(y - wpt_array_[i].y_m, 2) +
                             std::pow(z - wpt_array_[i].z_m, 2);
          if (d_squared < min_dist) { min_dist = d_squared; closest_idx_ = i; }
        }
        for (int i = 0; i < end_idx; i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                             std::pow(y - wpt_array_[i].y_m, 2) +
                             std::pow(z - wpt_array_[i].z_m, 2);
          if (d_squared < min_dist) { min_dist = d_squared; closest_idx_ = i; }
        }
      } else {
        for (int i = start; i < end_idx; i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                             std::pow(y - wpt_array_[i].y_m, 2) +
                             std::pow(z - wpt_array_[i].z_m, 2);
          if (d_squared < min_dist) { min_dist = d_squared; closest_idx_ = i; }
        }
      }
    }
    // ### HJ : 1m threshold (track width rarely exceeds 1m)
    if (min_dist > 1.0 || full_search) {
      for (int i = 0; i < (int)wpt_array_.size(); i++) {
        double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                           std::pow(y - wpt_array_[i].y_m, 2) +
                           std::pow(z - wpt_array_[i].z_m, 2);
        if (d_squared < min_dist) { min_dist = d_squared; closest_idx_ = i; }
      }
    }
    *idx = (closest_idx_ + 1); // account for removing the first element
  }
  // ### iy : end

  void FrenetConverter::UpdateClosestIndex(const double s) {
    closest_idx_ = wpt_array_.size() - 1; // lazy don't handle wrapping
    for (int i = 1; i < wpt_array_.size(); i++) {
      // find first waypoint with s > s
      if (wpt_array_[i].s_m > s) {
        if (wpt_array_[i].s_m - s > s - wpt_array_[i-1].s_m) {
          closest_idx_ = i - 1;
          break;
        } else {
          closest_idx_ = i;
          break;
        }
      }
    }
  }

} // end of namespace frenet_conversion
