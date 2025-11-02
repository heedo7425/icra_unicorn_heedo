#include "frenet_conversion.h"
#include <math.h>
#include <ros/ros.h>

namespace frenet_conversion{

  FrenetConverter::FrenetConverter() {}

  FrenetConverter::~FrenetConverter() {}

  /*** -------------- PUBLIC ----------------- ***/

  // ===== HJ ADDED: Map loading implementation =====
  void FrenetConverter::SetOccupancyMap(const nav_msgs::OccupancyGridConstPtr& map_msg) {
    if (!map_msg) {
      ROS_ERROR("[FrenetConverter] Received null map message");
      return;
    }

    // Extract map metadata
    map_resolution_ = map_msg->info.resolution;
    map_origin_x_ = map_msg->info.origin.position.x;
    map_origin_y_ = map_msg->info.origin.position.y;

    int width = map_msg->info.width;
    int height = map_msg->info.height;

    // Convert OccupancyGrid to OpenCV Mat
    // OccupancyGrid: -1 (unknown), 0 (free), 100 (occupied)
    // Convert to: 255 (free), 0 (occupied), 128 (unknown)
    occupancy_grid_ = cv::Mat(height, width, CV_8UC1);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int index = x + (height - 1 - y) * width;  // Flip Y-axis
        int8_t cell_value = map_msg->data[index];

        if (cell_value == -1) {
          // Unknown space - treat as occupied for safety
          occupancy_grid_.at<uchar>(y, x) = 0;
        } else if (cell_value == 0) {
          // Free space
          occupancy_grid_.at<uchar>(y, x) = 255;
        } else {
          // Occupied (cell_value == 100 or other positive values)
          occupancy_grid_.at<uchar>(y, x) = 0;
        }
      }
    }

    has_occupancy_map_ = true;
    ROS_DEBUG("[FrenetConverter] Occupancy map loaded from ROS: %dx%d pixels, resolution=%.3f m/pixel",
              width, height, map_resolution_);
    ROS_DEBUG("[FrenetConverter] Map origin: (%.2f, %.2f)", map_origin_x_, map_origin_y_);
  }
  // ===== HJ ADDED END =====

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

    UpdateClosestIndex(x, y, idx, full_search);
  

    // calculate frenet point
    CalcFrenetPoint(x, y, s, d);
  }

  void FrenetConverter::GetFrenetOdometry(const double x, const double y, 
                                          const double theta, const double v_x, 
                                          const double v_y, double* s, 
                                          double* d, double* v_s, double* v_d,
                                          int* idx) {
    
    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }
    std::unique_lock<std::mutex> lock(mutexGlobalTrajectory_);   

    UpdateClosestIndex(x, y, idx, false);

    CalcFrenetPoint(x, y, s, d);

    CalcFrenetVelocity(v_x, v_y, theta, v_s, v_d);
    lock.unlock();
  }

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
    UpdateClosestIndex(x, y, idx, false);
  }

  /*** -------------- PRIVATE ----------------- ***/

  void FrenetConverter::CalcFrenetPoint(const double x, const double y,
                                             double* s, double* d) {
    // project current position onto trajectory:
    // s = s_wp + cos(phi)*dx + sin(phi)*dy
    double d_x = x - wpt_array_.at(closest_idx_).x_m;
    double d_y = y - wpt_array_.at(closest_idx_).y_m;

    *s = d_x * std::cos(wpt_array_.at(closest_idx_).psi_rad) +
        d_y * std::sin(wpt_array_.at(closest_idx_).psi_rad) + 
        wpt_array_.at(closest_idx_).s_m;
    if (is_closed_contour_) {
       // limit to length of global trajectory
      *s = std::fmod(*s, global_trajectory_length_);
    }
    *d = - d_x * std::sin(wpt_array_.at(closest_idx_).psi_rad) +
        d_y * std::cos(wpt_array_.at(closest_idx_).psi_rad);
  }

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

  void FrenetConverter::CalcFrenetVelocity(const double v_x, const double v_y, 
                                          const double theta, double* v_s, 
                                          double* v_d) {
    double delta_psi = theta - wpt_array_.at(closest_idx_).psi_rad;
    *v_s = v_x * std::cos(delta_psi) - v_y * std::sin(delta_psi);
    *v_d = v_x * std::sin(delta_psi) + v_y * std::cos(delta_psi);
  }

  // TODO speed up by using intelligent search (binary search)
  void FrenetConverter::UpdateClosestIndex(const double x, const double y,
                                           int* idx, bool full_search) {
    // ===== HJ MODIFIED: Add wall-crossing check with rotational search =====

    // Step 1: Find nearest waypoint using original algorithm
    double min_dist = INFINITY;
    int search_ahead_radius = 20; // TODO don't hardcode this
    int start = (wpt_array_.size() + *idx - 1) % wpt_array_.size();
    int end_idx = (start + search_ahead_radius) % wpt_array_.size();

    // -- proximity search: --
    if (!full_search) {
      if (end_idx < start) {
        // search end of array
        for (int i = start; i < wpt_array_.size(); i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                            std::pow(y - wpt_array_[i].y_m, 2);
          if (d_squared < min_dist) {
            min_dist = d_squared;
            closest_idx_ = i;
          }
        }
        for (int i = 0; i < end_idx; i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                            std::pow(y - wpt_array_[i].y_m, 2);
          if (d_squared < min_dist) {
            min_dist = d_squared;
            closest_idx_ = i;
          }
        }
      } else {
        for (int i = start; i < end_idx; i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                            std::pow(y - wpt_array_[i].y_m, 2);
          if (d_squared < min_dist) {
            min_dist = d_squared;
            closest_idx_ = i;
          }
        }
      }
    }
    // if we didn't find anything good in proximity, search the whole array
    if (min_dist > 4 || full_search) { // TODO don't hardcode this
      //ROS_WARN_STREAM("[Converter] Searching Entire Array");
      for (int i = 0; i < wpt_array_.size(); i++) {
        double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                          std::pow(y - wpt_array_[i].y_m, 2);
        if (d_squared < min_dist) {
          min_dist = d_squared;
          closest_idx_ = i;
        }
      }
    }

    // Step 2: Check if wall crossing occurs
    if (has_occupancy_map_ && closest_idx_ >= 0) {
      if (!IsLineCrossingObstacle(x, y, wpt_array_[closest_idx_].x_m,
                                  wpt_array_[closest_idx_].y_m)) {
        // No wall crossing - use this waypoint
        prev_valid_closest_idx_ = closest_idx_;
        *idx = (closest_idx_ + 1);
        return;
      }

      // Step 3: Wall detected - try rotational search
      ROS_DEBUG_THROTTLE(1.0, "[FrenetConverter] Wall detected, trying rotational search");

      double vec_x = wpt_array_[closest_idx_].x_m - x;
      double vec_y = wpt_array_[closest_idx_].y_m - y;

      // Try 90°, 180°, 270° rotations
      int angles[] = {90, 180, 270};
      for (int angle_deg : angles) {
        double angle_rad = angle_deg * M_PI / 180.0;
        double cos_a = std::cos(angle_rad);
        double sin_a = std::sin(angle_rad);

        // Rotate vector
        double rotated_x = vec_x * cos_a - vec_y * sin_a;
        double rotated_y = vec_x * sin_a + vec_y * cos_a;

        // Target point
        double target_x = x + rotated_x;
        double target_y = y + rotated_y;

        // Find nearest waypoint to target
        int candidate_idx = FindNearestWaypointToPoint(target_x, target_y);

        if (candidate_idx >= 0 &&
            !IsLineCrossingObstacle(x, y, wpt_array_[candidate_idx].x_m,
                                   wpt_array_[candidate_idx].y_m)) {

          // Step 4: Search in s-direction for better waypoint
          int best_idx = candidate_idx;
          double best_dist = std::pow(x - wpt_array_[candidate_idx].x_m, 2) +
                             std::pow(y - wpt_array_[candidate_idx].y_m, 2);

          // Search ALL waypoints in both s directions until wall hit
          // Forward direction
          for (size_t s_offset = 1; s_offset < wpt_array_.size(); s_offset++) {
            int test_idx = (candidate_idx + s_offset) % wpt_array_.size();

            // Check if wrapped around to starting point
            if (test_idx == candidate_idx) {
              break;
            }

            if (!IsLineCrossingObstacle(x, y, wpt_array_[test_idx].x_m,
                                       wpt_array_[test_idx].y_m)) {
              double dist = std::pow(x - wpt_array_[test_idx].x_m, 2) +
                           std::pow(y - wpt_array_[test_idx].y_m, 2);
              if (dist < best_dist) {
                best_dist = dist;
                best_idx = test_idx;
              }
            } else {
              break;  // Hit wall, stop searching forward
            }
          }

          // Backward direction
          for (size_t s_offset = 1; s_offset < wpt_array_.size(); s_offset++) {
            int test_idx = (candidate_idx - s_offset + wpt_array_.size()) % wpt_array_.size();

            // Check if wrapped around to starting point
            if (test_idx == candidate_idx) {
              break;
            }

            if (!IsLineCrossingObstacle(x, y, wpt_array_[test_idx].x_m,
                                       wpt_array_[test_idx].y_m)) {
              double dist = std::pow(x - wpt_array_[test_idx].x_m, 2) +
                           std::pow(y - wpt_array_[test_idx].y_m, 2);
              if (dist < best_dist) {
                best_dist = dist;
                best_idx = test_idx;
              }
            } else {
              break;  // Hit wall, stop searching backward
            }
          }

          closest_idx_ = best_idx;
          prev_valid_closest_idx_ = best_idx;
          *idx = (best_idx + 1);
          ROS_DEBUG_THROTTLE(1.0, "[FrenetConverter] Wall avoided via %d° rotation", angle_deg);
          return;
        }
      }

      // Step 5: All directions blocked - use nearest distance (ignoring walls)
      ROS_WARN_THROTTLE(1.0, "[FrenetConverter] All directions blocked, using nearest waypoint");
      prev_valid_closest_idx_ = closest_idx_;
      *idx = (closest_idx_ + 1);
      return;
    }

    // No occupancy map or no wall check needed
    *idx = (closest_idx_ + 1); // account for removing the first element
    // ===== HJ MODIFIED END =====
  }

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

  // ===== HJ ADDED: Wall-crossing detection implementation =====
  bool FrenetConverter::IsLineCrossingObstacle(double x1, double y1, double x2, double y2) {
    if (!has_occupancy_map_) return false;

    // Convert world coordinates to grid coordinates
    // Note: PNG has origin at top-left, Y increases downward
    int px1 = (x1 - map_origin_x_) / map_resolution_;
    int py1 = occupancy_grid_.rows - 1 - (y1 - map_origin_y_) / map_resolution_;
    int px2 = (x2 - map_origin_x_) / map_resolution_;
    int py2 = occupancy_grid_.rows - 1 - (y2 - map_origin_y_) / map_resolution_;

    // Bresenham's line algorithm with max iteration safety
    int dx = std::abs(px2 - px1);
    int dy = std::abs(py2 - py1);
    int sx = (px1 < px2) ? 1 : -1;
    int sy = (py1 < py2) ? 1 : -1;
    int err = dx - dy;

    int px = px1, py = py1;
    int max_iterations = std::max(occupancy_grid_.cols, occupancy_grid_.rows) * 2;  // Safety limit
    int iteration = 0;

    while (iteration < max_iterations) {
      iteration++;

      // Boundary check
      if (px < 0 || px >= occupancy_grid_.cols ||
          py < 0 || py >= occupancy_grid_.rows) {
        return true;  // Out of bounds = wall
      }

      // Check pixel value (0=black=occupied, 255=white=free)
      uchar pixel_value = occupancy_grid_.at<uchar>(py, px);
      if (pixel_value < 128) {  // Threshold for occupied
        return true;  // Wall detected
      }

      // Reached end point
      if (px == px2 && py == py2) break;

      // Bresenham step
      int e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        px += sx;
      }
      if (e2 < dx) {
        err += dx;
        py += sy;
      }
    }

    return false;  // No wall crossed
  }

  int FrenetConverter::FindNearestWaypointToPoint(double target_x, double target_y) {
    double min_dist = INFINITY;
    int nearest_idx = -1;

    for (size_t i = 0; i < wpt_array_.size(); i++) {
      double dist_squared = std::pow(target_x - wpt_array_[i].x_m, 2) +
                           std::pow(target_y - wpt_array_[i].y_m, 2);
      if (dist_squared < min_dist) {
        min_dist = dist_squared;
        nearest_idx = i;
      }
    }

    return nearest_idx;
  }
  // ===== HJ ADDED END =====

} // end of namespace frenet_conversion
