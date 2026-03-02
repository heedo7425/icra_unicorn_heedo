#include "voxblox/dynamic/tracking.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace voxblox {
namespace dynamic {

void Tracking::track(const Pointcloud& cloud_W,
                     std::vector<Cluster>& clusters) {
  trackClusterIDs(cloud_W, clusters);

  for (Cluster& cluster : clusters) {
    if (cluster.track_length >= config_.min_track_duration) {
      cluster.valid = true;
    }
  }
}

void Tracking::trackClusterIDs(const Pointcloud& cloud_W,
                                std::vector<Cluster>& clusters) {
  // Compute centroids.
  std::vector<Point> centroids(clusters.size());
  for (size_t i = 0; i < clusters.size(); ++i) {
    Point centroid = Point::Zero();
    for (size_t idx : clusters[i].point_indices) {
      centroid += cloud_W[idx];
    }
    if (!clusters[i].point_indices.empty()) {
      centroid /= static_cast<float>(clusters[i].point_indices.size());
    }
    centroids[i] = centroid;
  }

  // Compute distance matrix [previous][current].
  struct Association {
    float distance;
    int previous_id;
    int current_id;
  };

  std::vector<std::vector<Association>> distances(previous_centroids_.size());
  for (size_t i = 0; i < previous_centroids_.size(); ++i) {
    distances[i].reserve(centroids.size());
    for (size_t j = 0; j < centroids.size(); ++j) {
      Association a;
      a.distance = (previous_centroids_[i] - centroids[j]).norm();
      a.previous_id = static_cast<int>(i);
      a.current_id = static_cast<int>(j);
      distances[i].push_back(a);
    }
  }

  // Greedy closest association.
  std::unordered_set<int> reused_ids;
  while (true) {
    float min_dist = std::numeric_limits<float>::max();
    int prev_id = 0, curr_id = 0;
    size_t erase_i = 0, erase_j = 0;

    for (size_t i = 0; i < distances.size(); ++i) {
      for (size_t j = 0; j < distances[i].size(); ++j) {
        if (distances[i][j].distance < min_dist) {
          min_dist = distances[i][j].distance;
          curr_id = distances[i][j].current_id;
          prev_id = distances[i][j].previous_id;
          erase_i = i;
          erase_j = j;
        }
      }
    }

    if (min_dist > config_.max_tracking_distance) break;

    clusters[curr_id].id = previous_ids_[prev_id];
    clusters[curr_id].track_length = previous_track_lengths_[prev_id] + 1;
    reused_ids.insert(previous_ids_[prev_id]);

    distances.erase(distances.begin() + erase_i);
    for (auto& vec : distances) {
      vec.erase(vec.begin() + erase_j);
    }
  }

  // Assign new IDs to unmatched clusters.
  previous_centroids_ = centroids;
  previous_ids_.clear();
  previous_ids_.reserve(clusters.size());
  previous_track_lengths_.clear();
  previous_track_lengths_.reserve(clusters.size());

  int id_counter = 0;
  for (Cluster& cluster : clusters) {
    if (cluster.id == -1) {
      while (reused_ids.count(id_counter) > 0) {
        ++id_counter;
      }
      cluster.id = id_counter;
      ++id_counter;
    }
    previous_ids_.push_back(cluster.id);
    previous_track_lengths_.push_back(cluster.track_length);
  }
}

}  // namespace dynamic
}  // namespace voxblox
