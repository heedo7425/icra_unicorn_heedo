#ifndef VOXBLOX_DYNAMIC_TRACKING_H_
#define VOXBLOX_DYNAMIC_TRACKING_H_

#include <limits>
#include <unordered_set>
#include <vector>

#include "voxblox/core/common.h"
#include "voxblox/dynamic/dynamic_segmenter.h"

namespace voxblox {
namespace dynamic {

class Tracking {
 public:
  struct Config {
    /// Number of frames a cluster needs to be tracked to be considered dynamic.
    int min_track_duration = 0;
    /// Maximum distance a cluster may have moved to be considered a track [m].
    float max_tracking_distance = 1.0f;
  };

  explicit Tracking(const Config& config) : config_(config) {}

  /**
   * Track clusters across frames by centroid association.
   * Sets cluster.id, cluster.track_length, and cluster.valid.
   */
  void track(const Pointcloud& cloud_W, std::vector<Cluster>& clusters);

 private:
  void trackClusterIDs(const Pointcloud& cloud_W,
                       std::vector<Cluster>& clusters);

  const Config config_;
  std::vector<Point> previous_centroids_;
  std::vector<int> previous_ids_;
  std::vector<int> previous_track_lengths_;
};

}  // namespace dynamic
}  // namespace voxblox

#endif  // VOXBLOX_DYNAMIC_TRACKING_H_
