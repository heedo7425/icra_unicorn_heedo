#ifndef VOXBLOX_DYNAMIC_DBSCAN_CLUSTERER_H_
#define VOXBLOX_DYNAMIC_DBSCAN_CLUSTERER_H_

#include <vector>

#include <Eigen/Core>

#include "voxblox/core/common.h"
#include "voxblox/dynamic/dynamic_segmenter.h"  // Cluster struct

namespace voxblox {
namespace dynamic {

class DbscanClusterer {
 public:
  struct Config {
    float eps = 0.5f;              // neighborhood radius [m]
    int min_points = 5;            // min neighbors to be a core point
    int min_cluster_size = 10;     // discard clusters smaller than this
    int max_cluster_size = 2500;   // discard clusters larger than this
    float min_extent = 0.25f;      // AABB diagonal min [m]
    float max_extent = 2.5f;       // AABB diagonal max [m]
  };

  explicit DbscanClusterer(const Config& config) : config_(config) {}

  /// Run DBSCAN on ever-free points and return clusters.
  /// @param cloud_W   Full input pointcloud (world frame).
  /// @param ever_free Per-point boolean: true if the voxel is ever-free.
  /// @return Clusters with point_indices referencing cloud_W.
  std::vector<Cluster> cluster(const Pointcloud& cloud_W,
                                const std::vector<bool>& ever_free) const;

  const Config& getConfig() const { return config_; }

 private:
  Config config_;
};

}  // namespace dynamic
}  // namespace voxblox

#endif  // VOXBLOX_DYNAMIC_DBSCAN_CLUSTERER_H_
