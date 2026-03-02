#ifndef VOXBLOX_CORE_VOXEL_H_
#define VOXBLOX_CORE_VOXEL_H_

#include <cstdint>
#include <string>

#include "voxblox/core/color.h"
#include "voxblox/core/common.h"

namespace voxblox {

struct TsdfVoxel {
  float distance = 0.0f;
  float weight = 0.0f;
  Color color;

  // Dynamic object detection fields (dynablox-style temporal tracking).
  // ever_free: set once a voxel has been consistently free for burn_in_period
  // consecutive frames with no occupied/unknown neighbors.
  bool ever_free = false;
  // dynamic: set during the current scan if this voxel is part of a detected
  // dynamic cluster.
  bool dynamic = false;
  // last_occupied: frame index when the occupancy counter was last incremented.
  // Used by EverFreeIntegrator to check the burn-in criterion.
  int last_occupied = 0;
  // last_lidar_occupied: frame index when a lidar point last fell into this
  // voxel. Used to determine if the voxel is occupied in the current scan.
  int last_lidar_occupied = 0;
  // occ_counter: number of consecutive frames this voxel has been observed as
  // occupied. Once it exceeds counter_to_reset the ever_free flag is cleared.
  int occ_counter = 0;
  // clustering_processed: internal flag reset each frame; prevents a voxel
  // from being added to multiple clusters during the cluster-growing step.
  bool clustering_processed = false;
};

struct EsdfVoxel {
  float distance = 0.0f;

  bool observed = false;
  /**
   * Whether the voxel was copied from the TSDF (false) or created from a pose
   * or some other source (true). This member is not serialized!!!
   */
  bool hallucinated = false;
  bool in_queue = false;
  bool fixed = false;

  /**
   * Relative direction toward parent. If itself, then either uninitialized
   * or in the fixed frontier.
   */
  Eigen::Vector3i parent = Eigen::Vector3i::Zero();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct OccupancyVoxel {
  float probability_log = 0.0f;
  bool observed = false;
};

struct IntensityVoxel {
  float intensity = 0.0f;
  float weight = 0.0f;
};

/// Used for serialization only.
namespace voxel_types {
const std::string kNotSerializable = "not_serializable";
const std::string kTsdf = "tsdf";
const std::string kEsdf = "esdf";
const std::string kOccupancy = "occupancy";
const std::string kIntensity = "intensity";
}  // namespace voxel_types

template <typename Type>
std::string getVoxelType() {
  return voxel_types::kNotSerializable;
}

template <>
inline std::string getVoxelType<TsdfVoxel>() {
  return voxel_types::kTsdf;
}

template <>
inline std::string getVoxelType<EsdfVoxel>() {
  return voxel_types::kEsdf;
}

template <>
inline std::string getVoxelType<OccupancyVoxel>() {
  return voxel_types::kOccupancy;
}

template <>
inline std::string getVoxelType<IntensityVoxel>() {
  return voxel_types::kIntensity;
}

}  // namespace voxblox

#endif  // VOXBLOX_CORE_VOXEL_H_
