#ifndef VOXBLOX_DYNAMIC_EVER_FREE_INTEGRATOR_H_
#define VOXBLOX_DYNAMIC_EVER_FREE_INTEGRATOR_H_

#include <memory>
#include <thread>

#include "voxblox/core/block_hash.h"
#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"
#include "voxblox/dynamic/neighborhood_search.h"

namespace voxblox {
namespace dynamic {

/**
 * Maintains the ever-free state of TsdfVoxels across successive lidar scans.
 *
 * A voxel is labeled *ever-free* once it has been observed as free space for
 * `burn_in_period` consecutive frames with no occupied or unknown neighbors.
 * A voxel loses the ever-free label once it has been occupied for
 * `counter_to_reset` frames (to account for newly placed static objects).
 *
 * Temporal information flows through two passes executed every frame
 * (after TSDF integration):
 *  1. Remove ever-free from voxels that have been occupied too many times.
 *  2. Grant ever-free to voxels that have been free long enough.
 *
 * These are the "high-confidence free" (ever_free == true) and
 * "low-confidence free" (observed free but not yet ever_free) regions used
 * by the DynamicSegmenter to seed dynamic object detection.
 */
class EverFreeIntegrator {
 public:
  struct Config {
    /// Connectivity for ever-free removal neighborhood (6, 18, or 26).
    int neighbor_connectivity = 18;

    /// Frames of consecutive occupancy before ever-free is revoked.
    int counter_to_reset = 50;

    /// Frames a voxel may be free between two occupied observations without
    /// resetting the occupancy counter (compensates for lidar sparsity).
    int temporal_buffer = 2;

    /// Consecutive free frames a voxel must see before becoming ever-free.
    int burn_in_period = 5;

    /// TSDF distance below which a voxel counts as occupied [m].
    float tsdf_occupancy_threshold = 0.3f;

    /// Number of worker threads (defaults to hardware concurrency).
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
  };

  EverFreeIntegrator(const Config& config,
                     std::shared_ptr<Layer<TsdfVoxel>> tsdf_layer);

  /**
   * Update the ever-free state of all TSDF-updated voxels.
   * Call this AFTER TSDF integration of the current scan.
   *
   * @param frame_counter  Index of the current lidar scan.
   */
  void updateEverFreeVoxels(int frame_counter) const;

  const Config& getConfig() const { return config_; }

  /**
   * Set prior map occupied voxels for instant ever-free initialization.
   * Voxels in this set are treated as static structure (never ever-free).
   * Voxels NOT in this set skip the burn-in period.
   * Pass nullptr to disable (default).
   */
  void setPriorOccupiedVoxels(const LongIndexSet* set) {
    prior_occupied_voxels_ = set;
  }

 private:
  /**
   * First pass: decrement/reset occ_counter and remove ever-free where the
   * counter has reached counter_to_reset.
   *
   * @param block_index    Index of the block to process.
   * @param frame_counter  Current frame index.
   * @param voxels_to_remove  Voxels outside this block that also need clearing.
   * @return True if any voxels were added to voxels_to_remove.
   */
  bool blockWiseUpdateEverFree(
      const BlockIndex& block_index, int frame_counter,
      AlignedVector<VoxelKey>& voxels_to_remove) const;

  /**
   * Second pass: grant ever-free to voxels that have been free long enough and
   * whose entire neighborhood is observed and free.
   */
  void blockWiseMakeEverFree(const BlockIndex& block_index,
                             int frame_counter) const;

  /**
   * Increment the occupancy counter, resetting it if the voxel was not
   * observed within the temporal_buffer window.
   */
  void updateOccupancyCounter(TsdfVoxel& voxel, int frame_counter) const;

  /**
   * Clear ever_free and dynamic from a voxel and its neighborhood.
   * Returns VoxelKeys outside the given block for deferred processing.
   */
  AlignedVector<VoxelKey> removeEverFree(Block<TsdfVoxel>& block,
                                         TsdfVoxel& voxel,
                                         const BlockIndex& block_index,
                                         const VoxelIndex& voxel_index) const;

  const Config config_;
  const std::shared_ptr<Layer<TsdfVoxel>> tsdf_layer_;
  const NeighborhoodSearch neighborhood_search_;

  const float voxel_size_;
  const size_t voxels_per_side_;
  const size_t voxels_per_block_;

  /// Prior map occupied voxels (nullptr = disabled, use burn-in as usual).
  const LongIndexSet* prior_occupied_voxels_ = nullptr;
};

}  // namespace dynamic
}  // namespace voxblox

#endif  // VOXBLOX_DYNAMIC_EVER_FREE_INTEGRATOR_H_
