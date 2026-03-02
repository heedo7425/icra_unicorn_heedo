#ifndef VOXBLOX_DYNAMIC_NEIGHBORHOOD_SEARCH_H_
#define VOXBLOX_DYNAMIC_NEIGHBORHOOD_SEARCH_H_

#include <functional>

#include "voxblox/core/common.h"
#include "voxblox/utils/neighbor_tools.h"

namespace voxblox {
namespace dynamic {

/**
 * Thin wrapper around voxblox::Neighborhood that selects 6-, 18-, or 26-
 * connected neighbor lookups at construction time, hiding the template
 * parameter behind a runtime choice.
 */
class NeighborhoodSearch {
 public:
  explicit NeighborhoodSearch(int connectivity) {
    if (connectivity == 6) {
      search_ = Neighborhood<Connectivity::kSix>::getFromBlockAndVoxelIndex;
    } else if (connectivity == 18) {
      search_ =
          Neighborhood<Connectivity::kEighteen>::getFromBlockAndVoxelIndex;
    } else {
      // Default to 26-connected.
      search_ =
          Neighborhood<Connectivity::kTwentySix>::getFromBlockAndVoxelIndex;
    }
  }

  AlignedVector<VoxelKey> search(const BlockIndex& block_index,
                                 const VoxelIndex& voxel_index,
                                 size_t voxels_per_side) const {
    AlignedVector<VoxelKey> neighbors;
    search_(block_index, voxel_index, voxels_per_side, &neighbors);
    return neighbors;
  }

 private:
  std::function<void(const BlockIndex&, const VoxelIndex&, size_t,
                     AlignedVector<VoxelKey>*)>
      search_;
};

}  // namespace dynamic
}  // namespace voxblox

#endif  // VOXBLOX_DYNAMIC_NEIGHBORHOOD_SEARCH_H_
