#include "voxblox/dynamic/ever_free_integrator.h"

#include <future>
#include <mutex>
#include <vector>

#include <glog/logging.h>

#include "voxblox/core/block.h"
#include "voxblox/core/common.h"

namespace voxblox {
namespace dynamic {

EverFreeIntegrator::EverFreeIntegrator(
    const Config& config, std::shared_ptr<Layer<TsdfVoxel>> tsdf_layer)
    : config_(config),
      tsdf_layer_(std::move(tsdf_layer)),
      neighborhood_search_(config_.neighbor_connectivity),
      voxel_size_(tsdf_layer_->voxel_size()),
      voxels_per_side_(tsdf_layer_->voxels_per_side()),
      voxels_per_block_(tsdf_layer_->voxels_per_side() *
                        tsdf_layer_->voxels_per_side() *
                        tsdf_layer_->voxels_per_side()) {}

// ---------------------------------------------------------------------------
// Public
// ---------------------------------------------------------------------------

void EverFreeIntegrator::updateEverFreeVoxels(int frame_counter) const {
  // Collect all blocks that were updated during TSDF integration.
  // We reuse the kEsdf update flag as a "needs ever-free update" marker so
  // that we don't have to add another flag to the Block bitset.
  BlockIndexList updated_blocks;
  tsdf_layer_->getAllUpdatedBlocks(Update::kEsdf, &updated_blocks);

  std::vector<BlockIndex> indices(updated_blocks.begin(),
                                  updated_blocks.end());

  // ---------------------------------------------------------------------------
  // Pass 1 — remove ever-free where occupancy counter has reached the limit.
  // ---------------------------------------------------------------------------
  AlignedVector<VoxelKey> voxels_to_remove;
  std::mutex result_mutex;

  std::atomic<size_t> next_index{0};
  auto worker_remove = [&]() {
    AlignedVector<VoxelKey> local_remove;
    while (true) {
      const size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
      if (i >= indices.size()) break;
      blockWiseUpdateEverFree(indices[i], frame_counter, local_remove);
    }
    std::lock_guard<std::mutex> lock(result_mutex);
    voxels_to_remove.insert(voxels_to_remove.end(), local_remove.begin(),
                            local_remove.end());
  };

  {
    std::vector<std::future<void>> threads;
    threads.reserve(config_.num_threads);
    for (int t = 0; t < config_.num_threads; ++t) {
      threads.emplace_back(std::async(std::launch::async, worker_remove));
    }
    for (auto& f : threads) f.get();
  }

  // Clear the voxels that fell outside their block during parallel processing.
  for (const auto& key : voxels_to_remove) {
    typename Block<TsdfVoxel>::Ptr block =
        tsdf_layer_->getBlockPtrByIndex(key.first);
    if (!block) continue;
    TsdfVoxel& v = block->getVoxelByVoxelIndex(key.second);
    v.ever_free = false;
    v.dynamic = false;
  }

  // ---------------------------------------------------------------------------
  // Pass 2 — grant ever-free to voxels that satisfy the burn-in criterion.
  // ---------------------------------------------------------------------------
  next_index.store(0, std::memory_order_relaxed);
  auto worker_label = [&]() {
    while (true) {
      const size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
      if (i >= indices.size()) break;
      blockWiseMakeEverFree(indices[i], frame_counter);
    }
  };

  {
    std::vector<std::future<void>> threads;
    threads.reserve(config_.num_threads);
    for (int t = 0; t < config_.num_threads; ++t) {
      threads.emplace_back(std::async(std::launch::async, worker_label));
    }
    for (auto& f : threads) f.get();
  }
}

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------

bool EverFreeIntegrator::blockWiseUpdateEverFree(
    const BlockIndex& block_index, int frame_counter,
    AlignedVector<VoxelKey>& voxels_to_remove) const {
  typename Block<TsdfVoxel>::Ptr block =
      tsdf_layer_->getBlockPtrByIndex(block_index);
  if (!block) return false;

  const size_t before = voxels_to_remove.size();

  for (size_t idx = 0; idx < voxels_per_block_; ++idx) {
    TsdfVoxel& voxel = block->getVoxelByLinearIndex(idx);

    // Update occupancy counter if the voxel is near a surface (both sides)
    // or was hit by a lidar ray this frame.
    if (std::abs(voxel.distance) < config_.tsdf_occupancy_threshold ||
        voxel.last_lidar_occupied == frame_counter) {
      updateOccupancyCounter(voxel, frame_counter);
    }

    // Clear dynamic label for voxels that were not hit recently.
    if (voxel.last_lidar_occupied <
        frame_counter - config_.temporal_buffer) {
      voxel.dynamic = false;
    }

    // Remove ever-free from voxels that have been consistently occupied.
    if (voxel.occ_counter >= config_.counter_to_reset && voxel.ever_free) {
      const VoxelIndex voxel_index =
          block->computeVoxelIndexFromLinearIndex(idx);
      AlignedVector<VoxelKey> extra =
          removeEverFree(*block, voxel, block_index, voxel_index);
      voxels_to_remove.insert(voxels_to_remove.end(), extra.begin(),
                              extra.end());
    }
  }

  return voxels_to_remove.size() > before;
}

void EverFreeIntegrator::blockWiseMakeEverFree(const BlockIndex& block_index,
                                               int frame_counter) const {
  typename Block<TsdfVoxel>::Ptr block =
      tsdf_layer_->getBlockPtrByIndex(block_index);
  if (!block) return;

  for (size_t idx = 0; idx < voxels_per_block_; ++idx) {
    TsdfVoxel& voxel = block->getVoxelByLinearIndex(idx);

    // Skip: already ever-free or unobserved.
    if (voxel.ever_free || voxel.weight <= 1e-6f) {
      continue;
    }

    const VoxelIndex voxel_index =
        block->computeVoxelIndexFromLinearIndex(idx);

    const bool has_prior =
        prior_occupied_voxels_ && !prior_occupied_voxels_->empty();

    if (has_prior) {
      // Prior map path: occupied in prior → static structure, never ever-free.
      const GlobalIndex gidx = getGlobalVoxelIndexFromBlockAndVoxelIndex(
          block_index, voxel_index, voxels_per_side_);
      if (prior_occupied_voxels_->count(gidx) > 0) {
        continue;
      }
      // Not in prior → skip burn-in, proceed to neighbor check.
    } else {
      // No prior map: use original burn-in check.
      if (voxel.last_occupied > frame_counter - config_.burn_in_period) {
        continue;
      }
    }

    // Check all neighbors: if any are unobserved or blocked the
    // current voxel cannot be labeled ever-free yet.
    const AlignedVector<VoxelKey> neighbors =
        neighborhood_search_.search(block_index, voxel_index, voxels_per_side_);

    bool blocked = false;
    for (const VoxelKey& nkey : neighbors) {
      const Block<TsdfVoxel>* nblock;
      if (nkey.first == block_index) {
        nblock = block.get();
      } else {
        nblock = tsdf_layer_->getBlockPtrByIndex(nkey.first).get();
        if (!nblock) {
          blocked = true;
          break;
        }
      }
      const TsdfVoxel& nv = nblock->getVoxelByVoxelIndex(nkey.second);
      if (nv.weight < 1e-6f) {
        blocked = true;
        break;
      }
      if (has_prior) {
        // Prior map path: neighbor in prior → blocked (static structure).
        const GlobalIndex ngidx = getGlobalVoxelIndexFromBlockAndVoxelIndex(
            nkey.first, nkey.second, voxels_per_side_);
        if (prior_occupied_voxels_->count(ngidx) > 0) {
          blocked = true;
          break;
        }
      } else {
        // No prior: use burn-in check on neighbor.
        if (nv.last_occupied > frame_counter - config_.burn_in_period) {
          blocked = true;
          break;
        }
      }
    }

    if (!blocked) {
      voxel.ever_free = true;
    }
  }

  // Signal that this block's ever-free state is up to date.
  block->updated().reset(Update::kEsdf);
}

void EverFreeIntegrator::updateOccupancyCounter(TsdfVoxel& voxel,
                                                int frame_counter) const {
  if (voxel.last_occupied >= frame_counter - config_.temporal_buffer) {
    ++voxel.occ_counter;
  } else {
    voxel.occ_counter = 1;
  }
  voxel.last_occupied = frame_counter;
}

AlignedVector<VoxelKey> EverFreeIntegrator::removeEverFree(
    Block<TsdfVoxel>& block, TsdfVoxel& voxel, const BlockIndex& block_index,
    const VoxelIndex& voxel_index) const {
  voxel.ever_free = false;
  voxel.dynamic = false;

  const AlignedVector<VoxelKey> neighbors =
      neighborhood_search_.search(block_index, voxel_index, voxels_per_side_);

  AlignedVector<VoxelKey> deferred;
  for (const VoxelKey& nkey : neighbors) {
    if (nkey.first == block_index) {
      // Same block — safe to modify inline.
      TsdfVoxel& nv = block.getVoxelByVoxelIndex(nkey.second);
      nv.ever_free = false;
      nv.dynamic = false;
    } else {
      // Different block — defer to single-threaded cleanup.
      deferred.push_back(nkey);
    }
  }
  return deferred;
}

}  // namespace dynamic
}  // namespace voxblox
