#include "voxblox/dynamic/dynamic_segmenter.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <limits>
#include <mutex>
#include <vector>

#include <glog/logging.h>

#include "voxblox/core/block.h"
#include "voxblox/utils/voxel_utils.h"

namespace voxblox {
namespace dynamic {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

DynamicSegmenter::DynamicSegmenter(const Config& config,
                                   std::shared_ptr<Layer<TsdfVoxel>> tsdf_layer)
    : config_(config),
      tsdf_layer_(std::move(tsdf_layer)),
      neighborhood_search_(config_.neighbor_connectivity),
      voxels_per_side_(tsdf_layer_->voxels_per_side()),
      voxels_per_block_(tsdf_layer_->voxels_per_side() *
                        tsdf_layer_->voxels_per_side() *
                        tsdf_layer_->voxels_per_side()) {}

// ---------------------------------------------------------------------------
// Public: segment()
// ---------------------------------------------------------------------------

std::vector<bool> DynamicSegmenter::segment(const Pointcloud& cloud_W,
                                            int frame_counter) {
  return segment(cloud_W, frame_counter, nullptr);
}

std::vector<bool> DynamicSegmenter::segment(const Pointcloud& cloud_W,
                                            int frame_counter,
                                            std::vector<bool>* out_ever_free) {
  const size_t n_pts = cloud_W.size();
  std::vector<bool> is_dynamic(n_pts, false);
  if (n_pts == 0) return is_dynamic;

  // 1. Build point-map and collect ever-free occupied voxels.
  BlockToVoxelPointMap point_map;
  AlignedVector<VoxelKey> occupied_ever_free_keys;
  std::vector<bool> ever_free_flags(n_pts, false);

  setUpPointMap(cloud_W, point_map, occupied_ever_free_keys, ever_free_flags,
                frame_counter);

  // 2. Voxel-level cluster growing from occupied ever-free seeds.
  const std::vector<ClusterVoxelIndices> voxel_clusters =
      voxelClustering(occupied_ever_free_keys, frame_counter);

  // 3. Map voxel clusters → point clusters.
  std::vector<Cluster> clusters =
      inducePointClusters(point_map, voxel_clusters);

  // 4. Compute AABBs.
  for (Cluster& c : clusters) computeAABB(cloud_W, c);

  // 5. Merge nearby clusters.
  mergeClusters(cloud_W, clusters);

  // 6. Filter by size / extent.
  applyClusterFilters(clusters);

  // 7. Label only points belonging to surviving clusters as dynamic
  //    (cluster_level_dynamic — the actual usable output).
  for (const Cluster& c : clusters) {
    for (size_t idx : c.point_indices) {
      is_dynamic[idx] = true;
    }
  }

  // Optionally return ever-free level flags (for evaluation).
  if (out_ever_free) {
    *out_ever_free = std::move(ever_free_flags);
  }

  last_clusters_ = std::move(clusters);
  return is_dynamic;
}

// ---------------------------------------------------------------------------
// Point-map construction
// ---------------------------------------------------------------------------

AnyIndexHashMapType<AlignedVector<size_t>>::type
DynamicSegmenter::buildBlockToPointsMap(const Pointcloud& cloud_W) const {
  AnyIndexHashMapType<AlignedVector<size_t>>::type result;
  for (size_t i = 0; i < cloud_W.size(); ++i) {
    const BlockIndex bidx =
        tsdf_layer_->computeBlockIndexFromCoordinates(cloud_W[i]);
    result[bidx].push_back(i);
  }
  return result;
}

void DynamicSegmenter::blockwiseBuildPointMap(
    const Pointcloud& cloud_W, const BlockIndex& block_index,
    const AlignedVector<size_t>& points_in_block,
    VoxelToPointsMap& voxel_map,
    AlignedVector<VoxelKey>& occupied_ever_free_keys,
    std::vector<bool>& ever_free_point_flags, int frame_counter) const {
  typename Block<TsdfVoxel>::Ptr block =
      tsdf_layer_->getBlockPtrByIndex(block_index);
  if (!block) return;

  // Map each point to the voxel it falls into.
  for (size_t i : points_in_block) {
    const VoxelIndex vidx =
        block->computeVoxelIndexFromCoordinates(cloud_W[i]);
    if (!block->isValidVoxelIndex(vidx)) continue;

    voxel_map[vidx].push_back(i);

    // Check if this voxel is ever-free (high-confidence free space).
    if (block->getVoxelByVoxelIndex(vidx).ever_free) {
      ever_free_point_flags[i] = true;
    }
  }

  // Update per-voxel frame stamps for all occupied voxels.
  for (auto& [vidx, pt_indices] : voxel_map) {
    TsdfVoxel& voxel = block->getVoxelByVoxelIndex(vidx);
    voxel.last_lidar_occupied = frame_counter;
    // Reset clustering flag so this voxel can be added to a cluster.
    voxel.clustering_processed = false;

    if (voxel.ever_free) {
      occupied_ever_free_keys.emplace_back(block_index, vidx);
    }
  }
}

void DynamicSegmenter::setUpPointMap(
    const Pointcloud& cloud_W, BlockToVoxelPointMap& point_map,
    AlignedVector<VoxelKey>& occupied_ever_free_keys,
    std::vector<bool>& ever_free_point_flags, int frame_counter) const {
  const auto block2pts = buildBlockToPointsMap(cloud_W);

  std::vector<BlockIndex> block_indices;
  block_indices.reserve(block2pts.size());
  for (const auto& [bidx, _] : block2pts) block_indices.push_back(bidx);

  std::atomic<size_t> next_block{0};
  std::mutex result_mutex;

  auto worker = [&]() {
    AlignedVector<VoxelKey> local_ever_free;
    BlockToVoxelPointMap local_map;

    while (true) {
      const size_t i = next_block.fetch_add(1, std::memory_order_relaxed);
      if (i >= block_indices.size()) break;

      const BlockIndex& bidx = block_indices[i];
      VoxelToPointsMap voxel_map;
      blockwiseBuildPointMap(cloud_W, bidx, block2pts.at(bidx), voxel_map,
                             local_ever_free, ever_free_point_flags,
                             frame_counter);
      local_map.emplace(bidx, std::move(voxel_map));
    }

    std::lock_guard<std::mutex> lock(result_mutex);
    occupied_ever_free_keys.insert(occupied_ever_free_keys.end(),
                                   local_ever_free.begin(),
                                   local_ever_free.end());
    point_map.merge(local_map);
  };

  std::vector<std::future<void>> threads;
  threads.reserve(config_.num_threads);
  for (int t = 0; t < config_.num_threads; ++t) {
    threads.emplace_back(std::async(std::launch::async, worker));
  }
  for (auto& f : threads) f.get();
}

// ---------------------------------------------------------------------------
// Clustering
// ---------------------------------------------------------------------------

bool DynamicSegmenter::growCluster(const VoxelKey& seed, int frame_counter,
                                   ClusterVoxelIndices& result) const {
  AlignedVector<VoxelKey> stack = {seed};

  while (!stack.empty()) {
    const VoxelKey key = stack.back();
    stack.pop_back();

    typename Block<TsdfVoxel>::Ptr block =
        tsdf_layer_->getBlockPtrByIndex(key.first);
    if (!block) continue;

    TsdfVoxel& voxel = block->getVoxelByVoxelIndex(key.second);
    if (voxel.clustering_processed) continue;

    // Accept this voxel into the cluster.
    voxel.dynamic = true;
    voxel.clustering_processed = true;
    result.push_back(key);

    // Expand to neighbors.
    const AlignedVector<VoxelKey> neighbors =
        neighborhood_search_.search(key.first, key.second, voxels_per_side_);

    for (const VoxelKey& nkey : neighbors) {
      typename Block<TsdfVoxel>::Ptr nblock =
          tsdf_layer_->getBlockPtrByIndex(nkey.first);
      if (!nblock) continue;

      TsdfVoxel& nv = nblock->getVoxelByVoxelIndex(nkey.second);
      if (nv.clustering_processed ||
          nv.last_lidar_occupied != frame_counter) {
        continue;
      }

      // Dynablox growth rules:
      // - ever-free neighbor: push to stack (continue growing)
      // - non-ever-free neighbor of ever-free voxel with grow_twice:
      //   push to stack (one extra propagation layer)
      // - otherwise: absorb as leaf node (add to cluster, don't propagate)
      if (nv.ever_free) {
        stack.push_back(nkey);
      } else if (voxel.ever_free && config_.grow_clusters_twice) {
        stack.push_back(nkey);
      } else {
        // Leaf node: absorb into cluster but don't propagate further.
        nv.dynamic = true;
        nv.clustering_processed = true;
        result.push_back(nkey);
      }
    }
  }
  return !result.empty();
}

std::vector<DynamicSegmenter::ClusterVoxelIndices>
DynamicSegmenter::voxelClustering(
    const AlignedVector<VoxelKey>& occupied_ever_free_keys,
    int frame_counter) const {
  std::vector<ClusterVoxelIndices> clusters;
  for (const VoxelKey& seed : occupied_ever_free_keys) {
    ClusterVoxelIndices cluster;
    if (growCluster(seed, frame_counter, cluster)) {
      clusters.push_back(std::move(cluster));
    }
  }
  return clusters;
}

std::vector<Cluster> DynamicSegmenter::inducePointClusters(
    const BlockToVoxelPointMap& point_map,
    const std::vector<ClusterVoxelIndices>& voxel_clusters) const {
  std::vector<Cluster> result;
  result.reserve(voxel_clusters.size());

  const float voxel_size = tsdf_layer_->voxel_size();

  for (const ClusterVoxelIndices& vkeys : voxel_clusters) {
    Cluster c;
    for (const VoxelKey& key : vkeys) {
      c.voxel_keys.push_back(key);

      // Compute voxel center for approximate AABB/merge.
      const GlobalIndex global_voxel_idx =
          getGlobalVoxelIndexFromBlockAndVoxelIndex(
              key.first, key.second, voxels_per_side_);
      const Point center =
          getCenterPointFromGridIndex(global_voxel_idx, voxel_size);
      c.voxel_centers.push_back(center);

      auto bit = point_map.find(key.first);
      if (bit == point_map.end()) continue;

      auto vit = bit->second.find(key.second);
      if (vit == bit->second.end()) continue;

      for (size_t pidx : vit->second) c.point_indices.push_back(pidx);
    }
    result.push_back(std::move(c));
  }
  return result;
}

void DynamicSegmenter::computeAABB(const Pointcloud& cloud_W,
                                   Cluster& cluster) const {
  if (cluster.point_indices.empty()) return;

  cluster.aabb_min = Point(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max());
  cluster.aabb_max = Point(std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest());

  if (config_.check_cluster_separation_exact) {
    // Exact AABB from points.
    for (size_t idx : cluster.point_indices) {
      cluster.aabb_min = cluster.aabb_min.cwiseMin(cloud_W[idx]);
      cluster.aabb_max = cluster.aabb_max.cwiseMax(cloud_W[idx]);
    }
  } else {
    // Approximate AABB from voxel centers +/- half voxel.
    const float vs = tsdf_layer_->voxel_size();
    for (const Point& vc : cluster.voxel_centers) {
      cluster.aabb_min = cluster.aabb_min.cwiseMin(vc);
      cluster.aabb_max = cluster.aabb_max.cwiseMax(vc);
    }
    cluster.aabb_min.array() -= 0.5f * vs;
    cluster.aabb_max.array() += 0.5f * vs;
  }
}

void DynamicSegmenter::mergeClusters(const Pointcloud& cloud_W,
                                     std::vector<Cluster>& clusters) const {
  if (config_.min_cluster_separation <= 0.f || clusters.size() < 2u) return;

  const float sep = config_.min_cluster_separation;
  size_t i = 0;
  while (i < clusters.size()) {
    size_t j = i + 1;
    while (j < clusters.size()) {
      Cluster& ci = clusters[i];
      Cluster& cj = clusters[j];

      // Quick AABB check: skip if clearly separate.
      bool aabb_overlap = true;
      for (int d = 0; d < 3; ++d) {
        if (ci.aabb_min[d] - sep > cj.aabb_max[d] ||
            cj.aabb_min[d] - sep > ci.aabb_max[d]) {
          aabb_overlap = false;
          break;
        }
      }
      if (!aabb_overlap) { ++j; continue; }

      // Minimum distance check (exact: point-to-point, approx: voxel-to-voxel).
      bool should_merge = false;
      if (config_.check_cluster_separation_exact) {
        for (size_t pi : ci.point_indices) {
          for (size_t pj : cj.point_indices) {
            if ((cloud_W[pi] - cloud_W[pj]).norm() <= sep) {
              should_merge = true;
              break;
            }
          }
          if (should_merge) break;
        }
      } else {
        for (const Point& vi : ci.voxel_centers) {
          for (const Point& vj : cj.voxel_centers) {
            if ((vi - vj).norm() <= sep) {
              should_merge = true;
              break;
            }
          }
          if (should_merge) break;
        }
      }

      if (should_merge) {
        ci.point_indices.insert(ci.point_indices.end(),
                                cj.point_indices.begin(),
                                cj.point_indices.end());
        ci.voxel_keys.insert(ci.voxel_keys.end(), cj.voxel_keys.begin(),
                             cj.voxel_keys.end());
        ci.voxel_centers.insert(ci.voxel_centers.end(),
                                cj.voxel_centers.begin(),
                                cj.voxel_centers.end());
        clusters.erase(clusters.begin() + j);
        computeAABB(cloud_W, ci);
      } else {
        ++j;
      }
    }
    ++i;
  }
}

void DynamicSegmenter::applyClusterFilters(
    std::vector<Cluster>& clusters) const {
  clusters.erase(
      std::remove_if(clusters.begin(), clusters.end(),
                     [this](const Cluster& c) {
                       const int sz = static_cast<int>(c.point_indices.size());
                       if (sz < config_.min_cluster_size ||
                           sz > config_.max_cluster_size)
                         return true;
                       const float ext = c.extent();
                       return ext < config_.min_extent ||
                              ext > config_.max_extent;
                     }),
      clusters.end());
}

}  // namespace dynamic
}  // namespace voxblox
