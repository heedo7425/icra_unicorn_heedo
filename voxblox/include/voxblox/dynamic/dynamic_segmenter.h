#ifndef VOXBLOX_DYNAMIC_DYNAMIC_SEGMENTER_H_
#define VOXBLOX_DYNAMIC_DYNAMIC_SEGMENTER_H_

#include <memory>
#include <thread>
#include <vector>

#include "voxblox/core/block_hash.h"
#include "voxblox/core/common.h"
#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"
#include "voxblox/dynamic/ever_free_integrator.h"
#include "voxblox/dynamic/neighborhood_search.h"

namespace voxblox {
namespace dynamic {

// ---------------------------------------------------------------------------
// Internal map types
// ---------------------------------------------------------------------------

/// Map: VoxelIndex  →  sorted list of point indices within that voxel.
using VoxelToPointsMap =
    typename AnyIndexHashMapType<AlignedVector<size_t>>::type;

/// Map: BlockIndex  →  VoxelToPointsMap for all voxels in that block.
using BlockToVoxelPointMap =
    typename AnyIndexHashMapType<VoxelToPointsMap>::type;

// ---------------------------------------------------------------------------
// Cluster
// ---------------------------------------------------------------------------
struct Cluster {
  /// Voxel keys that belong to this cluster.
  AlignedVector<VoxelKey> voxel_keys;
  /// Voxel center coordinates (for approximate AABB / merge distance).
  Pointcloud voxel_centers;
  /// Indices into the input Pointcloud for all points in this cluster.
  std::vector<size_t> point_indices;
  /// Axis-aligned bounding box (min / max corners).
  Point aabb_min;
  Point aabb_max;

  /// Tracking fields.
  int id = -1;
  int track_length = 0;
  bool valid = false;

  float extent() const { return (aabb_max - aabb_min).norm(); }
};

// ---------------------------------------------------------------------------
// DynamicSegmenter
// ---------------------------------------------------------------------------

/**
 * Identifies dynamic points in a single lidar scan by combining voxel-level
 * temporal information (ever-free / never-free) with spatial clustering.
 *
 * Usage (one call per incoming scan, BEFORE TSDF integration):
 * @code
 *   DynamicSegmenter segmenter(cfg, tsdf_layer);
 *   std::vector<bool> is_dynamic =
 *       segmenter.segment(cloud_W, frame_counter);
 *   // ... use is_dynamic to filter the cloud before passing to integrator
 *   ever_free_integrator.updateEverFreeVoxels(frame_counter);
 * @endcode
 *
 * Algorithm (mirrors dynablox):
 *  1. Build block → voxel → points mapping for the current scan.
 *  2. Mark points as *ever-free level dynamic* if they fall into an ever-free
 *     voxel (= high-confidence free space that is now occupied → dynamic).
 *  3. Grow voxel-level clusters starting from occupied ever-free voxels.
 *  4. Map voxel clusters back to individual points.
 *  5. Merge nearby clusters and apply size/extent filters.
 *  6. Return a per-point boolean mask.
 */
class DynamicSegmenter {
 public:
  struct Config {
    // ---- Clustering filters ------------------------------------------------
    /// Minimum / maximum number of points for a cluster to be kept.
    int min_cluster_size = 25;
    int max_cluster_size = 2500;

    /// Minimum / maximum AABB diagonal extent [m].
    float min_extent = 0.25f;
    float max_extent = 2.5f;

    /// Voxel connectivity for cluster growing (6, 18, or 26).
    int neighbor_connectivity = 6;

    /// When true, clusters grow two voxel layers instead of one beyond the
    /// ever-free region (more aggressive expansion).
    bool grow_clusters_twice = true;

    /// Clusters whose points are closer than this [m] will be merged.
    float min_cluster_separation = 0.2f;

    /// When false (default), use voxel centers for AABB and merge distance.
    /// When true, use actual point coordinates (exact but slower).
    bool check_cluster_separation_exact = false;

    // ---- Point-map building -------------------------------------------------
    /// Number of worker threads.
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
  };

  DynamicSegmenter(const Config& config,
                   std::shared_ptr<Layer<TsdfVoxel>> tsdf_layer);

  /**
   * Run the full segmentation pipeline on one lidar scan.
   *
   * @param cloud_W        Point cloud expressed in the world / map frame.
   * @param frame_counter  Monotonically increasing scan index (starts at 1).
   * @return               Per-point boolean mask; true  = dynamic.
   */
  std::vector<bool> segment(const Pointcloud& cloud_W, int frame_counter);

  /**
   * Same as segment(), but also returns per-point ever-free flags
   * (true = point falls in an ever-free voxel, i.e. ever_free_level_dynamic).
   */
  std::vector<bool> segment(const Pointcloud& cloud_W, int frame_counter,
                            std::vector<bool>* ever_free_flags);

  /// Access the clusters identified in the last call to segment().
  const std::vector<Cluster>& lastClusters() const { return last_clusters_; }

 private:
  // ---- Internal types ------------------------------------------------------
  using ClusterVoxelIndices = AlignedVector<VoxelKey>;

  // ---- Point-map construction ----------------------------------------------

  /**
   * Map each point in cloud_W to its block index (fast, single-threaded pass).
   */
  AnyIndexHashMapType<AlignedVector<size_t>>::type buildBlockToPointsMap(
      const Pointcloud& cloud_W) const;

  /**
   * For one block: build the voxel→points mapping, record ever-free hits, and
   * update per-voxel frame stamps.  Called in parallel from setUpPointMap.
   */
  void blockwiseBuildPointMap(
      const Pointcloud& cloud_W, const BlockIndex& block_index,
      const AlignedVector<size_t>& points_in_block,
      VoxelToPointsMap& voxel_map,
      AlignedVector<VoxelKey>& occupied_ever_free_keys,
      std::vector<bool>& ever_free_point_flags, int frame_counter) const;

  /**
   * Parallel driver: calls blockwiseBuildPointMap for all blocks and
   * aggregates results.
   */
  void setUpPointMap(const Pointcloud& cloud_W,
                     BlockToVoxelPointMap& point_map,
                     AlignedVector<VoxelKey>& occupied_ever_free_keys,
                     std::vector<bool>& ever_free_point_flags,
                     int frame_counter) const;

  // ---- Clustering ----------------------------------------------------------

  /**
   * Grow one cluster from a seed voxel key using BFS.
   * Only ever-free voxels can propagate the frontier further.
   */
  bool growCluster(const VoxelKey& seed, int frame_counter,
                   ClusterVoxelIndices& result) const;

  /**
   * Run cluster growing for all seed voxels (ever-free + occupied this frame).
   */
  std::vector<ClusterVoxelIndices> voxelClustering(
      const AlignedVector<VoxelKey>& occupied_ever_free_keys,
      int frame_counter) const;

  /**
   * Assign each voxel cluster's points using the pre-built point_map.
   */
  std::vector<Cluster> inducePointClusters(
      const BlockToVoxelPointMap& point_map,
      const std::vector<ClusterVoxelIndices>& voxel_clusters) const;

  /**
   * Compute the axis-aligned bounding box of a cluster from its points.
   */
  void computeAABB(const Pointcloud& cloud_W, Cluster& cluster) const;

  /**
   * Merge clusters whose points are closer than min_cluster_separation.
   */
  void mergeClusters(const Pointcloud& cloud_W,
                     std::vector<Cluster>& clusters) const;

  /**
   * Remove clusters that fail the size or extent filter.
   */
  void applyClusterFilters(std::vector<Cluster>& clusters) const;

  // ---- Members -------------------------------------------------------------
  const Config config_;
  const std::shared_ptr<Layer<TsdfVoxel>> tsdf_layer_;
  const NeighborhoodSearch neighborhood_search_;

  const size_t voxels_per_side_;
  const size_t voxels_per_block_;

  // Retained for external inspection after each call to segment().
  std::vector<Cluster> last_clusters_;
};

}  // namespace dynamic
}  // namespace voxblox

#endif  // VOXBLOX_DYNAMIC_DYNAMIC_SEGMENTER_H_
