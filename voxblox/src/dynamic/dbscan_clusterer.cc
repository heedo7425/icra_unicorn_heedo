#include "voxblox/dynamic/dbscan_clusterer.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "voxblox/dynamic/nanoflann.hpp"

namespace nanoflann = kiss_matcher;

namespace voxblox {
namespace dynamic {

// ---------------------------------------------------------------------------
// nanoflann adaptor for an Eigen MatrixXf (N x 3).
// ---------------------------------------------------------------------------
struct EigenMatrixAdaptor {
  const Eigen::MatrixXf& pts;
  explicit EigenMatrixAdaptor(const Eigen::MatrixXf& p) : pts(p) {}
  size_t kdtree_get_point_count() const {
    return static_cast<size_t>(pts.rows());
  }
  float kdtree_get_pt(size_t idx, size_t dim) const {
    return pts(static_cast<Eigen::Index>(idx), static_cast<Eigen::Index>(dim));
  }
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const {
    return false;
  }
};

using KDTree3f = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, EigenMatrixAdaptor>,
    EigenMatrixAdaptor, 3>;

// radiusSearch result type for this nanoflann version.
using MatchPair = std::pair<unsigned int, float>;  // (index, squared distance)

// ---------------------------------------------------------------------------
// DBSCAN implementation
// ---------------------------------------------------------------------------

static constexpr int UNVISITED = -1;
static constexpr int NOISE = -2;

std::vector<Cluster> DbscanClusterer::cluster(
    const Pointcloud& cloud_W,
    const std::vector<bool>& ever_free) const {
  // 1. Extract ever-free point indices.
  std::vector<size_t> ef_indices;
  ef_indices.reserve(cloud_W.size());
  for (size_t i = 0; i < cloud_W.size(); ++i) {
    if (i < ever_free.size() && ever_free[i]) {
      ef_indices.push_back(i);
    }
  }

  if (static_cast<int>(ef_indices.size()) < config_.min_points) {
    return {};
  }

  // 2. Build Eigen matrix of ever-free points.
  const size_t n = ef_indices.size();
  Eigen::MatrixXf pts(n, 3);
  for (size_t j = 0; j < n; ++j) {
    pts(j, 0) = cloud_W[ef_indices[j]].x();
    pts(j, 1) = cloud_W[ef_indices[j]].y();
    pts(j, 2) = cloud_W[ef_indices[j]].z();
  }

  // 3. Build KD-tree.
  EigenMatrixAdaptor adaptor(pts);
  KDTree3f kd_tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  kd_tree.buildIndex();

  const float eps_sq = config_.eps * config_.eps;
  const nanoflann::SearchParams search_params;

  // 4. DBSCAN core loop.
  std::vector<int> labels(n, UNVISITED);
  int cluster_id = 0;

  std::vector<MatchPair> matches;

  for (size_t j = 0; j < n; ++j) {
    if (labels[j] != UNVISITED) continue;

    // Range query for point j.
    const float query[3] = {pts(j, 0), pts(j, 1), pts(j, 2)};
    matches.clear();
    kd_tree.radiusSearch(query, eps_sq, matches, search_params);

    if (static_cast<int>(matches.size()) < config_.min_points) {
      labels[j] = NOISE;
      continue;
    }

    // Start new cluster.
    labels[j] = cluster_id;

    // Seed queue: indices into the ef array.
    std::vector<size_t> seed_queue;
    seed_queue.reserve(matches.size());
    for (const auto& m : matches) {
      if (m.first != j) seed_queue.push_back(m.first);
    }

    size_t k = 0;
    while (k < seed_queue.size()) {
      const size_t q = seed_queue[k++];

      if (labels[q] == NOISE) {
        labels[q] = cluster_id;  // border point
        continue;
      }
      if (labels[q] != UNVISITED) continue;

      labels[q] = cluster_id;

      // Range query for point q.
      const float qpt[3] = {pts(q, 0), pts(q, 1), pts(q, 2)};
      std::vector<MatchPair> q_matches;
      kd_tree.radiusSearch(qpt, eps_sq, q_matches, search_params);

      if (static_cast<int>(q_matches.size()) >= config_.min_points) {
        for (const auto& m : q_matches) {
          if (labels[m.first] == UNVISITED || labels[m.first] == NOISE) {
            seed_queue.push_back(m.first);
          }
        }
      }
    }
    ++cluster_id;
  }

  // 5. Build Cluster structs from labels.
  std::vector<Cluster> clusters(cluster_id);
  for (size_t j = 0; j < n; ++j) {
    if (labels[j] < 0) continue;  // noise
    clusters[labels[j]].point_indices.push_back(ef_indices[j]);
  }

  // 6. Compute AABBs.
  for (Cluster& c : clusters) {
    if (c.point_indices.empty()) continue;
    c.aabb_min = Point(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max());
    c.aabb_max = Point(std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest());
    for (size_t idx : c.point_indices) {
      c.aabb_min = c.aabb_min.cwiseMin(cloud_W[idx]);
      c.aabb_max = c.aabb_max.cwiseMax(cloud_W[idx]);
    }
  }

  // 7. Filter by size / extent.
  clusters.erase(
      std::remove_if(clusters.begin(), clusters.end(),
                     [this](const Cluster& c) {
                       const int sz =
                           static_cast<int>(c.point_indices.size());
                       if (sz < config_.min_cluster_size ||
                           sz > config_.max_cluster_size)
                         return true;
                       const float ext = c.extent();
                       return ext < config_.min_extent ||
                              ext > config_.max_extent;
                     }),
      clusters.end());

  return clusters;
}

}  // namespace dynamic
}  // namespace voxblox
