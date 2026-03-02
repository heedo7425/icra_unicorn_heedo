#include "voxblox/dynamic/ukf_tracker.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace voxblox {
namespace dynamic {

// ---------------------------------------------------------------------------
// Constructor — precompute UKF weights.
// ---------------------------------------------------------------------------

UkfTracker::UkfTracker(const Config& config) : config_(config) {
  constexpr int n = kStateDim;
  lambda_ = config_.alpha * config_.alpha * (n + config_.kappa) - n;

  wm_.setZero();
  wc_.setZero();
  wm_(0) = lambda_ / (n + lambda_);
  wc_(0) = lambda_ / (n + lambda_) +
            (1.0f - config_.alpha * config_.alpha + config_.beta);
  for (int i = 1; i < kSigmaN; ++i) {
    wm_(i) = 1.0f / (2.0f * (n + lambda_));
    wc_(i) = wm_(i);
  }
}

// ---------------------------------------------------------------------------
// Sigma point generation via Cholesky decomposition.
// ---------------------------------------------------------------------------

void UkfTracker::generateSigmaPoints(const StateVec& x, const StateMat& P,
                                       SigmaPoints& sigma) const {
  constexpr int n = kStateDim;
  const StateMat scaled_P = (n + lambda_) * P;

  // Cholesky: L * L^T = scaled_P.
  Eigen::LLT<StateMat> llt(scaled_P);
  StateMat L;
  if (llt.info() == Eigen::Success) {
    L = llt.matrixL();
  } else {
    // Fallback: reset to diagonal if P is not positive-definite.
    L = scaled_P.diagonal().cwiseMax(1e-4f).cwiseSqrt().asDiagonal();
  }

  sigma.col(0) = x;
  for (int i = 0; i < n; ++i) {
    sigma.col(1 + i) = x + L.col(i);
    sigma.col(1 + n + i) = x - L.col(i);
  }
}

// ---------------------------------------------------------------------------
// Constant-velocity process model.
// ---------------------------------------------------------------------------

static UkfTracker::StateVec processModel(const UkfTracker::StateVec& x,
                                          float dt) {
  UkfTracker::StateVec xp;
  xp(0) = x(0) + x(3) * dt;  // px += vx * dt
  xp(1) = x(1) + x(4) * dt;  // py += vy * dt
  xp(2) = x(2) + x(5) * dt;  // pz += vz * dt
  xp(3) = x(3);               // vx
  xp(4) = x(4);               // vy
  xp(5) = x(5);               // vz
  return xp;
}

// ---------------------------------------------------------------------------
// Measurement model: extract position.
// ---------------------------------------------------------------------------

static UkfTracker::MeasVec measurementModel(const UkfTracker::StateVec& x) {
  return x.head<3>();
}

// ---------------------------------------------------------------------------
// UKF predict step.
// ---------------------------------------------------------------------------

void UkfTracker::predict(Track& t, float dt) const {
  // Generate sigma points.
  SigmaPoints sigma;
  generateSigmaPoints(t.x, t.P, sigma);

  // Propagate through process model.
  SigmaPoints sigma_pred;
  for (int i = 0; i < kSigmaN; ++i) {
    sigma_pred.col(i) = processModel(sigma.col(i), dt);
  }

  // Recover predicted mean.
  StateVec x_pred = StateVec::Zero();
  for (int i = 0; i < kSigmaN; ++i) {
    x_pred += wm_(i) * sigma_pred.col(i);
  }

  // Recover predicted covariance + process noise.
  StateMat P_pred = StateMat::Zero();
  for (int i = 0; i < kSigmaN; ++i) {
    const StateVec d = sigma_pred.col(i) - x_pred;
    P_pred += wc_(i) * d * d.transpose();
  }

  // Process noise Q.
  const float qp = config_.process_noise_pos * config_.process_noise_pos * dt;
  const float qv = config_.process_noise_vel * config_.process_noise_vel * dt;
  StateMat Q = StateMat::Zero();
  Q(0, 0) = qp; Q(1, 1) = qp; Q(2, 2) = qp;
  Q(3, 3) = qv; Q(4, 4) = qv; Q(5, 5) = qv;
  P_pred += Q;

  t.x = x_pred;
  t.P = P_pred;
}

// ---------------------------------------------------------------------------
// UKF update step.
// ---------------------------------------------------------------------------

void UkfTracker::update(Track& t, const MeasVec& z) const {
  // Generate sigma points from predicted state.
  SigmaPoints sigma;
  generateSigmaPoints(t.x, t.P, sigma);

  // Transform sigma points through measurement model.
  Eigen::Matrix<float, kMeasDim, kSigmaN> Z_sigma;
  for (int i = 0; i < kSigmaN; ++i) {
    Z_sigma.col(i) = measurementModel(sigma.col(i));
  }

  // Predicted measurement mean.
  MeasVec z_pred = MeasVec::Zero();
  for (int i = 0; i < kSigmaN; ++i) {
    z_pred += wm_(i) * Z_sigma.col(i);
  }

  // Innovation covariance S.
  const float r = config_.measurement_noise * config_.measurement_noise;
  MeasMat S = MeasMat::Zero();
  for (int i = 0; i < kSigmaN; ++i) {
    const MeasVec dz = Z_sigma.col(i) - z_pred;
    S += wc_(i) * dz * dz.transpose();
  }
  S(0, 0) += r; S(1, 1) += r; S(2, 2) += r;

  // Cross-covariance Pxz.
  CrossMat Pxz = CrossMat::Zero();
  for (int i = 0; i < kSigmaN; ++i) {
    const StateVec dx = sigma.col(i) - t.x;
    const MeasVec dz = Z_sigma.col(i) - z_pred;
    Pxz += wc_(i) * dx * dz.transpose();
  }

  // Kalman gain.
  const CrossMat K = Pxz * S.inverse();

  // State and covariance update.
  t.x += K * (z - z_pred);
  t.P -= K * S * K.transpose();

  // Ensure symmetry.
  t.P = 0.5f * (t.P + t.P.transpose());
}

// ---------------------------------------------------------------------------
// Compute cluster centroid.
// ---------------------------------------------------------------------------

Eigen::Vector3f UkfTracker::computeCentroid(const Pointcloud& cloud_W,
                                             const Cluster& c) const {
  Eigen::Vector3f sum = Eigen::Vector3f::Zero();
  for (size_t idx : c.point_indices) {
    sum += cloud_W[idx].cast<float>();
  }
  return sum / static_cast<float>(c.point_indices.size());
}

// ---------------------------------------------------------------------------
// Main tracking entry point.
// ---------------------------------------------------------------------------

void UkfTracker::track(const Pointcloud& cloud_W,
                        std::vector<Cluster>& clusters, float dt) {
  if (dt <= 0.0f) dt = 0.1f;  // safety fallback

  // 1. Predict all existing tracks.
  for (Track& t : tracks_) {
    predict(t, dt);
  }

  // 2. Compute cluster centroids.
  std::vector<Eigen::Vector3f> centroids(clusters.size());
  for (size_t j = 0; j < clusters.size(); ++j) {
    centroids[j] = computeCentroid(cloud_W, clusters[j]);
  }

  // 3. Greedy association: nearest predicted position within gate.
  const float gate_sq = config_.gate_distance * config_.gate_distance;
  std::vector<bool> track_matched(tracks_.size(), false);
  std::vector<bool> cluster_matched(clusters.size(), false);

  // Build distance pairs and sort by distance.
  struct Pair {
    size_t track_idx;
    size_t cluster_idx;
    float dist_sq;
  };
  std::vector<Pair> pairs;
  pairs.reserve(tracks_.size() * clusters.size());
  for (size_t ti = 0; ti < tracks_.size(); ++ti) {
    const Eigen::Vector3f pred_pos = tracks_[ti].x.head<3>();
    for (size_t ci = 0; ci < clusters.size(); ++ci) {
      const float d2 = (pred_pos - centroids[ci]).squaredNorm();
      if (d2 <= gate_sq) {
        pairs.push_back({ti, ci, d2});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const Pair& a, const Pair& b) { return a.dist_sq < b.dist_sq; });

  // Assign greedily.
  for (const Pair& p : pairs) {
    if (track_matched[p.track_idx] || cluster_matched[p.cluster_idx]) continue;

    track_matched[p.track_idx] = true;
    cluster_matched[p.cluster_idx] = true;

    Track& t = tracks_[p.track_idx];
    update(t, centroids[p.cluster_idx].cast<float>());
    t.hits++;
    t.misses = 0;
    t.total_length++;
    if (t.hits >= config_.min_hits_to_confirm) t.confirmed = true;

    Cluster& c = clusters[p.cluster_idx];
    c.id = t.id;
    c.track_length = t.total_length;
    c.valid = t.confirmed && (t.total_length >= config_.min_track_duration);
  }

  // 4. Unmatched tracks: increment misses.
  for (size_t ti = tracks_.size(); ti-- > 0;) {
    if (track_matched[ti]) continue;
    Track& t = tracks_[ti];
    t.misses++;
    t.hits = 0;
    if (t.misses > config_.max_misses_to_delete) {
      tracks_.erase(tracks_.begin() + ti);
    }
  }

  // 5. Unmatched clusters: create new tracks.
  for (size_t ci = 0; ci < clusters.size(); ++ci) {
    if (cluster_matched[ci]) continue;

    Track t;
    t.id = next_id_++;
    t.x.head<3>() = centroids[ci];
    t.x.tail<3>().setZero();  // zero initial velocity
    t.P = StateMat::Identity();
    t.P(0, 0) = 1.0f; t.P(1, 1) = 1.0f; t.P(2, 2) = 1.0f;
    t.P(3, 3) = 10.0f; t.P(4, 4) = 10.0f; t.P(5, 5) = 10.0f;
    t.hits = 1;
    t.misses = 0;
    t.total_length = 1;
    t.confirmed = (config_.min_hits_to_confirm <= 1);
    tracks_.push_back(t);

    Cluster& c = clusters[ci];
    c.id = t.id;
    c.track_length = 1;
    c.valid = t.confirmed && (1 >= config_.min_track_duration);
  }
}

}  // namespace dynamic
}  // namespace voxblox
