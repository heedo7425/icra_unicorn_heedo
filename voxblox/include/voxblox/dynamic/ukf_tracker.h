#ifndef VOXBLOX_DYNAMIC_UKF_TRACKER_H_
#define VOXBLOX_DYNAMIC_UKF_TRACKER_H_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "voxblox/core/common.h"
#include "voxblox/dynamic/dynamic_segmenter.h"  // Cluster struct

namespace voxblox {
namespace dynamic {

class UkfTracker {
 public:
  static constexpr int kStateDim = 6;   // [x, y, z, vx, vy, vz]
  static constexpr int kMeasDim = 3;    // [x, y, z]
  static constexpr int kSigmaN = 2 * kStateDim + 1;  // 13

  using StateVec = Eigen::Matrix<float, kStateDim, 1>;
  using StateMat = Eigen::Matrix<float, kStateDim, kStateDim>;
  using MeasVec = Eigen::Matrix<float, kMeasDim, 1>;
  using MeasMat = Eigen::Matrix<float, kMeasDim, kMeasDim>;
  using CrossMat = Eigen::Matrix<float, kStateDim, kMeasDim>;
  using SigmaPoints = Eigen::Matrix<float, kStateDim, kSigmaN>;

  struct Config {
    // UKF tuning
    float alpha = 1e-3f;
    float beta = 2.0f;
    float kappa = 0.0f;
    float process_noise_pos = 0.1f;    // [m] std per step
    float process_noise_vel = 1.0f;    // [m/s] std per step
    float measurement_noise = 0.15f;   // [m] std

    // Track management
    float gate_distance = 2.0f;        // [m] max association distance
    int min_hits_to_confirm = 3;       // consecutive hits to confirm
    int max_misses_to_delete = 5;      // consecutive misses to delete
    int min_track_duration = 0;        // min frames to set valid
  };

  explicit UkfTracker(const Config& config);

  /// Track clusters across frames. Sets cluster.id, .track_length, .valid.
  /// @param dt Time since last call [seconds].
  void track(const Pointcloud& cloud_W, std::vector<Cluster>& clusters,
             float dt);

  const Config& getConfig() const { return config_; }

 private:
  struct Track {
    int id = -1;
    StateVec x = StateVec::Zero();
    StateMat P = StateMat::Identity();
    int hits = 0;
    int misses = 0;
    int total_length = 0;
    bool confirmed = false;
  };

  void predict(Track& t, float dt) const;
  void update(Track& t, const MeasVec& z) const;
  void generateSigmaPoints(const StateVec& x, const StateMat& P,
                            SigmaPoints& sigma) const;

  Eigen::Vector3f computeCentroid(const Pointcloud& cloud_W,
                                   const Cluster& c) const;

  Config config_;
  std::vector<Track> tracks_;
  int next_id_ = 0;

  // Precomputed UKF weights.
  float lambda_;
  Eigen::Matrix<float, kSigmaN, 1> wm_;  // mean weights
  Eigen::Matrix<float, kSigmaN, 1> wc_;  // covariance weights
};

}  // namespace dynamic
}  // namespace voxblox

#endif  // VOXBLOX_DYNAMIC_UKF_TRACKER_H_
