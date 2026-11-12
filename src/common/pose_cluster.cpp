#include "common/pose_cluster.h"

#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <numeric>
#include <random>

#include "common/pose_match.h"

namespace math = ozz::math;

namespace hyperpose {

void PoseSamples::Add(const PoseFeatures& _features, int _frame) {
  if (num_joints == 0) {
    num_joints = _features.num_joints;
  }
  const size_t first = static_cast<size_t>(_frame) * _features.num_joints;
  positions.insert(positions.end(), _features.positions.begin() + first,
                   _features.positions.begin() + first + num_joints);
  velocities.insert(velocities.end(), _features.velocities.begin() + first,
                    _features.velocities.begin() + first + num_joints);
}

namespace {

// Copies sample _from of _samples over sample _to of _centers.
void CopySample(const PoseSamples& _samples, int _from, PoseSamples* _centers, int _to) {
  const int joints = _samples.num_joints;
  std::copy(_samples.position(_from), _samples.position(_from) + joints,
            _centers->positions.begin() + static_cast<size_t>(_to) * joints);
  std::copy(_samples.velocity(_from), _samples.velocity(_from) + joints,
            _centers->velocities.begin() + static_cast<size_t>(_to) * joints);
}

}  // namespace

std::vector<PoseCluster> ClusterPoses(const PoseSamples& _samples,
                                      float _velocity_weight, int _k,
                                      uint32_t _seed, int _max_iterations) {
  const int n = _samples.size();
  const int k = std::min(_k, n);
  const int joints = _samples.num_joints;
  if (k <= 0) {
    return {};
  }
  // Squared pose distance (see PoseDistanceSqr) of a sample to a center.
  auto distance_sqr = [&](int _sample, const PoseSamples& _centers, int _center) {
    return PoseDistanceSqr(_samples.position(_sample), _samples.velocity(_sample),
                         _centers.position(_center), _centers.velocity(_center), joints,
                         _velocity_weight);
  };

  // Cluster centers: one pose per cluster.
  PoseSamples centers;
  centers.num_joints = joints;
  centers.positions.resize(static_cast<size_t>(k) * joints);
  centers.velocities.resize(static_cast<size_t>(k) * joints);

  // k-means++ seeding: next centers drawn proportionally to squared distance
  // from the nearest existing center.
  std::mt19937 rng(_seed);
  std::vector<float> nearest(n, std::numeric_limits<float>::max());
  int pick = std::uniform_int_distribution<int>(0, n - 1)(rng);
  for (int c = 0; c < k; ++c) {
    CopySample(_samples, pick, &centers, c);
    for (int i = 0; i < n; ++i) {
      nearest[i] = std::min(nearest[i], distance_sqr(i, centers, c));
    }
    std::discrete_distribution<int> next(nearest.begin(), nearest.end());
    pick = next(rng);
  }

  // Lloyd iterations: assign to nearest center (in parallel), then move
  // centers to their members mean.
  std::vector<int> assignment(n, -1);
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  for (int iteration = 0; iteration < _max_iterations; ++iteration) {
    std::vector<int> changed(n, 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int _i) {
      int best = 0;
      float best_d = std::numeric_limits<float>::max();
      for (int c = 0; c < k; ++c) {
        const float d = distance_sqr(_i, centers, c);
        if (d < best_d) {
          best_d = d;
          best = c;
        }
      }
      changed[_i] = assignment[_i] != best;
      assignment[_i] = best;
    });

    // Members mean.
    PoseSamples sums;
    sums.num_joints = joints;
    sums.positions.assign(centers.positions.size(), math::Float3::zero());
    sums.velocities.assign(centers.velocities.size(), math::Float3::zero());
    std::vector<int> counts(k, 0);
    for (int i = 0; i < n; ++i) {
      const int c = assignment[i];
      ++counts[c];
      for (int j = 0; j < joints; ++j) {
        math::Float3& position = sums.positions[static_cast<size_t>(c) * joints + j];
        math::Float3& velocity = sums.velocities[static_cast<size_t>(c) * joints + j];
        position = position + _samples.position(i)[j];
        velocity = velocity + _samples.velocity(i)[j];
      }
    }
    for (int c = 0; c < k; ++c) {
      if (counts[c] == 0) {
        continue;  // Empty: keeps its previous center.
      }
      const float scale = 1.f / counts[c];
      for (int j = 0; j < joints; ++j) {
        const size_t index = static_cast<size_t>(c) * joints + j;
        centers.positions[index] = sums.positions[index] * scale;
        centers.velocities[index] = sums.velocities[index] * scale;
      }
    }
    // Converged when almost no sample changes cluster.
    const int num_changed = std::accumulate(changed.begin(), changed.end(), 0);
    if (num_changed <= n / 1000) {
      break;
    }
  }

  // Clusters: members sorted by distance to center, medoid and spread.
  std::vector<PoseCluster> clusters(k);
  std::vector<float> distances(n);
  for (int i = 0; i < n; ++i) {
    distances[i] = distance_sqr(i, centers, assignment[i]);
    clusters[assignment[i]].members.push_back(i);
  }
  for (PoseCluster& cluster : clusters) {
    if (cluster.members.empty()) {
      continue;
    }
    std::sort(cluster.members.begin(), cluster.members.end(),
              [&](int _a, int _b) { return distances[_a] < distances[_b]; });
    cluster.medoid = cluster.members.front();
    double sum = 0.;
    cluster.distances.reserve(cluster.members.size());
    for (int i : cluster.members) {
      sum += distances[i];
      cluster.distances.push_back(PoseRms(distances[i], joints));
    }
    // RMS of the members' distances to the center.
    cluster.spread = PoseRms(static_cast<float>(sum / cluster.members.size()), joints);
  }
  clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
                                [](const PoseCluster& _c) { return _c.members.empty(); }),
                 clusters.end());
  std::sort(clusters.begin(), clusters.end(),
            [](const PoseCluster& _a, const PoseCluster& _b) {
              return _a.members.size() > _b.members.size();
            });
  return clusters;
}

}  // namespace hyperpose
