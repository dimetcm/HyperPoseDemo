#include "common/pose_cluster.h"

#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <numeric>
#include <random>

namespace hyperpose {

namespace {

float Distance2(const float* _a, const float* _b, int _dim) {
  float d = 0.f;
  for (int i = 0; i < _dim; ++i) {
    const float x = _a[i] - _b[i];
    d += x * x;
  }
  return d;
}

}  // namespace

std::vector<PoseCluster> ClusterPoses(const std::vector<float>& _data,
                                      int _num_joints, int _k, uint32_t _seed,
                                      int _max_iterations) {
  const int dim = _num_joints * 3;
  const int n = dim > 0 ? static_cast<int>(_data.size() / dim) : 0;
  const int k = std::min(_k, n);
  if (k <= 0) {
    return {};
  }
  auto sample = [&](int _i) { return &_data[static_cast<size_t>(_i) * dim]; };

  // k-means++ seeding: next centers drawn proportionally to squared distance
  // from the nearest existing center.
  std::mt19937 rng(_seed);
  std::vector<float> centers(static_cast<size_t>(k) * dim);
  std::vector<float> nearest(n, std::numeric_limits<float>::max());
  int pick = std::uniform_int_distribution<int>(0, n - 1)(rng);
  for (int c = 0; c < k; ++c) {
    std::copy(sample(pick), sample(pick) + dim, &centers[static_cast<size_t>(c) * dim]);
    for (int i = 0; i < n; ++i) {
      nearest[i] = std::min(nearest[i], Distance2(sample(i), &centers[c * dim], dim));
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
        const float d = Distance2(sample(_i), &centers[static_cast<size_t>(c) * dim], dim);
        if (d < best_d) {
          best_d = d;
          best = c;
        }
      }
      changed[_i] = assignment[_i] != best;
      assignment[_i] = best;
    });

    std::vector<double> sums(static_cast<size_t>(k) * dim, 0.);
    std::vector<int> counts(k, 0);
    for (int i = 0; i < n; ++i) {
      const int c = assignment[i];
      ++counts[c];
      for (int d = 0; d < dim; ++d) {
        sums[static_cast<size_t>(c) * dim + d] += sample(i)[d];
      }
    }
    for (int c = 0; c < k; ++c) {
      if (counts[c] == 0) {
        continue;  // Empty: keeps its previous center.
      }
      for (int d = 0; d < dim; ++d) {
        centers[static_cast<size_t>(c) * dim + d] =
            static_cast<float>(sums[static_cast<size_t>(c) * dim + d] / counts[c]);
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
    distances[i] = Distance2(sample(i), &centers[static_cast<size_t>(assignment[i]) * dim], dim);
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
      cluster.distances.push_back(
          std::sqrt(distances[i] / std::max(_num_joints, 1)));
    }
    // Per joint RMS distance to the center.
    cluster.spread = static_cast<float>(
        std::sqrt(sum / cluster.members.size() / std::max(_num_joints, 1)));
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
