#pragma once

// Pose clustering: groups similar poses (pose features, see pose_match.h)
// with k-means, to find the most common poses in a set of clips.

#include <cstdint>
#include <vector>

#include "ozz/base/maths/vec_float.h"

namespace hyperpose {

struct PoseFeatures;

// Poses to cluster: per sample, the joint positions and velocities of a pose
// feature frame (root relative, heading aligned).
struct PoseSamples {
  int num_joints = 0;
  std::vector<ozz::math::Float3> positions;   // [sample * num_joints + joint] (m).
  std::vector<ozz::math::Float3> velocities;  // Same layout (m/s).

  int size() const {
    return num_joints ? static_cast<int>(positions.size()) / num_joints : 0;
  }
  const ozz::math::Float3* position(int _sample) const {
    return &positions[static_cast<size_t>(_sample) * num_joints];
  }
  const ozz::math::Float3* velocity(int _sample) const {
    return &velocities[static_cast<size_t>(_sample) * num_joints];
  }
  // Appends frame _frame of _features (sets num_joints on the first one).
  void Add(const PoseFeatures& _features, int _frame);
};

struct PoseCluster {
  int medoid = -1;          // Sample closest to the cluster center.
  float spread = 0.f;       // RMS of member distances to the center (m).
  std::vector<int> members; // Sample indices, closest to the center first.
  std::vector<float> distances;  // Per member: distance to the center (m).
};

// Clusters _samples into _k clusters with k-means (k-means++ seeding). Poses
// are compared by their joint positions, plus their joint velocities times
// _velocity_weight (seconds: a velocity difference of 1 m/s counts as that
// many meters; 0 for positions only). Distances are per joint: the root of
// the mean over joints of the squared differences. Returns non-empty
// clusters, largest first.
std::vector<PoseCluster> ClusterPoses(const PoseSamples& _samples,
                                      float _velocity_weight, int _k,
                                      uint32_t _seed, int _max_iterations = 50);

}  // namespace hyperpose
