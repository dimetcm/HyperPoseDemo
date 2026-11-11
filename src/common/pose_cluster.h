#pragma once

// Pose clustering: groups similar poses (pose feature vectors, see
// pose_match.h) with k-means, to find the most common poses in a set of clips.

#include <cstdint>
#include <vector>

namespace hyperpose {

struct PoseCluster {
  int medoid = -1;          // Sample closest to the cluster center.
  float spread = 0.f;       // RMS of member distances to the center (m).
  std::vector<int> members; // Sample indices, closest to the center first.
  std::vector<float> distances;  // Per member: per joint distance to the center (m).
};

// Clusters samples into _k clusters with k-means (k-means++ seeding).
// _data holds _dim floats per sample: joint positions (x, y, z), possibly
// followed by more per joint values (weighted velocities). Distances are
// reported per joint: the root of the squared distance over _num_joints.
// Returns non-empty clusters, largest first.
std::vector<PoseCluster> ClusterPoses(const std::vector<float>& _data, int _dim,
                                      int _num_joints, int _k, uint32_t _seed,
                                      int _max_iterations = 50);

}  // namespace hyperpose
