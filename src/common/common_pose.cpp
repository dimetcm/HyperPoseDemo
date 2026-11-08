#include "common/common_pose.h"

#include <chrono>

#include "common/pose_cluster.h"
#include "common/pose_match.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/skeleton.h"

namespace hyperpose {

CommonPoseResult FindCommonPoses(std::vector<PoseSetEntry> _entries,
                                 const ozz::animation::Skeleton* _skeleton,
                                 float _rate, int _k, uint32_t _seed) {
  const auto start = std::chrono::steady_clock::now();
  CommonPoseResult result;
  struct Sample {
    int clip;
    float time;
  };
  std::vector<Sample> samples;
  std::vector<float> data;
  ozz::animation::Animation animation;
  PoseFeatures features;
  for (size_t e = 0; e < _entries.size(); ++e) {
    const PoseSetEntry& entry = _entries[e];
    result.clips.push_back(entry.path);
    if (!ozz::sample::LoadAnimation(entry.path.string().c_str(), &animation) ||
        animation.num_tracks() != _skeleton->num_joints() ||
        !ExtractPoseFeatures(animation, *_skeleton, _rate, &features)) {
      result.error = "Failed to load " + entry.path.filename().string();
      return result;
    }
    const float end = entry.end < 0.f ? animation.duration() : entry.end;
    for (int frame = 0; frame < features.num_frames; ++frame) {
      const float time = frame / _rate;
      if (time < entry.begin || time > end) {
        continue;
      }
      samples.push_back({static_cast<int>(e), time});
      const ozz::math::Float3* p = &features.positions[frame * features.num_joints];
      for (int j = 0; j < features.num_joints; ++j) {
        data.insert(data.end(), {p[j].x, p[j].y, p[j].z});
      }
    }
  }
  result.num_frames = static_cast<int>(samples.size());
  for (const PoseCluster& cluster :
       ClusterPoses(data, _skeleton->num_joints(), _k, _seed)) {
    CommonPose pose;
    pose.spread = cluster.spread;
    pose.frames.reserve(cluster.members.size());
    for (size_t m = 0; m < cluster.members.size(); ++m) {
      const Sample& sample = samples[cluster.members[m]];
      pose.frames.push_back({sample.clip, sample.time, cluster.distances[m]});
    }
    result.poses.push_back(std::move(pose));
  }
  result.seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  return result;
}

}  // namespace hyperpose
