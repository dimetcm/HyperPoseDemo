#include "common/common_pose.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <utility>

#include "common/pose_cluster.h"
#include "common/pose_match.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/skeleton.h"

namespace hyperpose {
namespace {

// A sampled frame. Samples of a clip are consecutive, in time order.
struct Sample {
  int clip;
  float time;
};

// Fills the transitions from the samples' pose labels: along each clip,
// runs of samples in the same pose, then an entry into the first run, a
// transition between consecutive runs and a dead end after the last one.
// Samples near a cluster border flicker (A B A), so a run shorter than
// kMinPoseStay that returns to the previous pose is a bounce, merged into it.
// Short runs between different poses are kept: fast motions pass through
// poses quickly. _ranges: per clip, the time range sampled (trimmed).
void BuildTransitions(const std::vector<Sample>& _samples,
                      const std::vector<int>& _labels,
                      const std::vector<std::pair<float, float>>& _ranges,
                      float _rate, CommonPoseResult* _result) {
  struct Run {
    int pose;
    int first;  // Sample index.
    int length;
  };
  const int min_length = std::max(1, static_cast<int>(std::ceil(kMinPoseStay * _rate)));
  for (size_t begin = 0; begin < _samples.size();) {
    const int clip = _samples[begin].clip;
    size_t end = begin;
    while (end < _samples.size() && _samples[end].clip == clip) {
      ++end;
    }
    // Runs of the clip.
    std::vector<Run> all;
    for (size_t i = begin; i < end;) {
      size_t j = i;
      while (j < end && _labels[j] == _labels[i]) {
        ++j;
      }
      all.push_back({_labels[i], static_cast<int>(i), static_cast<int>(j - i)});
      i = j;
    }
    // Without bounces, and merged when consecutive in a pose.
    std::vector<Run> runs;
    for (size_t r = 0; r < all.size(); ++r) {
      const bool bounce = all[r].length < min_length && !runs.empty() &&
                          r + 1 < all.size() && all[r + 1].pose == runs.back().pose;
      if (bounce || (!runs.empty() && runs.back().pose == all[r].pose)) {
        continue;
      }
      runs.push_back(all[r]);
    }
    auto& transitions = _result->transitions;
    transitions.push_back({clip, _ranges[clip].first, -1, runs.front().pose});
    for (size_t r = 1; r < runs.size(); ++r) {
      transitions.push_back(
          {clip, _samples[runs[r].first].time, runs[r - 1].pose, runs[r].pose});
    }
    transitions.push_back({clip, _ranges[clip].second, runs.back().pose, -1});
    begin = end;
  }

  // Grouped by source, then target pose.
  std::vector<std::map<int, std::vector<int>>> found(_result->poses.size());
  for (size_t t = 0; t < _result->transitions.size(); ++t) {
    const PoseTransition& transition = _result->transitions[t];
    if (transition.from >= 0) {
      found[transition.from][transition.to].push_back(static_cast<int>(t));
    }
  }
  _result->exits.resize(_result->poses.size());
  for (size_t p = 0; p < found.size(); ++p) {
    PoseExits& exits = _result->exits[p];
    for (auto& [to, transitions] : found[p]) {
      exits.count += static_cast<int>(transitions.size());
      exits.edges.push_back({to, std::move(transitions)});
    }
    std::stable_sort(exits.edges.begin(), exits.edges.end(),
                     [](const PoseEdge& _a, const PoseEdge& _b) {
                       return _a.transitions.size() > _b.transitions.size();
                     });
  }
}

}  // namespace

float EnteredAt(const CommonPoseResult& _result, int _transition) {
  const PoseTransition& transition = _result.transitions[_transition];
  if (_transition > 0 && _result.transitions[_transition - 1].clip == transition.clip) {
    return _result.transitions[_transition - 1].time;
  }
  return transition.time;
}

CommonPoseResult FindCommonPoses(std::vector<PoseSetEntry> _entries,
                                 const ozz::animation::Skeleton* _skeleton,
                                 float _rate, int _k, uint32_t _seed) {
  const auto start = std::chrono::steady_clock::now();
  CommonPoseResult result;
  std::vector<Sample> samples;
  std::vector<float> data;
  std::vector<std::pair<float, float>> ranges;  // Per clip, sampled.
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
    result.features.emplace_back();
    if (!ExtractPoseFeatures(animation, *_skeleton, kMatchRate, &result.features.back())) {
      result.error = "Failed to sample " + entry.path.filename().string();
      return result;
    }
    const float end = entry.end < 0.f ? animation.duration()
                                      : std::min(entry.end, animation.duration());
    ranges.push_back({entry.begin, end});
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
  std::vector<int> labels(samples.size(), -1);  // Pose of each sample.
  for (const PoseCluster& cluster :
       ClusterPoses(data, _skeleton->num_joints(), _k, _seed)) {
    CommonPose pose;
    pose.spread = cluster.spread;
    pose.frames.reserve(cluster.members.size());
    for (size_t m = 0; m < cluster.members.size(); ++m) {
      const Sample& sample = samples[cluster.members[m]];
      pose.frames.push_back({sample.clip, sample.time, cluster.distances[m]});
      labels[cluster.members[m]] = static_cast<int>(result.poses.size());
    }
    result.poses.push_back(std::move(pose));
  }
  BuildTransitions(samples, labels, ranges, _rate, &result);
  result.seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  return result;
}

}  // namespace hyperpose
