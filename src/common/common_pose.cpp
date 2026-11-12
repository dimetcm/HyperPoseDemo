#include "common/common_pose.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <utility>

#include "common/anim_utils.h"
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
                      CommonPoseResult* _result) {
  struct Run {
    int pose;
    int first;  // Sample index.
  };
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
      all.push_back({_labels[i], static_cast<int>(i)});
      i = j;
    }
    // Without bounces, and merged when consecutive in a pose.
    std::vector<Run> runs;
    for (size_t r = 0; r < all.size(); ++r) {
      // A run lasts until the next one starts.
      const bool bounce = !runs.empty() && r + 1 < all.size() &&
                          all[r + 1].pose == runs.back().pose &&
                          _samples[all[r + 1].first].time - _samples[all[r].first].time <
                              kMinPoseStay - 1e-4f;
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

CommonPoseSettings CommonPoseSettings::Clamped() const {
  CommonPoseSettings settings = *this;
  settings.clusters = std::clamp(clusters, kMinClusters, kMaxClusters);
  settings.stride = std::clamp(stride, 1, kMaxStride);
  settings.velocity_weight = std::max(velocity_weight, 0.f);
  return settings;
}

std::string PoseName(int _pose) {
  return _pose < 0 ? "end" : "#" + std::to_string(_pose + 1);
}

std::string PoseExitsText(const PoseExits& _exits, size_t _max) {
  std::string text;
  for (size_t e = 0; e < std::min(_exits.edges.size(), _max); ++e) {
    const PoseEdge& edge = _exits.edges[e];
    char item[32];
    std::snprintf(item, sizeof(item), " %s %.0f%%", PoseName(edge.to).c_str(),
                  _exits.share(edge) * 100.f);
    text += item;
  }
  return text;
}

float EnteredAt(const CommonPoseResult& _result, int _transition) {
  const PoseTransition& transition = _result.transitions[_transition];
  if (_transition > 0 && _result.transitions[_transition - 1].clip == transition.clip) {
    return _result.transitions[_transition - 1].time;
  }
  return transition.time;
}

CommonPoseResult FindCommonPoses(std::vector<PoseSetEntry> _entries,
                                 const ozz::animation::Skeleton* _skeleton,
                                 CommonPoseSettings _settings) {
  _settings = _settings.Clamped();
  const auto start = std::chrono::steady_clock::now();
  CommonPoseResult result;
  std::vector<Sample> samples;  // Clip and time of each pose.
  PoseSamples poses;            // Pose features of each sample.
  std::vector<std::pair<float, float>> ranges;  // Per clip, sampled.
  ozz::animation::Animation animation;
  for (size_t e = 0; e < _entries.size(); ++e) {
    const PoseSetEntry& entry = _entries[e];
    result.clips.push_back(entry.path);
    result.features.emplace_back();
    PoseFeatures& features = result.features.back();
    if (!ozz::sample::LoadAnimation(entry.path.string().c_str(), &animation) ||
        animation.num_tracks() != _skeleton->num_joints() ||
        !ExtractPoseFeatures(animation, *_skeleton, AnimationFrameRate(animation),
                             &features)) {
      result.error = "Failed to load " + entry.path.filename().string();
      return result;
    }
    // From the motion start: a leading bind pose is left out.
    const float begin = std::max(entry.begin, features.first_frame / features.frame_rate);
    const float end = entry.end < 0.f ? animation.duration()
                                      : std::min(entry.end, animation.duration());
    ranges.push_back({begin, end});
    for (int frame = features.first_frame; frame < features.num_frames; frame += _settings.stride) {
      const float time = frame / features.frame_rate;
      if (time < begin || time > end) {
        continue;
      }
      samples.push_back({static_cast<int>(e), time});
      poses.Add(features, frame);
    }
  }
  result.num_frames = static_cast<int>(samples.size());
  std::vector<int> labels(samples.size(), -1);  // Pose of each sample.
  const std::vector<PoseCluster> clusters =
      ClusterPoses(poses, _settings.velocity_weight, _settings.clusters,
                   static_cast<uint32_t>(_settings.seed));
  for (const PoseCluster& cluster : clusters) {
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
  BuildTransitions(samples, labels, ranges, &result);
  result.seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  return result;
}

}  // namespace hyperpose
