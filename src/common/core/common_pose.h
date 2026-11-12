#pragma once

// Common poses of a set of clips: every frame of the clips is turned into a
// pose feature (root relative, heading aligned joint positions, plus weighted
// joint velocities), then grouped with k-means (see pose_cluster.h).

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "common/core/pose_match.h"

namespace ozz::animation {
class Skeleton;
}

namespace hyperpose {

// A clip of a pose set, with the time range to consider (trimming).
struct PoseSetEntry {
  std::filesystem::path path;
  float duration = 0.f;  // Of the whole clip (s), read when it's added.
  float begin = 0.f;
  float end = -1.f;  // Negative: clip end.

  // End of the range used (s): the clip end when end < 0.
  float trim_end() const { return end < 0.f ? duration : std::min(end, duration); }
  // True if only part of the clip is used.
  bool trimmed() const { return begin > 0.f || trim_end() < duration; }
  // Sets the range used, clamped to the clip (an end at the clip end is
  // stored as -1).
  void SetTrim(float _begin, float _end) {
    begin = std::clamp(_begin, 0.f, duration);
    _end = std::clamp(_end, begin, duration);
    end = _end >= duration - 1e-3f ? -1.f : _end;
  }

  // Same clip and range (the duration comes from the clip).
  bool operator==(const PoseSetEntry& _o) const {
    return path == _o.path && begin == _o.begin && end == _o.end;
  }
  bool operator!=(const PoseSetEntry& _o) const { return !(*this == _o); }
};

// A common pose: the pose set frames that look alike, as clip + time, ordered
// by distance to the pose (cluster center), closest (most typical) first.
struct CommonPose {
  struct Frame {
    int clip;        // Index in CommonPoseResult::clips.
    float time;      // Seconds.
    float distance;  // Per joint distance to the cluster center (m, see
                     // FindCommonPoses).
  };
  std::vector<Frame> frames;
  float spread = 0.f;  // RMS of frame distances (m).
};

// A place in a clip where it changes common pose. Every clip's transitions
// start with an entry (from -1, at the clip start) and end with a dead end
// (to -1, at the clip end), so the clip enters `from` at the previous
// transition (see EnteredAt). Bounces (A B A, with B shorter than
// kMinPoseStay) are skipped, so flicker between neighbor poses doesn't count.
struct PoseTransition {
  int clip;    // Index in CommonPoseResult::clips.
  float time;  // First frame in `to`, or the clip end (s).
  int from;    // Index in CommonPoseResult::poses, -1: clip start.
  int to;      // Index in CommonPoseResult::poses, -1: clip end.
};

// The transitions out of a pose into the same pose (-1: the clip ends).
struct PoseEdge {
  int to;
  std::vector<int> transitions;  // Indices in CommonPoseResult::transitions.
};

// Where the clips go from a pose: edges most frequent first, dead end
// included, and the number of transitions out of it.
struct PoseExits {
  std::vector<PoseEdge> edges;
  int count = 0;
  float share(const PoseEdge& _edge) const {
    return count ? static_cast<float>(_edge.transitions.size()) / count : 0.f;
  }
};

// Shortest stay in a common pose that isn't a bounce (s).
constexpr float kMinPoseStay = .2f;

// Settings for finding common poses (saved with projects).
struct CommonPoseSettings {
  static constexpr int kMinClusters = 2;
  static constexpr int kMaxClusters = 64;
  static constexpr int kMaxStride = 10;

  int clusters = 12;  // k-means clusters.
  int stride = 1;     // Clusters every stride-th frame of the clips.
  int seed = 1;       // k-means++ seeding.
  // Joint velocities against positions (seconds: a velocity difference of
  // 1 m/s counts as that many meters; 0: positions only).
  float velocity_weight = .1f;

  // These settings with every value in its range.
  CommonPoseSettings Clamped() const;

  bool operator==(const CommonPoseSettings& _o) const {
    return clusters == _o.clusters && stride == _o.stride && seed == _o.seed &&
           velocity_weight == _o.velocity_weight;
  }
  bool operator!=(const CommonPoseSettings& _o) const { return !(*this == _o); }
};

struct CommonPoseResult {
  std::vector<std::filesystem::path> clips;
  std::vector<CommonPose> poses;  // Most frames first.
  // Every clip's transitions in clip, then time order.
  std::vector<PoseTransition> transitions;
  std::vector<PoseExits> exits;  // Per pose, grouped from transitions.
  // Per clip, pose features of every frame at the clip's own frame rate
  // (see AnimationFrameRate; whole clip, trimming ignored). Clustering uses
  // every _stride-th frame of them, the best match blend all of them.
  std::vector<PoseFeatures> features;
  int num_frames = 0;
  double seconds = 0.;
  std::string error;
};

// When the clip entered the transition's `from` pose (s): the previous
// transition's time (the transition's own for an entry).
float EnteredAt(const CommonPoseResult& _result, int _transition);

// Labels for UI and logs: "#3" for pose 2, or "end" for -1 (a clip end).
std::string PoseName(int _pose);
// The first _max edges out of a pose with their shares: " #3 60% end 20%".
std::string PoseExitsText(const PoseExits& _exits, size_t _max);

// Samples every entry at its own frame rate, and clusters every stride-th
// frame (1: every frame) in common poses (see CommonPoseSettings). Poses are
// compared by joint positions plus, when velocity_weight > 0, joint
// velocities times velocity_weight, so the same pose moving in opposite
// directions can split; distances are the root of the mean per joint squared
// difference. Only reads its arguments, so it can run on a worker thread.
CommonPoseResult FindCommonPoses(std::vector<PoseSetEntry> _entries,
                                 const ozz::animation::Skeleton* _skeleton,
                                 CommonPoseSettings _settings);

}  // namespace hyperpose
