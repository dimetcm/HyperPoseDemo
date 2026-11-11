#pragma once

// Common poses of a set of clips: every frame of the clips is turned into a
// pose feature (root relative, heading aligned joint positions, plus weighted
// joint velocities), then grouped with k-means (see pose_cluster.h).

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "common/pose_match.h"

namespace ozz::animation {
class Skeleton;
}

namespace hyperpose {

// A clip of a pose set, with the time range to consider (trimming).
struct PoseSetEntry {
  std::filesystem::path path;
  float duration = 0.f;
  float begin = 0.f;
  float end = -1.f;  // Negative: clip end.
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

// Frame rate of the per clip match features (CommonPoseResult::features),
// for fine pose matching at runtime (see PoseWalk's best match blend).
constexpr float kMatchRate = 30.f;

// Shortest stay in a common pose that isn't a bounce (s).
constexpr float kMinPoseStay = .2f;

struct CommonPoseResult {
  std::vector<std::filesystem::path> clips;
  std::vector<CommonPose> poses;  // Most frames first.
  // Every clip's transitions in clip, then time order.
  std::vector<PoseTransition> transitions;
  std::vector<PoseExits> exits;  // Per pose, grouped from transitions.
  // Per clip, pose features at kMatchRate (whole clip, trimming ignored).
  std::vector<PoseFeatures> features;
  int num_frames = 0;
  double seconds = 0.;
  std::string error;
};

// When the clip entered the transition's `from` pose (s): the previous
// transition's time (the transition's own for an entry).
float EnteredAt(const CommonPoseResult& _result, int _transition);

// Samples every entry at _rate (frames per second), and clusters the poses in
// _k common poses. Poses are compared by joint positions plus, when
// _velocity_weight (seconds) > 0, joint velocities times _velocity_weight, so
// the same pose moving in opposite directions can split; distances are the
// root of the mean per joint squared difference. Only reads its arguments, so
// it can run on a worker thread.
CommonPoseResult FindCommonPoses(std::vector<PoseSetEntry> _entries,
                                 const ozz::animation::Skeleton* _skeleton,
                                 float _rate, int _k, uint32_t _seed,
                                 float _velocity_weight);

}  // namespace hyperpose
