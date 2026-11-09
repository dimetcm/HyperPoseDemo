#pragma once

// Common poses of a set of clips: every frame of the clips is turned into a
// pose feature (root relative, heading aligned joint positions), then grouped
// with k-means (see pose_cluster.h).

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

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
    float distance;  // RMS joint distance to the cluster center (m).
  };
  std::vector<Frame> frames;
  float spread = 0.f;  // RMS of frame distances (m).

  // Where the clips go from this pose: the next common pose along each clip,
  // most frequent first. Bounces (A B A, with B shorter than kMinPoseStay)
  // are skipped, so flicker between neighbor poses doesn't count.
  struct Transition {
    int to;  // Index in CommonPoseResult::poses.
    struct Example {
      int clip;
      float time;  // First frame in the next pose (s).
    };
    std::vector<Example> examples;
  };
  std::vector<Transition> transitions;
  int num_transitions = 0;  // Sum of transition examples.
};

// Shortest stay in a common pose that isn't a bounce (s).
constexpr float kMinPoseStay = .2f;

struct CommonPoseResult {
  std::vector<std::filesystem::path> clips;
  std::vector<CommonPose> poses;  // Most frames first.
  int num_frames = 0;
  double seconds = 0.;
  std::string error;
};

// Samples every entry at _rate (frames per second), and clusters the poses in
// _k common poses. Only reads its arguments, so it can run on a worker thread.
CommonPoseResult FindCommonPoses(std::vector<PoseSetEntry> _entries,
                                 const ozz::animation::Skeleton* _skeleton,
                                 float _rate, int _k, uint32_t _seed);

}  // namespace hyperpose
