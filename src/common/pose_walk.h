#pragma once

// Random walk over the common pose transition graph: from the current pose,
// picks the next one weighted by how often the clips go there, plays a real
// example of that transition (a clip segment), then picks again from the
// reached pose. Segments are crossfaded, with every clip's root aligned to
// the origin, so the character stays in place.

#include <memory>
#include <random>

#include "common/clip_player.h"
#include "common/common_pose.h"
#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/span.h"

namespace ozz::animation {
class Skeleton;
}

namespace hyperpose {

class PoseWalk {
 public:
  PoseWalk();
  ~PoseWalk();

  // Settings (seconds). Each pose plays as long as the example clip stayed in
  // it, up to max_stay.
  float max_stay = 4.f;
  float blend = .25f;  // Crossfade between segments.

  // Starts walking from _pose (-1: a random one, weighted by frames). _result
  // and _skeleton must outlive the walk (or Stop it before they change).
  bool Start(const CommonPoseResult* _result,
             const ozz::animation::Skeleton* _skeleton, int _heading_axis,
             int _pose);
  void Stop();
  bool active() const { return result_ != nullptr; }

  // Advances by _dt and outputs the pose, aligned to the origin.
  bool Update(float _dt, ozz::span<ozz::math::Float4x4> _models);

  // Played segment: the transition from -> to of a clip, from when it entered
  // `from` (start) until it reaches `to` (time = stop). The next segment
  // starts where its clip entered `to`, crossfading from this clip, which
  // plays on into `to`. A jump (no transition out of `from`: the clips end
  // there) plays a frame of `to` from `time` to `stop`.
  struct Segment {
    int from = -1;
    int to = -1;
    bool jump = false;
    int clip = -1;  // Index in CommonPoseResult::clips.
    float start = 0.f;
    float time = 0.f;
    float stop = 0.f;
  };
  const Segment& segment() const { return segment_; }
  // Playback time in the segment clip.
  float clip_time() const;
  float clip_duration() const;

 private:
  bool Next(int _from);
  bool SampleAligned(ClipPlayer* _clip, ozz::span<ozz::math::SoaTransform> _out);

  const CommonPoseResult* result_ = nullptr;
  const ozz::animation::Skeleton* skeleton_ = nullptr;
  int heading_axis_ = 0;
  Segment segment_;
  // Playing segment clip, and the previous one fading out.
  std::unique_ptr<ClipPlayer> current_, previous_;
  float fade_ = 0.f;  // Time since the segment started (s).
  ozz::vector<ozz::math::SoaTransform> current_locals_, previous_locals_, blended_;
  std::mt19937 rng_;
};

}  // namespace hyperpose
