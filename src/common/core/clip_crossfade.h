#pragma once

// Clips crossfaded on one skeleton: each clip played fades in over the ones
// already playing. Tracks are stacked, oldest first, so a clip shorter than
// the blend doesn't cut a fade short; tracks under a fully faded in one are
// dropped (pooled, with their clip kept loaded). With root motion, each clip
// starts where the character stands and moves as recorded; without, the root
// stays above the origin, facing the default heading.

#include <filesystem>
#include <memory>
#include <vector>

#include "common/core/clip_player.h"
#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/span.h"

namespace ozz::animation {
class Skeleton;
}

namespace hyperpose {

class ClipCrossfade {
 public:
  ClipCrossfade();
  ~ClipCrossfade();

  // Settings.
  float blend = .25f;        // Fade in time of each clip (s).
  bool root_motion = false;  // Switching it on continues from where the
                             // character stands.

  // Plays on _skeleton (which must outlive the crossfade), with root headings
  // measured on _heading_axis (see HeadingAxis). Clears everything, and puts
  // the character back at the origin.
  void Reset(const ozz::animation::Skeleton* _skeleton, int _heading_axis);
  // Stops every clip, unloading them (they may not suit the next skeleton).
  void Clear();
  bool empty() const { return tracks_.empty(); }

  // Plays _clip from _time, fading in over the playing clips, its root placed
  // where the character stands.
  bool Play(const std::filesystem::path& _clip, float _time);
  // The newest clip (the one fading in or played), when not empty.
  const ClipPlayer& current() const { return tracks_.back()->clip; }

  // Moves the character back to the origin, facing the default heading.
  void ResetPosition();

  // Advances every clip by _dt.
  void Advance(float _dt);
  // Blends the clips into _models (model space) and keeps where the
  // character root stands, for the next clips.
  bool Sample(ozz::span<ozz::math::Float4x4> _models);

 private:
  // Root motion placement of a clip: its root is rotated by yaw about Y,
  // then shifted on the ground.
  struct RootOffset {
    float yaw = 0.f;
    ozz::math::Float3 shift = ozz::math::Float3::zero();
  };

  // A clip fading in over the older ones.
  struct Track {
    ClipPlayer clip;
    RootOffset offset;
    float age = 0.f;  // Since it started (s).
    ozz::vector<ozz::math::SoaTransform> locals;
  };

  // Samples the track's clip at its time into its locals, with the root
  // aligned to the origin, or placed by its offset with root motion.
  bool SampleRoot(Track* _track);
  // Sets the track's offset so that its root at its time lands on the
  // character root (root_position_, root_heading_).
  void Anchor(Track* _track);

  const ozz::animation::Skeleton* skeleton_ = nullptr;
  int heading_axis_ = 0;
  std::vector<std::unique_ptr<Track>> tracks_;  // Oldest first.
  std::vector<std::unique_ptr<Track>> pool_;    // Unused, clip kept loaded.
  // Character root on the ground (last output), and the root motion setting
  // the offsets were set for.
  ozz::math::Float3 root_position_ = ozz::math::Float3::zero();
  float root_heading_ = 0.f;
  bool anchored_motion_ = false;
  ozz::vector<ozz::math::SoaTransform> blended_;
};

}  // namespace hyperpose
