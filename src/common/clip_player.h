#pragma once

// Clip loading, playback and sampling, shared by the apps.

#include <filesystem>
#include <string>

#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/span.h"

namespace hyperpose {

// The skeleton clips are played on: the nearest skeleton.ozz of a clip (see
// FindClipSkeleton), loaded once and kept while clips share it.
class ClipSkeleton {
 public:
  enum class Result { kSame, kChanged, kFailed };

  // Makes the skeleton of _clip (searched up to _root) current. With
  // _allow_change false, a clip using another skeleton fails. On kChanged,
  // anything sized for or loaded with the previous skeleton is stale.
  Result Use(const std::filesystem::path& _clip,
             const std::filesystem::path& _root, bool _allow_change);

  bool loaded() const { return !path_.empty(); }
  const ozz::animation::Skeleton& skeleton() const { return skeleton_; }
  const std::filesystem::path& path() const { return path_; }
  int num_joints() const { return skeleton_.num_joints(); }
  // Rest pose axis used for root heading (see HeadingAxis).
  int heading_axis() const { return heading_axis_; }
  // Why the last Use failed.
  const std::string& error() const { return error_; }

 private:
  std::filesystem::path path_;
  ozz::animation::Skeleton skeleton_;
  int heading_axis_ = 0;
  std::string error_;
};

// An animation clip with its sampling state and playback time.
//
// Loading always resets the sampling context: it caches keyframes per
// animation address, which is reused when a player loads another clip.
class ClipPlayer {
 public:
  // Loads _path, which must have a track per _skeleton joint. Playback
  // restarts at 0. Unloads on failure. _skeleton must outlive the clip.
  bool Load(const std::filesystem::path& _path,
            const ozz::animation::Skeleton& _skeleton);
  void Unload();

  bool loaded() const { return skeleton_ != nullptr; }
  const std::filesystem::path& path() const { return path_; }
  const ozz::animation::Animation& animation() const { return animation_; }
  float duration() const { return animation_.duration(); }

  // Playback state, advanced by Advance.
  float time = 0.f;   // Seconds.
  float speed = 1.f;  // Negative plays backward.
  bool playing = true;
  bool loop = true;

  // Moves time by _dt * speed while playing. Looping wraps around the clip,
  // otherwise time stops at the ends (and playback pauses).
  void Advance(float _dt);

  // Samples at _time (seconds, clamped to the clip) into locals().
  bool SampleLocals(float _time);
  // Samples at _time into model-space joint transforms.
  bool Sample(float _time, ozz::span<ozz::math::Float4x4> _joint_model_transforms);
  // Samples at the playback time.
  bool Sample(ozz::span<ozz::math::Float4x4> _joint_model_transforms) {
    return Sample(time, _joint_model_transforms);
  }

  // Local transforms of the last sample.
  ozz::span<ozz::math::SoaTransform> locals() { return make_span(locals_); }

 private:
  const ozz::animation::Skeleton* skeleton_ = nullptr;
  std::filesystem::path path_;
  ozz::animation::Animation animation_;
  ozz::animation::SamplingJob::Context context_;
  ozz::vector<ozz::math::SoaTransform> locals_;
};

}  // namespace hyperpose
