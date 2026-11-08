#include "common/clip_player.h"

#include <algorithm>
#include <cmath>

#include "common/clip_library.h"
#include "common/pose_match.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/local_to_model_job.h"

namespace fs = std::filesystem;

namespace hyperpose {

ClipSkeleton::Result ClipSkeleton::Use(const fs::path& _clip,
                                       const fs::path& _root,
                                       bool _allow_change) {
  error_.clear();
  const fs::path path = FindClipSkeleton(_clip, _root);
  if (path.empty()) {
    error_ = "No skeleton.ozz found for " + _clip.filename().string();
    return Result::kFailed;
  }
  if (path == path_) {
    return Result::kSame;
  }
  if (!_allow_change && loaded()) {
    error_ = "Clips must share a skeleton: " + _clip.filename().string() +
             " doesn't use " + path_.string();
    return Result::kFailed;
  }
  path_.clear();
  if (!ozz::sample::LoadSkeleton(path.string().c_str(), &skeleton_)) {
    error_ = "Failed to load " + path.string();
    return Result::kFailed;
  }
  path_ = path;
  heading_axis_ = HeadingAxis(skeleton_);
  return Result::kChanged;
}

bool ClipPlayer::Load(const fs::path& _path,
                      const ozz::animation::Skeleton& _skeleton) {
  Unload();
  if (!ozz::sample::LoadAnimation(_path.string().c_str(), &animation_) ||
      animation_.num_tracks() != _skeleton.num_joints()) {
    return false;
  }
  skeleton_ = &_skeleton;
  path_ = _path;
  context_.Resize(_skeleton.num_joints());  // Also invalidates the cache.
  locals_.resize(_skeleton.num_soa_joints());
  time = 0.f;
  return true;
}

void ClipPlayer::Unload() {
  skeleton_ = nullptr;
  path_.clear();
  context_.Invalidate();
}

void ClipPlayer::Advance(float _dt) {
  const float duration = animation_.duration();
  if (!loaded() || !playing || duration <= 0.f) {
    return;
  }
  time += _dt * speed;
  if (loop) {
    time = std::fmod(time, duration);
    if (time < 0.f) {
      time += duration;
    }
  } else if (time >= duration || time <= 0.f) {
    time = std::clamp(time, 0.f, duration);
    playing = false;
  }
}

bool ClipPlayer::SampleLocals(float _time) {
  if (!loaded()) {
    return false;
  }
  const float duration = animation_.duration();
  ozz::animation::SamplingJob job;
  job.animation = &animation_;
  job.context = &context_;
  job.ratio = duration > 0.f ? std::clamp(_time / duration, 0.f, 1.f) : 0.f;
  job.output = make_span(locals_);
  return job.Run();
}

bool ClipPlayer::Sample(float _time,
                        ozz::span<ozz::math::Float4x4> _joint_model_transforms) {
  if (!SampleLocals(_time)) {
    return false;
  }
  ozz::animation::LocalToModelJob ltm;
  ltm.skeleton = skeleton_;
  ltm.input = make_span(locals_);
  ltm.output = _joint_model_transforms;
  return ltm.Run();
}

}  // namespace hyperpose
