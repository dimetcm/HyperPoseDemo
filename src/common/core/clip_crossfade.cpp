#include "common/core/clip_crossfade.h"

#include <algorithm>

#include "common/core/pose_match.h"
#include "ozz/animation/runtime/blending_job.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/skeleton.h"

namespace math = ozz::math;

namespace hyperpose {
namespace {

// Most tracks blended at once; older ones are dropped (only with clips much
// shorter than the blend).
const size_t kMaxTracks = 8;

}  // namespace

ClipCrossfade::ClipCrossfade() = default;
ClipCrossfade::~ClipCrossfade() = default;

void ClipCrossfade::Reset(const ozz::animation::Skeleton* _skeleton, int _heading_axis) {
  Clear();
  skeleton_ = _skeleton;
  heading_axis_ = _heading_axis;
  root_position_ = math::Float3::zero();
  root_heading_ = 0.f;
  anchored_motion_ = root_motion;
  blended_.resize(skeleton_->num_soa_joints());
}

void ClipCrossfade::Clear() {
  tracks_.clear();
  pool_.clear();
}

bool ClipCrossfade::Play(const std::filesystem::path& _clip, float _time) {
  // A new track over the playing ones, reusing a pooled one that holds the
  // clip already.
  auto pooled = std::find_if(pool_.begin(), pool_.end(),
                             [&](const auto& _track) { return _track->clip.path() == _clip; });
  if (pooled == pool_.end() && !pool_.empty()) {
    pooled = pool_.begin();
  }
  std::unique_ptr<Track> track;
  if (pooled != pool_.end()) {
    track = std::move(*pooled);
    pool_.erase(pooled);
  } else {
    track = std::make_unique<Track>();
  }
  if (track->clip.path() != _clip && !track->clip.Load(_clip, *skeleton_)) {
    return false;
  }
  track->locals.resize(skeleton_->num_soa_joints());
  track->clip.time = _time;
  track->clip.playing = true;
  track->clip.loop = false;
  track->clip.speed = 1.f;
  track->age = 0.f;
  // Continues from where the character stands.
  Anchor(track.get());
  tracks_.push_back(std::move(track));
  if (tracks_.size() > kMaxTracks) {
    pool_.push_back(std::move(tracks_.front()));
    tracks_.erase(tracks_.begin());
  }
  return true;
}

void ClipCrossfade::Anchor(Track* _track) {
  if (!_track->clip.loaded() || !_track->clip.SampleLocals(_track->clip.time)) {
    return;
  }
  math::Float3 t;
  math::Quaternion q;
  ReadRoot(_track->clip.locals(), &t, &q);
  _track->placement =
      RootPlacement::Between(t, RootHeading(math::Float4x4::FromQuaternion(q), heading_axis_),
                             root_position_, root_heading_);
}

bool ClipCrossfade::SampleRoot(Track* _track) {
  if (!_track->clip.SampleLocals(_track->clip.time)) {
    return false;
  }
  const auto locals = _track->clip.locals();
  std::copy(locals.begin(), locals.end(), _track->locals.begin());
  if (root_motion) {
    // Placed where the track started: moves as the clip does.
    _track->placement.ApplyToRoot(make_span(_track->locals));
  } else {
    // Above the origin, facing the default heading, as AlignToOrigin does
    // in model space.
    math::Float3 t;
    math::Quaternion q;
    ReadRoot(make_span(_track->locals), &t, &q);
    RootPlacement::Between(t, RootHeading(math::Float4x4::FromQuaternion(q), heading_axis_),
                           math::Float3::zero(), 0.f)
        .ApplyToRoot(make_span(_track->locals));
  }
  return true;
}

void ClipCrossfade::ResetPosition() {
  root_position_ = math::Float3::zero();
  root_heading_ = 0.f;
  for (auto& track : tracks_) {
    Anchor(track.get());
  }
}

void ClipCrossfade::Advance(float _dt) {
  for (auto& track : tracks_) {
    track->clip.Advance(_dt);
    track->age += _dt;
  }
}

bool ClipCrossfade::Sample(ozz::span<math::Float4x4> _models) {
  if (tracks_.empty()) {
    return false;
  }
  // Root motion switched on: continues from where the character stands.
  if (root_motion != anchored_motion_) {
    anchored_motion_ = root_motion;
    for (auto& track : tracks_) {
      Anchor(track.get());
    }
  }

  // Crossfade weights, newest first: each track fades in over the older
  // ones, which share what's left. Tracks under a fully faded in one are
  // done.
  auto fade_in = [&](const Track& _track) {
    return blend > 0.f ? std::min(1.f, _track.age / blend) : 1.f;
  };
  for (size_t i = tracks_.size() - 1; i > 0; --i) {
    if (fade_in(*tracks_[i]) >= 1.f) {
      for (size_t j = 0; j < i; ++j) {
        pool_.push_back(std::move(tracks_[j]));
      }
      tracks_.erase(tracks_.begin(), tracks_.begin() + i);
      break;
    }
  }
  ozz::animation::BlendingJob::Layer layers[kMaxTracks];
  float remaining = 1.f;
  for (size_t i = tracks_.size(); i-- > 0;) {
    Track& track = *tracks_[i];
    if (!SampleRoot(&track)) {
      return false;
    }
    const float in = i == 0 ? 1.f : fade_in(track);  // The oldest takes the rest.
    layers[i].weight = remaining * in;
    layers[i].transform = make_span(track.locals);
    remaining *= 1.f - in;
  }
  ozz::span<const math::SoaTransform> locals = make_span(tracks_.back()->locals);
  if (tracks_.size() > 1) {
    ozz::animation::BlendingJob job;
    job.layers = ozz::span<const ozz::animation::BlendingJob::Layer>(layers, tracks_.size());
    job.rest_pose = skeleton_->joint_rest_poses();
    job.output = make_span(blended_);
    if (job.Run()) {
      locals = make_span(blended_);
    }
  }

  ozz::animation::LocalToModelJob ltm;
  ltm.skeleton = skeleton_;
  ltm.input = locals;
  ltm.output = _models;
  if (!ltm.Run()) {
    return false;
  }
  // Where the next clips continue from.
  math::Store3PtrU(_models[0].cols[3], &root_position_.x);
  root_position_.y = 0.f;
  root_heading_ = RootHeading(_models[0], heading_axis_);
  return true;
}

}  // namespace hyperpose
