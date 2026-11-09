#include "common/pose_walk.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/pose_match.h"
#include "ozz/animation/runtime/blending_job.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/maths/quaternion.h"

namespace math = ozz::math;

namespace hyperpose {
namespace {

// Most tracks blended at once; older ones are dropped (only with segments
// much shorter than the blend).
const size_t kMaxTracks = 8;

// Root joint (joint 0: lane x of the first soa transform) translation and
// rotation.
void ReadRoot(const math::SoaTransform& _soa, math::Float3* _t, math::Quaternion* _q) {
  *_t = math::Float3(math::GetX(_soa.translation.x), math::GetX(_soa.translation.y),
                     math::GetX(_soa.translation.z));
  *_q = math::Quaternion(math::GetX(_soa.rotation.x), math::GetX(_soa.rotation.y),
                         math::GetX(_soa.rotation.z), math::GetX(_soa.rotation.w));
}

void WriteRoot(const math::Float3& _t, const math::Quaternion& _q, math::SoaTransform* _soa) {
  using math::simd_float4::Load1;
  _soa->translation.x = math::SetX(_soa->translation.x, Load1(_t.x));
  _soa->translation.y = math::SetX(_soa->translation.y, Load1(_t.y));
  _soa->translation.z = math::SetX(_soa->translation.z, Load1(_t.z));
  _soa->rotation.x = math::SetX(_soa->rotation.x, Load1(_q.x));
  _soa->rotation.y = math::SetX(_soa->rotation.y, Load1(_q.y));
  _soa->rotation.z = math::SetX(_soa->rotation.z, Load1(_q.z));
  _soa->rotation.w = math::SetX(_soa->rotation.w, Load1(_q.w));
}

math::Quaternion YawRotation(float _yaw) {
  return math::Quaternion::FromAxisAngle(math::Float3::y_axis(), _yaw);
}

}  // namespace

PoseWalk::PoseWalk() : rng_(std::random_device{}()) {}
PoseWalk::~PoseWalk() = default;

bool PoseWalk::Start(const CommonPoseResult* _result,
                     const ozz::animation::Skeleton* _skeleton,
                     int _heading_axis, int _pose) {
  Stop();
  if (_result->poses.empty()) {
    return false;
  }
  result_ = _result;
  skeleton_ = _skeleton;
  heading_axis_ = _heading_axis;
  root_position_ = math::Float3::zero();
  root_heading_ = 0.f;
  anchored_motion_ = root_motion;
  blended_.resize(skeleton_->num_soa_joints());
  if (_pose < 0) {
    std::vector<double> weights;
    for (const CommonPose& pose : result_->poses) {
      weights.push_back(static_cast<double>(pose.frames.size()));
    }
    _pose = std::discrete_distribution<int>(weights.begin(), weights.end())(rng_);
  }
  if (!Next(_pose, false)) {
    Stop();
    return false;
  }
  return true;
}

void PoseWalk::Stop() {
  result_ = nullptr;
  segment_ = Segment();
  // Loaded clips may not match the next skeleton.
  tracks_.clear();
  pool_.clear();
}

bool PoseWalk::Next(int _from, bool _after_end) {
  Segment segment;
  segment.after_end = _after_end;
  // Right after a clip end, only transitions to a pose, so the walk goes on
  // instead of ending again.
  auto usable = [&](const PoseEdge& _edge) { return !_after_end || _edge.to >= 0; };
  auto leads_on = [&](int _pose) {
    const auto& edges = result_->exits[_pose].edges;
    return std::any_of(edges.begin(), edges.end(), usable);
  };
  if (!leads_on(_from)) {
    // Nowhere to go from this pose: jumps to a pose the clips start in,
    // weighted by how many do.
    std::vector<double> weights(result_->poses.size(), 0.);
    for (const PoseTransition& transition : result_->transitions) {
      if (transition.from < 0 && leads_on(transition.to)) {
        weights[transition.to] += 1.;
      }
    }
    if (std::all_of(weights.begin(), weights.end(), [](double _w) { return _w == 0.; })) {
      return false;
    }
    _from = std::discrete_distribution<int>(weights.begin(), weights.end())(rng_);
    segment.jump = true;
  }
  // Next pose (or the clip end), weighted by how often the clips go there,
  // then one of the transitions there.
  const PoseExits& exits = result_->exits[_from];
  std::vector<double> weights;
  for (const PoseEdge& edge : exits.edges) {
    weights.push_back(usable(edge) ? static_cast<double>(edge.transitions.size()) : 0.);
  }
  const PoseEdge& edge =
      exits.edges[std::discrete_distribution<int>(weights.begin(), weights.end())(rng_)];
  const int index = edge.transitions[std::uniform_int_distribution<size_t>(
      0, edge.transitions.size() - 1)(rng_)];
  const PoseTransition& transition = result_->transitions[index];
  segment.transition = index;
  segment.from = _from;
  segment.to = transition.to;
  segment.clip = transition.clip;
  segment.time = transition.time;
  segment.start = std::max(EnteredAt(*result_, index), transition.time - max_stay);

  // A new track over the playing ones, reusing a pooled one that holds the
  // clip already.
  const auto& path = result_->clips[segment.clip];
  auto pooled = std::find_if(pool_.begin(), pool_.end(),
                             [&](const auto& _track) { return _track->clip.path() == path; });
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
  if (track->clip.path() != path && !track->clip.Load(path, *skeleton_)) {
    return false;
  }
  track->locals.resize(skeleton_->num_soa_joints());
  track->clip.time = segment.start;
  track->clip.playing = true;
  track->clip.loop = false;
  track->clip.speed = 1.f;
  track->age = 0.f;
  segment.time = std::min(segment.time, track->clip.duration());
  segment_ = segment;
  // Continues from where the character stands.
  Anchor(track.get());
  tracks_.push_back(std::move(track));
  if (tracks_.size() > kMaxTracks) {
    pool_.push_back(std::move(tracks_.front()));
    tracks_.erase(tracks_.begin());
  }
  return true;
}

void PoseWalk::Anchor(Track* _track) {
  if (!_track->clip.loaded() || !_track->clip.SampleLocals(_track->clip.time)) {
    return;
  }
  math::Float3 t;
  math::Quaternion q;
  ReadRoot(_track->clip.locals()[0], &t, &q);
  const float heading = RootHeading(math::Float4x4::FromQuaternion(q), heading_axis_);
  _track->offset.yaw = root_heading_ - heading;
  _track->offset.shift = root_position_ - RotateY(t, _track->offset.yaw);
  _track->offset.shift.y = 0.f;
}

bool PoseWalk::SampleRoot(Track* _track) {
  if (!_track->clip.SampleLocals(_track->clip.time)) {
    return false;
  }
  const auto locals = _track->clip.locals();
  std::copy(locals.begin(), locals.end(), _track->locals.begin());
  math::Float3 t;
  math::Quaternion q;
  ReadRoot(_track->locals[0], &t, &q);
  if (root_motion) {
    // Placed by the segment offset: moves as the clip does.
    const RootOffset& offset = _track->offset;
    WriteRoot(RotateY(t, offset.yaw) + offset.shift, YawRotation(offset.yaw) * q,
              &_track->locals[0]);
  } else {
    // Above the origin, facing the default heading, as AlignToOrigin does
    // in model space.
    const float heading = RootHeading(math::Float4x4::FromQuaternion(q), heading_axis_);
    WriteRoot(math::Float3(0.f, t.y, 0.f), YawRotation(-heading) * q, &_track->locals[0]);
  }
  return true;
}

void PoseWalk::ResetPosition() {
  root_position_ = math::Float3::zero();
  root_heading_ = 0.f;
  for (auto& track : tracks_) {
    Anchor(track.get());
  }
}

bool PoseWalk::Update(float _dt, ozz::span<math::Float4x4> _models) {
  if (!active()) {
    return false;
  }
  for (auto& track : tracks_) {
    track->clip.Advance(_dt);
    track->age += _dt;
  }
  Track* current = tracks_.back().get();
  if (current->clip.time >= segment_.time || !current->clip.playing) {
    // After a clip end, goes on from the pose it ended in, with another
    // clip that leaves that pose.
    const bool ended = segment_.to < 0;
    if (!Next(ended ? segment_.from : segment_.to, ended)) {
      Stop();
      return false;
    }
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
  // Where the next segment continues from.
  math::Store3PtrU(_models[0].cols[3], &root_position_.x);
  root_position_.y = 0.f;
  root_heading_ = RootHeading(_models[0], heading_axis_);
  return true;
}

float PoseWalk::clip_time() const {
  return tracks_.empty() ? 0.f : tracks_.back()->clip.time;
}
float PoseWalk::clip_duration() const {
  return tracks_.empty() ? 0.f : tracks_.back()->clip.duration();
}

}  // namespace hyperpose
