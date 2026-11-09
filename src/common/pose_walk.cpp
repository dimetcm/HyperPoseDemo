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

PoseWalk::PoseWalk()
    : current_(std::make_unique<ClipPlayer>()),
      previous_(std::make_unique<ClipPlayer>()),
      rng_(std::random_device{}()) {}
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
  current_locals_.resize(skeleton_->num_soa_joints());
  previous_locals_.resize(skeleton_->num_soa_joints());
  blended_.resize(skeleton_->num_soa_joints());
  if (_pose < 0) {
    std::vector<double> weights;
    for (const CommonPose& pose : result_->poses) {
      weights.push_back(static_cast<double>(pose.frames.size()));
    }
    _pose = std::discrete_distribution<int>(weights.begin(), weights.end())(rng_);
  }
  if (!Next(_pose)) {
    Stop();
    return false;
  }
  return true;
}

void PoseWalk::Stop() {
  result_ = nullptr;
  segment_ = Segment();
  current_->Unload();
  previous_->Unload();
}

bool PoseWalk::Next(int _from) {
  const CommonPose& from = result_->poses[_from];
  Segment segment;
  segment.from = _from;
  if (!from.transitions.empty()) {
    // Next pose, weighted by how often the clips go there, then one of the
    // examples of that transition.
    std::vector<double> weights;
    for (const CommonPose::Transition& t : from.transitions) {
      weights.push_back(static_cast<double>(t.examples.size()));
    }
    const CommonPose::Transition& transition =
        from.transitions[std::discrete_distribution<int>(weights.begin(), weights.end())(rng_)];
    const CommonPose::Transition::Example& example = transition.examples
        [std::uniform_int_distribution<size_t>(0, transition.examples.size() - 1)(rng_)];
    segment.to = transition.to;
    segment.clip = example.clip;
    segment.time = segment.stop = example.time;
    segment.start = std::max(example.begin, example.time - max_stay);
  } else {
    // The clips end in this pose: jumps to a frame of a random pose
    // (weighted by frames), typical frames more often.
    std::vector<double> weights;
    for (const CommonPose& pose : result_->poses) {
      weights.push_back(static_cast<double>(pose.frames.size()));
    }
    segment.jump = true;
    segment.to = std::discrete_distribution<int>(weights.begin(), weights.end())(rng_);
    const CommonPose& to = result_->poses[segment.to];
    const float sigma = std::max(to.spread, 1e-4f);
    weights.clear();
    for (const CommonPose::Frame& frame : to.frames) {
      const double x = frame.distance / sigma;
      weights.push_back(std::exp(-.5 * x * x));
    }
    const CommonPose::Frame& frame =
        to.frames[std::discrete_distribution<int>(weights.begin(), weights.end())(rng_)];
    segment.clip = frame.clip;
    segment.time = segment.start = frame.time;
    // No stay length known for a frame: a short one, the next segment starts
    // where a clip entered this pose anyway.
    segment.stop = frame.time + std::min(max_stay, .5f);
  }

  // The playing clip fades out while the new one starts. Reuses the faded
  // out player when it holds the clip already.
  std::swap(current_, previous_);
  const auto& path = result_->clips[segment.clip];
  if (current_->path() != path && !current_->Load(path, *skeleton_)) {
    return false;
  }
  current_->time = segment.start;
  current_->playing = true;
  current_->loop = false;
  current_->speed = 1.f;
  segment.stop = std::min(segment.stop, current_->duration());
  segment_ = segment;
  fade_ = 0.f;
  return true;
}

bool PoseWalk::SampleAligned(ClipPlayer* _clip, ozz::span<math::SoaTransform> _out) {
  if (!_clip->SampleLocals(_clip->time)) {
    return false;
  }
  const auto locals = _clip->locals();
  std::copy(locals.begin(), locals.end(), _out.begin());

  // Root (joint 0, lane x of the first soa transform) above the origin,
  // facing the default heading, as AlignToOrigin does in model space.
  math::SoaTransform& root = _out[0];
  const math::Quaternion q(math::GetX(root.rotation.x), math::GetX(root.rotation.y),
                           math::GetX(root.rotation.z), math::GetX(root.rotation.w));
  const float heading = RootHeading(math::Float4x4::FromQuaternion(q), heading_axis_);
  const math::Quaternion r =
      math::Quaternion::FromAxisAngle(math::Float3::y_axis(), -heading) * q;
  const math::SimdFloat4 zero = math::simd_float4::zero();
  root.translation.x = math::SetX(root.translation.x, zero);
  root.translation.z = math::SetX(root.translation.z, zero);
  root.rotation.x = math::SetX(root.rotation.x, math::simd_float4::Load1(r.x));
  root.rotation.y = math::SetX(root.rotation.y, math::simd_float4::Load1(r.y));
  root.rotation.z = math::SetX(root.rotation.z, math::simd_float4::Load1(r.z));
  root.rotation.w = math::SetX(root.rotation.w, math::simd_float4::Load1(r.w));
  return true;
}

bool PoseWalk::Update(float _dt, ozz::span<math::Float4x4> _models) {
  if (!active()) {
    return false;
  }
  current_->Advance(_dt);
  if (previous_->loaded()) {
    previous_->Advance(_dt);
  }
  fade_ += _dt;
  if (current_->time >= segment_.stop || !current_->playing) {
    if (!Next(segment_.to)) {
      Stop();
      return false;
    }
  }

  if (!SampleAligned(current_.get(), make_span(current_locals_))) {
    return false;
  }
  ozz::span<const math::SoaTransform> locals = make_span(current_locals_);
  if (fade_ < blend && previous_->loaded() &&
      SampleAligned(previous_.get(), make_span(previous_locals_))) {
    const float weight = fade_ / blend;
    ozz::animation::BlendingJob::Layer layers[2];
    layers[0].weight = 1.f - weight;
    layers[0].transform = make_span(previous_locals_);
    layers[1].weight = weight;
    layers[1].transform = make_span(current_locals_);
    ozz::animation::BlendingJob job;
    job.layers = layers;
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
  return ltm.Run();
}

float PoseWalk::clip_time() const { return current_->time; }
float PoseWalk::clip_duration() const { return current_->duration(); }

}  // namespace hyperpose
