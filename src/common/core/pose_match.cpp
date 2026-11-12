#include "common/core/pose_match.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "common/core/anim_utils.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/maths/soa_transform.h"

namespace math = ozz::math;

namespace hyperpose {

math::Float3 RotateY(const math::Float3& _v, float _angle) {
  const float c = std::cos(_angle), s = std::sin(_angle);
  return math::Float3(_v.x * c + _v.z * s, _v.y, -_v.x * s + _v.z * c);
}

int HeadingAxis(const ozz::animation::Skeleton& _skeleton) {
  ozz::vector<math::Float4x4> models(_skeleton.num_joints());
  ozz::animation::LocalToModelJob job;
  job.skeleton = &_skeleton;
  job.input = _skeleton.joint_rest_poses();
  job.output = make_span(models);
  if (!job.Run() || models.empty()) {
    return 0;
  }
  int best = 0;
  float best_y = 2.f;
  for (int axis = 0; axis < 3; ++axis) {
    math::Float3 v;
    math::Store3PtrU(math::Normalize3(models[0].cols[axis]), &v.x);
    if (std::fabs(v.y) < best_y) {
      best_y = std::fabs(v.y);
      best = axis;
    }
  }
  return best;
}

float RootHeading(const math::Float4x4& _root, int _axis) {
  math::Float3 v;
  math::Store3PtrU(_root.cols[_axis], &v.x);
  // RotateY(x axis, heading) gives (cos, 0, -sin).
  return std::atan2(-v.z, v.x);
}

void AlignToOrigin(ozz::span<math::Float4x4> _models, int _axis) {
  if (!_models.empty()) {
    RootPlacement::ToOrigin(_models[0], _axis).ApplyToModels(_models);
  }
}

RootPlacement RootPlacement::Between(const math::Float3& _from, float _from_heading,
                                     const math::Float3& _to, float _to_heading) {
  RootPlacement placement;
  placement.yaw = _to_heading - _from_heading;
  placement.shift = _to - RotateY(_from, placement.yaw);
  placement.shift.y = 0.f;  // Heights unchanged.
  return placement;
}

RootPlacement RootPlacement::Between(const math::Float4x4& _from,
                                     const math::Float4x4& _to, int _axis) {
  math::Float3 from, to;
  math::Store3PtrU(_from.cols[3], &from.x);
  math::Store3PtrU(_to.cols[3], &to.x);
  return Between(from, RootHeading(_from, _axis), to, RootHeading(_to, _axis));
}

RootPlacement RootPlacement::ToOrigin(const math::Float4x4& _root, int _axis) {
  math::Float3 root;
  math::Store3PtrU(_root.cols[3], &root.x);
  return Between(root, RootHeading(_root, _axis), math::Float3::zero(), 0.f);
}

void RootPlacement::ApplyToModels(ozz::span<math::Float4x4> _models) const {
  const math::Float4x4 placement =
      math::Float4x4::Translation(math::simd_float4::Load(shift.x, 0.f, shift.z, 1.f)) *
      math::Float4x4::FromAxisAngle(math::simd_float4::y_axis(),
                                    math::simd_float4::Load1(yaw));
  for (math::Float4x4& model : _models) {
    model = placement * model;
  }
}

void ReadRoot(ozz::span<const math::SoaTransform> _locals, math::Float3* _translation,
              math::Quaternion* _rotation) {
  const math::SoaTransform& soa = _locals[0];
  *_translation = math::Float3(math::GetX(soa.translation.x), math::GetX(soa.translation.y),
                               math::GetX(soa.translation.z));
  *_rotation = math::Quaternion(math::GetX(soa.rotation.x), math::GetX(soa.rotation.y),
                                math::GetX(soa.rotation.z), math::GetX(soa.rotation.w));
}

void RootPlacement::ApplyToRoot(ozz::span<math::SoaTransform> _locals) const {
  math::Float3 t;
  math::Quaternion q;
  ReadRoot(_locals, &t, &q);
  t = Apply(t);
  q = Apply(q);
  // Root joint: lane x of the first soa transform.
  math::SoaTransform& soa = _locals[0];
  using math::simd_float4::Load1;
  soa.translation.x = math::SetX(soa.translation.x, Load1(t.x));
  soa.translation.y = math::SetX(soa.translation.y, Load1(t.y));
  soa.translation.z = math::SetX(soa.translation.z, Load1(t.z));
  soa.rotation.x = math::SetX(soa.rotation.x, Load1(q.x));
  soa.rotation.y = math::SetX(soa.rotation.y, Load1(q.y));
  soa.rotation.z = math::SetX(soa.rotation.z, Load1(q.z));
  soa.rotation.w = math::SetX(soa.rotation.w, Load1(q.w));
}

bool ExtractPoseFeatures(const ozz::animation::Animation& _animation,
                         const ozz::animation::Skeleton& _skeleton,
                         float _frame_rate, PoseFeatures* _features) {
  PoseFeatures& f = *_features;
  const float duration = _animation.duration();
  f.frame_rate = _frame_rate;
  // Frames on the grid up to the end (the epsilon keeps the last frame when
  // the duration is a whole number of frames, despite rounding).
  f.num_frames = std::max(1, static_cast<int>(duration * _frame_rate + 1e-3f) + 1);
  f.num_joints = _skeleton.num_joints();
  f.positions.resize(f.num_frames * f.num_joints);
  f.velocities.resize(f.num_frames * f.num_joints);
  f.root_positions.resize(f.num_frames);
  f.root_headings.resize(f.num_frames);

  const int axis = HeadingAxis(_skeleton);
  ozz::vector<math::SoaTransform> locals(_skeleton.num_soa_joints());
  ozz::vector<math::Float4x4> models(_skeleton.num_joints());
  ozz::animation::SamplingJob::Context context(_skeleton.num_joints());

  // Pose features of model-space joint matrices (models): relative to the
  // whole root position, height included, so clips whose root sits at a
  // different height (vertical offset) still compare, and heading aligned.
  auto to_features = [&](math::Float3* _positions, math::Float3* _root, float* _heading) {
    math::Float3 root;
    math::Store3PtrU(models[0].cols[3], &root.x);
    const float heading = RootHeading(models[0], axis);
    for (int j = 0; j < f.num_joints; ++j) {
      math::Float3 p;
      math::Store3PtrU(models[j].cols[3], &p.x);
      _positions[j] = RotateY(p - root, -heading);
    }
    if (_root) {
      *_root = root;
      *_heading = heading;
    }
  };
  // Samples the pose at _time into _positions (num_joints).
  auto sample = [&](float _time, math::Float3* _positions, math::Float3* _root,
                    float* _heading) {
    if (!SamplePose(_animation, _skeleton, duration > 0.f ? _time / duration : 0.f,
                    &context, make_span(locals), make_span(models))) {
      return false;
    }
    to_features(_positions, _root, _heading);
    return true;
  };
  auto frame_time = [&](int _frame) { return std::min(_frame / _frame_rate, duration); };

  for (int frame = 0; frame < f.num_frames; ++frame) {
    if (!sample(frame_time(frame), &f.positions[frame * f.num_joints],
                &f.root_positions[frame], &f.root_headings[frame])) {
      return false;
    }
  }

  // Leading bind pose: frames in the skeleton's rest pose.
  ozz::vector<math::Float3> rest(f.num_joints);
  {
    ozz::animation::LocalToModelJob job;
    job.skeleton = &_skeleton;
    job.input = _skeleton.joint_rest_poses();
    job.output = make_span(models);
    if (!job.Run()) {
      return false;
    }
    to_features(rest.data(), nullptr, nullptr);
  }
  // Pose distance of positions (m, per joint RMS). Converted bind poses land
  // a few cm off the rest pose, real motion tens of cm.
  const float kBindPoseTolerance = .05f;
  f.first_frame = 0;
  while (f.first_frame < f.num_frames) {
    const math::Float3* p = &f.positions[f.first_frame * f.num_joints];
    const float distance =
        PoseRms(PoseDistanceSqr(p, nullptr, rest.data(), nullptr, f.num_joints, 0.f),
                f.num_joints);
    if (distance > kBindPoseTolerance) {
      break;
    }
    ++f.first_frame;
  }
  if (f.first_frame == f.num_frames) {
    f.first_frame = 0;  // All rest pose: nothing else to use.
  }
  const float start = frame_time(f.first_frame);

  // Velocities from two extra samples kVelocityStep around each frame,
  // rather than from the neighbor frames: as precise at the clip ends as in
  // the middle (one sided there, but over a short step), and independent of
  // the frame rate. Not sampled before the motion starts, so the snap out of
  // a bind pose doesn't count.
  const float kVelocityStep = 1.f / 60.f;
  ozz::vector<math::Float3> before(f.num_joints), after(f.num_joints);
  for (int frame = 0; frame < f.num_frames; ++frame) {
    const float time = frame_time(frame);
    const float t0 = std::max(time - kVelocityStep, start);
    const float t1 = std::min(time + kVelocityStep, duration);
    math::Float3* velocities = &f.velocities[frame * f.num_joints];
    if (t1 <= t0) {  // A single frame clip, or a bind pose frame.
      std::fill(velocities, velocities + f.num_joints, math::Float3::zero());
      continue;
    }
    if (!sample(t0, before.data(), nullptr, nullptr) ||
        !sample(t1, after.data(), nullptr, nullptr)) {
      return false;
    }
    for (int j = 0; j < f.num_joints; ++j) {
      velocities[j] = (after[j] - before[j]) * (1.f / (t1 - t0));
    }
  }
  return true;
}

void ComputePoseDifference(const PoseFeatures& _a, const PoseFeatures& _b,
                           float _velocity_weight, PoseDifference* _difference) {
  PoseDifference& d = *_difference;
  d.rows = _a.num_frames;
  d.cols = _b.num_frames;
  d.values.assign(static_cast<size_t>(d.rows) * d.cols, 0.f);
  const int num_joints = std::min(_a.num_joints, _b.num_joints);
  if (num_joints == 0) {
    return;
  }
  for (int a = 0; a < d.rows; ++a) {
    for (int b = 0; b < d.cols; ++b) {
      d.values[a * d.cols + b] = PoseFrameDifference(_a, a, _b, b, _velocity_weight);
    }
  }
}

float PoseFrameDifference(const PoseFeatures& _a, int _a_frame,
                          const PoseFeatures& _b, int _b_frame,
                          float _velocity_weight) {
  const int num_joints = std::min(_a.num_joints, _b.num_joints);
  if (num_joints == 0) {
    return 0.f;
  }
  const size_t a = static_cast<size_t>(_a_frame) * _a.num_joints;
  const size_t b = static_cast<size_t>(_b_frame) * _b.num_joints;
  return PoseRms(PoseDistanceSqr(&_a.positions[a], &_a.velocities[a], &_b.positions[b],
                               &_b.velocities[b], num_joints, _velocity_weight),
                 num_joints);
}

float PoseDistanceSqr(const math::Float3* _a_positions,
                    const math::Float3* _a_velocities,
                    const math::Float3* _b_positions,
                    const math::Float3* _b_velocities, int _num_joints,
                    float _velocity_weight) {
  float position = 0.f;
  for (int j = 0; j < _num_joints; ++j) {
    position += LengthSqr(_a_positions[j] - _b_positions[j]);
  }
  if (_velocity_weight <= 0.f) {
    return position;
  }
  float velocity = 0.f;
  for (int j = 0; j < _num_joints; ++j) {
    velocity += LengthSqr(_a_velocities[j] - _b_velocities[j]);
  }
  return position + _velocity_weight * _velocity_weight * velocity;
}

bool IsValidTransition(const PoseDifference& _difference, int _a_frame,
                       int _b_frame, int _blend_frames, int _min_gap) {
  return _a_frame >= 0 && _b_frame >= 0 &&
         _a_frame + _blend_frames < _difference.rows &&
         _b_frame + _blend_frames < _difference.cols &&
         (_min_gap <= 0 || std::abs(_a_frame - _b_frame) >= _min_gap);
}

Transition FindBestTransition(const PoseDifference& _difference,
                              int _blend_frames, int _min_gap) {
  Transition best;
  for (int a = 0; a < _difference.rows; ++a) {
    for (int b = 0; b < _difference.cols; ++b) {
      if (!IsValidTransition(_difference, a, b, _blend_frames, _min_gap)) {
        continue;
      }
      const float value = _difference.at(a, b);
      if (!best.valid() || value < best.difference) {
        best = {a, b, value};
      }
    }
  }
  return best;
}

}  // namespace hyperpose
