#include "common/pose_match.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "common/anim_utils.h"
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
  if (_models.empty()) {
    return;
  }
  math::Float3 root;
  math::Store3PtrU(_models[0].cols[3], &root.x);
  const float heading = RootHeading(_models[0], _axis);
  const math::Float4x4 align =
      math::Float4x4::FromAxisAngle(math::simd_float4::y_axis(),
                                    math::simd_float4::Load1(-heading)) *
      math::Float4x4::Translation(math::simd_float4::Load(-root.x, 0.f, -root.z, 1.f));
  for (auto& model : _models) {
    model = align * model;
  }
}

bool ExtractPoseFeatures(const ozz::animation::Animation& _animation,
                         const ozz::animation::Skeleton& _skeleton,
                         float _frame_rate, PoseFeatures* _features) {
  PoseFeatures& f = *_features;
  const float duration = _animation.duration();
  f.frame_rate = _frame_rate;
  f.num_frames = std::max(1, static_cast<int>(duration * _frame_rate) + 1);
  f.num_joints = _skeleton.num_joints();
  f.positions.resize(f.num_frames * f.num_joints);
  f.velocities.resize(f.num_frames * f.num_joints);
  f.root_positions.resize(f.num_frames);
  f.root_headings.resize(f.num_frames);

  const int axis = HeadingAxis(_skeleton);
  ozz::vector<math::SoaTransform> locals(_skeleton.num_soa_joints());
  ozz::vector<math::Float4x4> models(_skeleton.num_joints());
  ozz::animation::SamplingJob::Context context(_skeleton.num_joints());

  // Samples the pose at _time into _positions (num_joints): relative to the
  // whole root position, height included, so clips whose root sits at a
  // different height (vertical offset) still compare, and heading aligned.
  auto sample = [&](float _time, math::Float3* _positions, math::Float3* _root,
                    float* _heading) {
    if (!SamplePose(_animation, _skeleton, duration > 0.f ? _time / duration : 0.f,
                    &context, make_span(locals), make_span(models))) {
      return false;
    }
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
    return true;
  };

  // Velocities from two extra samples kVelocityStep around each frame,
  // rather than from the neighbor frames: as precise at the clip ends as in
  // the middle (one sided there, but over a short step), and independent of
  // the frame rate.
  const float kVelocityStep = 1.f / 60.f;
  ozz::vector<math::Float3> before(f.num_joints), after(f.num_joints);
  for (int frame = 0; frame < f.num_frames; ++frame) {
    const float time = std::min(frame / _frame_rate, duration);
    if (!sample(time, &f.positions[frame * f.num_joints], &f.root_positions[frame],
                &f.root_headings[frame])) {
      return false;
    }
    const float t0 = std::max(time - kVelocityStep, 0.f);
    const float t1 = std::min(time + kVelocityStep, duration);
    math::Float3* velocities = &f.velocities[frame * f.num_joints];
    if (t1 <= t0) {  // A single frame clip.
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
  const math::Float3* pa = &_a.positions[_a_frame * _a.num_joints];
  const math::Float3* va = &_a.velocities[_a_frame * _a.num_joints];
  const math::Float3* pb = &_b.positions[_b_frame * _b.num_joints];
  const math::Float3* vb = &_b.velocities[_b_frame * _b.num_joints];
  float position = 0.f, velocity = 0.f;
  for (int j = 0; j < num_joints; ++j) {
    position += Length(pa[j] - pb[j]);
    velocity += Length(va[j] - vb[j]);
  }
  return (position + _velocity_weight * velocity) / num_joints;
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
