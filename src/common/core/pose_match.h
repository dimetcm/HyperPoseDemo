#pragma once

// Pose matching between animation clips: per frame pose features, pose
// difference matrices and best transition search.
//
// Pose features are joint positions relative to the root, with the root
// position (height included) and heading (rotation about Y) removed, so poses
// compare regardless of where the character stands, which way it faces and
// any vertical offset of the clip. Joint
// velocities are included, so poses moving in opposite directions don't match.

#include <cmath>

#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/quaternion.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/maths/vec_float.h"
#include "ozz/base/span.h"

namespace ozz::animation {
class Animation;
class Skeleton;
}  // namespace ozz::animation

namespace hyperpose {

struct PoseFeatures {
  float frame_rate = 0.f;
  int num_frames = 0;
  int num_joints = 0;
  // [frame * num_joints + joint], root relative and heading aligned.
  ozz::vector<ozz::math::Float3> positions;
  ozz::vector<ozz::math::Float3> velocities;  // Meters per second.
  // Per frame root position (model space) and heading (radians about Y).
  ozz::vector<ozz::math::Float3> root_positions;
  ozz::vector<float> root_headings;
  // First frame of the motion: frames before it are a leading bind pose (the
  // skeleton's rest pose, as conversions often leave on the first frame), and
  // velocities aren't sampled before it.
  int first_frame = 0;
};

// Samples _animation at _frame_rate and extracts pose features. The root is
// joint 0 of _skeleton. Leading frames in the skeleton's rest pose (within
// 5 cm, see PoseDistanceSqr) are detected as a bind pose (see first_frame),
// unless every frame is.
bool ExtractPoseFeatures(const ozz::animation::Animation& _animation,
                         const ozz::animation::Skeleton& _skeleton,
                         float _frame_rate, PoseFeatures* _features);

// Heading (radians about Y) of a root joint model-space transform, measured
// on its most horizontal axis in the rest pose (see HeadingAxis).
float RootHeading(const ozz::math::Float4x4& _root, int _axis);

// Index (0 x, 1 y, 2 z) of the root rest pose axis closest to horizontal.
int HeadingAxis(const ozz::animation::Skeleton& _skeleton);

// Moves a pose (joint model-space transforms, root first) so the root stands
// above the origin, facing the default heading (heading 0 on _axis, see
// HeadingAxis), so poses compare from the same viewpoint (see
// RootPlacement::ToOrigin).
void AlignToOrigin(ozz::span<ozz::math::Float4x4> _models, int _axis);

// Rotates _v about the Y axis by _angle radians (right handed, as
// ozz::math::Quaternion::FromAxisAngle).
ozz::math::Float3 RotateY(const ozz::math::Float3& _v, float _angle);

// Places a pose on the ground: rotates it by yaw about the vertical axis (Y),
// then moves it by shift on the ground. Made to take one root (position and
// heading) onto another, heights unchanged: aligning a pose to the origin,
// continuing a clip from where the character stands (root motion), lining up
// two clips at a transition.
struct RootPlacement {
  float yaw = 0.f;  // Radians about Y.
  ozz::math::Float3 shift = ozz::math::Float3::zero();  // y = 0.

  // Takes a root at _from, heading _from_heading, onto _to, heading
  // _to_heading (radians about Y).
  static RootPlacement Between(const ozz::math::Float3& _from, float _from_heading,
                               const ozz::math::Float3& _to, float _to_heading);
  // Same between root joint model-space transforms, headings measured on
  // _axis (see RootHeading).
  static RootPlacement Between(const ozz::math::Float4x4& _from,
                               const ozz::math::Float4x4& _to, int _axis);
  // Takes _root above the origin, facing the default heading (0).
  static RootPlacement ToOrigin(const ozz::math::Float4x4& _root, int _axis);

  ozz::math::Float3 Apply(const ozz::math::Float3& _position) const {
    return RotateY(_position, yaw) + shift;
  }
  ozz::math::Quaternion Apply(const ozz::math::Quaternion& _rotation) const {
    return ozz::math::Quaternion::FromAxisAngle(ozz::math::Float3::y_axis(), yaw) *
           _rotation;
  }
  // Places model-space joint transforms.
  void ApplyToModels(ozz::span<ozz::math::Float4x4> _models) const;
  // Places the root joint of local transforms (joint 0: having no parent,
  // its local transform is in model space), which places the whole pose.
  void ApplyToRoot(ozz::span<ozz::math::SoaTransform> _locals) const;
};

// Root joint (joint 0) local translation and rotation, from local transforms.
void ReadRoot(ozz::span<const ozz::math::SoaTransform> _locals,
              ozz::math::Float3* _translation, ozz::math::Quaternion* _rotation);

// Pose differences between every frame of a (rows) and b (columns).
struct PoseDifference {
  int rows = 0;
  int cols = 0;
  ozz::vector<float> values;  // [a_frame * cols + b_frame], meters.
  float at(int _a, int _b) const { return values[_a * cols + _b]; }
};

// The pose distance, shared by clustering, matching and blending: for each
// joint, the squared position difference plus _velocity_weight (seconds)
// squared times the squared velocity difference. A velocity difference of
// 1 m/s counts as _velocity_weight meters.
//
// Sum over _num_joints joints of that (m^2): what k-means minimizes.
float PoseDistanceSqr(const ozz::math::Float3* _a_positions,
                    const ozz::math::Float3* _a_velocities,
                    const ozz::math::Float3* _b_positions,
                    const ozz::math::Float3* _b_velocities, int _num_joints,
                    float _velocity_weight);
// Its per joint RMS (m): the root of the mean over joints, as shown.
inline float PoseRms(float _distance_sqr, int _num_joints) {
  return _num_joints > 0 ? std::sqrt(_distance_sqr / _num_joints) : 0.f;
}

// Pose difference between frame _a_frame of a and _b_frame of b: per joint
// RMS of PoseDistanceSqr (m). a and b must share joint count.
float PoseFrameDifference(const PoseFeatures& _a, int _a_frame,
                          const PoseFeatures& _b, int _b_frame,
                          float _velocity_weight);

// PoseFrameDifference between every frame of a and b.
void ComputePoseDifference(const PoseFeatures& _a, const PoseFeatures& _b,
                           float _velocity_weight, PoseDifference* _difference);

struct Transition {
  int a_frame = -1;
  int b_frame = -1;
  float difference = 0.f;
  bool valid() const { return a_frame >= 0; }
};

// True if a transition from a_frame to b_frame leaves _blend_frames in both
// clips, and frames are at least _min_gap apart (when _min_gap > 0, to skip
// trivial matches when a and b are the same clip).
bool IsValidTransition(const PoseDifference& _difference, int _a_frame,
                       int _b_frame, int _blend_frames, int _min_gap);

// Lowest difference valid transition.
Transition FindBestTransition(const PoseDifference& _difference,
                              int _blend_frames, int _min_gap);

}  // namespace hyperpose
