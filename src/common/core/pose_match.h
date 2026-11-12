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
#include "ozz/base/maths/simd_math.h"
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
// HeadingAxis), so poses compare from the same viewpoint.
void AlignToOrigin(ozz::span<ozz::math::Float4x4> _models, int _axis);

// Rotates _v about the Y axis by _angle radians (right handed, as
// ozz::math::Quaternion::FromAxisAngle).
ozz::math::Float3 RotateY(const ozz::math::Float3& _v, float _angle);

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
