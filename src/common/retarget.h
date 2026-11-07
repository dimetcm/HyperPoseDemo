#pragma once

// Runtime humanoid retargeting between two ozz skeletons.
//
// Joints are matched by name through a canonical humanoid bone table (CMU/BVH,
// Mixamo and similar naming). Each frame, every mapped source bone's world
// rotation relative to its rest pose is applied to the matching target bone.
// Setup compensates for:
// - different facing directions (aligned from the thighs' left-right axis),
// - different rest poses, e.g. T-pose vs A-pose (rest bone directions are
//   aligned to the source's),
// - different proportions (hips motion scaled by the hip height ratio).
// Unmapped target joints keep their rest pose relative to their parent.

#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/quaternion.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/vec_float.h"
#include "ozz/base/span.h"

namespace ozz::animation {
class Skeleton;
}

class Retargeter {
 public:
  // Builds the joint mapping and rest pose corrections. Returns false if too
  // few humanoid bones could be matched between the two skeletons.
  bool Setup(const ozz::animation::Skeleton& _source,
             const ozz::animation::Skeleton& _target);

  // Computes target model-space matrices from source model-space matrices.
  void Run(
      ozz::span<const ozz::math::Float4x4> _source_joint_model_transforms,
      ozz::span<ozz::math::Float4x4> _target_joint_model_transforms) const;

  int num_mapped() const { return num_mapped_; }

 private:
  struct Joint {
    int parent = -1;
    int source = -1;  // Mapped source joint, -1 if unmapped.
    bool hips = false;
    ozz::math::Quaternion rest_local_rotation;
    ozz::math::Float3 rest_local_translation;
    // Constant part of mapped joints model rotation, folded from rest poses:
    //   model = facing * source_model * offset, with
    //   offset = inverse(source_rest) * inverse(facing) * rest_alignment *
    //            target_rest
    // where rest_alignment rotates the target rest bone onto the (facing
    // aligned) source rest bone.
    ozz::math::Quaternion offset;
  };
  ozz::vector<Joint> joints_;

  ozz::math::Quaternion facing_;  // Source to target facing rotation.
  float hips_scale_ = 1.f;
  int num_mapped_ = 0;

  // Per-joint scratch, mutable so Run stays const.
  mutable ozz::vector<ozz::math::Quaternion> model_rotations_;
  mutable ozz::vector<ozz::math::Float3> model_translations_;
};
