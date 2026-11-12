#pragma once

// Animation helpers shared by the demo apps.

#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/span.h"

namespace ozz {
namespace animation {
class Animation;
class Skeleton;
}  // namespace animation
namespace math {
struct SoaTransform;
}
namespace sample {
struct Mesh;
}
}  // namespace ozz

namespace hyperpose {

// Reads an animation archive header up to its duration, without loading the
// keyframes. Returns false if the file isn't an ozz animation.
bool ReadClipDuration(const char* _path, float* _duration);

// Frame rate _animation was sampled at before its keyframes were optimized
// (frames per second): the remaining key times all lie on that frame grid, so
// it's the inverse of the shortest gap between them (the gap to the final key,
// at the duration, aside). _fallback when it can't tell (fewer than two key
// times).
float AnimationFrameRate(const ozz::animation::Animation& _animation,
                         float _fallback = 30.f);

// Samples _animation at _ratio ([0,1]) into local transforms, then converts
// them to model-space matrices.
bool SamplePose(const ozz::animation::Animation& _animation,
                const ozz::animation::Skeleton& _skeleton, float _ratio,
                ozz::animation::SamplingJob::Context* _context,
                ozz::span<ozz::math::SoaTransform> _locals,
                ozz::span<ozz::math::Float4x4> _joint_model_transforms);

// Fills _skinning_matrices (at least _mesh.joint_remaps.size() long) from
// model-space joint matrices.
void ComputeSkinningMatrices(
    const ozz::sample::Mesh& _mesh,
    ozz::span<const ozz::math::Float4x4> _joint_model_transforms,
    ozz::span<ozz::math::Float4x4> _skinning_matrices);

}  // namespace hyperpose
