#include "common/core/anim_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "framework/mesh.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/io/archive.h"
#include "ozz/base/io/stream.h"
#include "ozz/base/maths/soa_transform.h"

namespace hyperpose {

float AnimationFrameRate(const ozz::animation::Animation& _animation,
                         float _fallback) {
  // Key times are stored as ratios of the duration, sorted. The last key is
  // at the duration, which may fall between two frames (a source clip
  // resampled at another rate), so the gap before it is skipped when there
  // are others.
  const auto ratios = _animation.timepoints();
  const float duration = _animation.duration();
  const size_t end = ratios.size() > 2 ? ratios.size() - 1 : ratios.size();
  float gap = std::numeric_limits<float>::max();
  for (size_t i = 1; i < end; ++i) {
    const float seconds = (ratios[i] - ratios[i - 1]) * duration;
    if (seconds > 1e-4f) {  // Skips duplicates from rounding.
      gap = std::min(gap, seconds);
    }
  }
  if (gap == std::numeric_limits<float>::max()) {
    return _fallback;
  }
  const float rate = std::round(1.f / gap);
  return rate >= 1.f && rate <= 1000.f ? rate : _fallback;
}

bool ReadClipDuration(const char* _path, float* _duration) {
  ozz::io::File file(_path, "rb");
  if (!file.opened()) {
    return false;
  }
  ozz::io::IArchive archive(&file);
  if (!archive.TestTag<ozz::animation::Animation>()) {
    return false;
  }
  // Layout: tag, version, then duration first (see Animation::Load).
  file.Seek(ozz::io::internal::Tag<const ozz::animation::Animation>::kTagLength,
            ozz::io::Stream::kCurrent);
  uint32_t version = 0;
  archive >> version;
  archive >> *_duration;
  return version ==
         ozz::io::internal::Version<const ozz::animation::Animation>::kValue;
}

bool SamplePose(const ozz::animation::Animation& _animation,
                const ozz::animation::Skeleton& _skeleton, float _ratio,
                ozz::animation::SamplingJob::Context* _context,
                ozz::span<ozz::math::SoaTransform> _locals,
                ozz::span<ozz::math::Float4x4> _joint_model_transforms) {
  ozz::animation::SamplingJob sampling_job;
  sampling_job.animation = &_animation;
  sampling_job.context = _context;
  sampling_job.ratio = _ratio;
  sampling_job.output = _locals;
  if (!sampling_job.Run()) {
    return false;
  }

  ozz::animation::LocalToModelJob ltm_job;
  ltm_job.skeleton = &_skeleton;
  ltm_job.input = _locals;
  ltm_job.output = _joint_model_transforms;
  return ltm_job.Run();
}

void ComputeSkinningMatrices(
    const ozz::sample::Mesh& _mesh,
    ozz::span<const ozz::math::Float4x4> _joint_model_transforms,
    ozz::span<ozz::math::Float4x4> _skinning_matrices) {
  assert(_skinning_matrices.size() >= _mesh.joint_remaps.size());
  for (size_t i = 0; i < _mesh.joint_remaps.size(); ++i) {
    _skinning_matrices[i] =
        _joint_model_transforms[_mesh.joint_remaps[i]] *
        _mesh.inverse_bind_poses[i];
  }
}

}  // namespace hyperpose
