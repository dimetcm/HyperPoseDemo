#include "common/anim_utils.h"

#include <cassert>

#include "framework/mesh.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/io/archive.h"
#include "ozz/base/io/stream.h"
#include "ozz/base/maths/soa_transform.h"

namespace hyperpose {

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
                ozz::span<ozz::math::Float4x4> _models) {
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
  ltm_job.output = _models;
  return ltm_job.Run();
}

void ComputeSkinningMatrices(const ozz::sample::Mesh& _mesh,
                             ozz::span<const ozz::math::Float4x4> _models,
                             ozz::span<ozz::math::Float4x4> _skinning_matrices) {
  assert(_skinning_matrices.size() >= _mesh.joint_remaps.size());
  for (size_t i = 0; i < _mesh.joint_remaps.size(); ++i) {
    _skinning_matrices[i] =
        _models[_mesh.joint_remaps[i]] * _mesh.inverse_bind_poses[i];
  }
}

}  // namespace hyperpose
