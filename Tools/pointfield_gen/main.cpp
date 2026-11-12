// pointfield_gen: generates a point field (points on the character's skinned
// surface) for every frame of an animation clip.
//
// Methods:
//   skinned:  samples once in the rest pose, then follows the skinned surface.
//             Points keep their identity across frames.
//   perframe: samples the posed surface again every frame.
//
// Clips made for another skeleton are retargeted onto the character.
//
// Example:
//   pointfield_gen --character=Assets/ozz/dagger --animation=Assets/ozz/cmu/01_01.ozz
//                  --method=skinned --sampler=uniform --count=10000 --output=01_01.hppf

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "common/core/anim_utils.h"
#include "common/core/point_field.h"
#include "common/core/retarget.h"
#include "framework/mesh.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/log.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/options/options.h"

namespace fs = std::filesystem;
using hyperpose::PointFieldGenerator;

OZZ_OPTIONS_DECLARE_STRING(character, "Character folder (skeleton.ozz + mesh.ozz).", "", true)
OZZ_OPTIONS_DECLARE_STRING(animation, "Animation clip (.ozz).", "", true)
OZZ_OPTIONS_DECLARE_STRING(clip_skeleton, "Clip skeleton (default: skeleton.ozz next to the clip).", "", false)
OZZ_OPTIONS_DECLARE_STRING(output, "Output point field file (.hppf).", "", true)
OZZ_OPTIONS_DECLARE_STRING(method, "skinned or perframe.", "skinned", false)
OZZ_OPTIONS_DECLARE_STRING(sampler, "random or uniform.", "uniform", false)
OZZ_OPTIONS_DECLARE_INT(count, "Points per frame.", 10000, false)
OZZ_OPTIONS_DECLARE_INT(seed, "Random seed.", 1, false)
OZZ_OPTIONS_DECLARE_FLOAT(frame_rate, "Output frames per second.", 30.f, false)

int main(int _argc, const char** _argv) {
  if (ozz::options::ParseCommandLine(_argc, _argv, "1.0",
                                     "Generates per frame point fields.") !=
      ozz::options::kSuccess) {
    return EXIT_FAILURE;
  }
  const bool skinned = std::strcmp(OPTIONS_method, "skinned") == 0;
  if (!skinned && std::strcmp(OPTIONS_method, "perframe") != 0) {
    ozz::log::Err() << "Unknown method " << OPTIONS_method.value() << std::endl;
    return EXIT_FAILURE;
  }
  const PointFieldGenerator::Sampler sampler =
      PointFieldGenerator::FindSampler(OPTIONS_sampler);
  if (!sampler) {
    ozz::log::Err() << "Unknown sampler " << OPTIONS_sampler.value() << std::endl;
    return EXIT_FAILURE;
  }

  // Character.
  const fs::path character_dir = OPTIONS_character.value();
  const fs::path character_skeleton_path = character_dir / "skeleton.ozz";
  ozz::animation::Skeleton character_skeleton;
  ozz::vector<ozz::sample::Mesh> meshes;
  if (!ozz::sample::LoadSkeleton(character_skeleton_path.string().c_str(),
                                 &character_skeleton) ||
      !ozz::sample::LoadMeshes((character_dir / "mesh.ozz").string().c_str(),
                               &meshes)) {
    return EXIT_FAILURE;
  }

  // Clip.
  const fs::path clip_path = OPTIONS_animation.value();
  const fs::path clip_skeleton_path =
      OPTIONS_clip_skeleton.value()[0] ? fs::path(OPTIONS_clip_skeleton.value())
                                       : clip_path.parent_path() / "skeleton.ozz";
  ozz::animation::Skeleton clip_skeleton;
  ozz::animation::Animation animation;
  if (!ozz::sample::LoadSkeleton(clip_skeleton_path.string().c_str(), &clip_skeleton) ||
      !ozz::sample::LoadAnimation(clip_path.string().c_str(), &animation) ||
      animation.num_tracks() != clip_skeleton.num_joints()) {
    ozz::log::Err() << "Failed to load clip or it doesn't match its skeleton." << std::endl;
    return EXIT_FAILURE;
  }

  // Retargets unless the clip is made for the character skeleton.
  std::error_code ec;
  const bool retarget = !fs::equivalent(clip_skeleton_path, character_skeleton_path, ec);
  Retargeter retargeter;
  if (retarget && !retargeter.Setup(clip_skeleton, character_skeleton)) {
    ozz::log::Err() << "Can't retarget clip onto the character." << std::endl;
    return EXIT_FAILURE;
  }

  ozz::vector<ozz::math::SoaTransform> locals(clip_skeleton.num_soa_joints());
  ozz::vector<ozz::math::Float4x4> clip_models(clip_skeleton.num_joints());
  ozz::vector<ozz::math::Float4x4> character_models(character_skeleton.num_joints());
  ozz::animation::SamplingJob::Context context(clip_skeleton.num_joints());

  hyperpose::TriMesh mesh;
  hyperpose::BuildRestMesh(meshes, &mesh);
  ozz::vector<hyperpose::SurfacePoint> samples;
  if (skinned) {
    samples = sampler(mesh, OPTIONS_count, OPTIONS_seed);
  }

  const float duration = animation.duration();
  const int num_frames = static_cast<int>(duration * OPTIONS_frame_rate) + 1;

  std::ofstream out(OPTIONS_output.value(), std::ios::binary);
  if (!out) {
    ozz::log::Err() << "Cannot write " << OPTIONS_output.value() << std::endl;
    return EXIT_FAILURE;
  }
  hyperpose::PointFieldHeader header;
  header.num_frames = num_frames;
  header.num_points = OPTIONS_count;
  header.frame_rate = OPTIONS_frame_rate;
  out.write(reinterpret_cast<const char*>(&header), sizeof(header));

  const auto start = std::chrono::steady_clock::now();
  ozz::vector<ozz::math::Float3> points;
  for (int frame = 0; frame < num_frames; ++frame) {
    const float time = std::min(frame / OPTIONS_frame_rate.value(), duration);
    const float ratio = duration > 0.f ? time / duration : 0.f;
    if (!hyperpose::SamplePose(animation, clip_skeleton, ratio, &context,
                               make_span(locals), make_span(clip_models))) {
      return EXIT_FAILURE;
    }
    if (retarget) {
      retargeter.Run(make_span(clip_models), make_span(character_models));
    }
    if (!hyperpose::PoseMesh(meshes,
                             make_span(retarget ? character_models : clip_models),
                             &mesh)) {
      ozz::log::Err() << "Failed to skin the mesh." << std::endl;
      return EXIT_FAILURE;
    }
    if (!skinned) {
      samples = sampler(mesh, OPTIONS_count, OPTIONS_seed + frame);
    }
    if (static_cast<int>(samples.size()) != OPTIONS_count) {
      ozz::log::Err() << "Sampler produced " << samples.size() << " points instead of "
                      << OPTIONS_count.value() << "." << std::endl;
      return EXIT_FAILURE;
    }
    PointFieldGenerator::Evaluate(mesh, make_span(samples), &points);
    out.write(reinterpret_cast<const char*>(points.data()),
              points.size() * sizeof(ozz::math::Float3));
  }

  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ozz::log::Out() << "Wrote " << num_frames << " frames x " << OPTIONS_count.value()
                  << " points (" << OPTIONS_method.value() << ", "
                  << OPTIONS_sampler.value() << ") in " << seconds << "s to "
                  << OPTIONS_output.value() << std::endl;
  return out ? EXIT_SUCCESS : EXIT_FAILURE;
}
