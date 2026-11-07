// ozz_fixup: post-processes raw ozz archives produced by fbx2ozz ("raw": true).
//
// - Applies a global scale (e.g. 0.01 for cm-authored data) and bakes uniform
//   joint scales into child translations, so the output has unit joint scales
//   and real-world (meter) translations. Fixes Blender exports whose armature
//   root carries a x100 scale.
// - Optimizes and builds runtime archives.
//
// Usage:
//   ozz_fixup --skeleton=raw_skeleton.ozz --scale=0.01 --skeleton_output=skeleton.ozz
//   ozz_fixup --skeleton=raw_skeleton.ozz --scale=0.01 --animation=raw_anim.ozz --animation_output=anim.ozz

#include <cmath>
#include <cstring>

#include "ozz/animation/offline/animation_builder.h"
#include "ozz/animation/offline/animation_optimizer.h"
#include "ozz/animation/offline/raw_animation.h"
#include "ozz/animation/offline/raw_skeleton.h"
#include "ozz/animation/offline/skeleton_builder.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/containers/map.h"
#include "ozz/base/containers/string.h"
#include "ozz/base/io/archive.h"
#include "ozz/base/io/stream.h"
#include "ozz/base/log.h"
#include "ozz/options/options.h"

OZZ_OPTIONS_DECLARE_STRING(skeleton, "Input raw skeleton archive.", "", true)
OZZ_OPTIONS_DECLARE_STRING(skeleton_output, "Output runtime skeleton archive.", "", false)
OZZ_OPTIONS_DECLARE_STRING(animation, "Input raw animation archive.", "", false)
OZZ_OPTIONS_DECLARE_STRING(animation_output, "Output runtime animation archive.", "", false)
OZZ_OPTIONS_DECLARE_FLOAT(scale, "Global scale applied to the data.", 1.f, false)
OZZ_OPTIONS_DECLARE_BOOL(optimize, "Optimizes animation keyframes.", true, false)
OZZ_OPTIONS_DECLARE_FLOAT(tolerance, "Optimization tolerance (meters).", 1e-3f, false)
OZZ_OPTIONS_DECLARE_FLOAT(distance, "Optimization error distance (meters).", 1e-1f, false)

namespace offline = ozz::animation::offline;

namespace {

// Per-joint scale baking factors.
struct JointFactors {
  float parent = 1.f;  // Accumulated uniform scale of the parent chain.
  float self = 1.f;    // Joint's own uniform rest scale, baked out.
};
using FactorMap = ozz::map<ozz::string, JointFactors>;

bool IsUniform(const ozz::math::Float3& _s) {
  return std::fabs(_s.x - _s.y) < 1e-4f * std::fabs(_s.x) &&
         std::fabs(_s.x - _s.z) < 1e-4f * std::fabs(_s.x) && _s.x != 0.f;
}

// Bakes uniform scales down the hierarchy: T(t)R S(k) under parent scale a
// becomes T(a*t)R S(1), with children receiving a*k.
void BakeJoints(offline::RawSkeleton::Joint::Children& _joints, float _parent,
                FactorMap* _factors) {
  for (auto& joint : _joints) {
    JointFactors& f = (*_factors)[joint.name];
    f.parent = _parent;
    f.self = IsUniform(joint.transform.scale) ? joint.transform.scale.x : 1.f;
    joint.transform.translation = joint.transform.translation * _parent;
    joint.transform.scale = joint.transform.scale * (1.f / f.self);
    BakeJoints(joint.children, _parent * f.self, _factors);
  }
}

template <typename _Type>
bool Load(const char* _path, _Type* _object) {
  ozz::io::File file(_path, "rb");
  if (!file.opened()) {
    ozz::log::Err() << "Cannot open " << _path << std::endl;
    return false;
  }
  ozz::io::IArchive archive(&file);
  if (!archive.TestTag<_Type>()) {
    ozz::log::Err() << _path << " doesn't contain the expected raw archive type."
                    << std::endl;
    return false;
  }
  archive >> *_object;
  return true;
}

template <typename _Type>
bool Save(const char* _path, const _Type& _object) {
  ozz::io::File file(_path, "wb");
  if (!file.opened()) {
    ozz::log::Err() << "Cannot write " << _path << std::endl;
    return false;
  }
  ozz::io::OArchive archive(&file);
  archive << _object;
  return true;
}

}  // namespace

int main(int _argc, const char** _argv) {
  if (ozz::options::ParseCommandLine(_argc, _argv, "1.0",
                                     "Rescales and builds raw ozz archives.") !=
      ozz::options::kSuccess) {
    return EXIT_FAILURE;
  }

  offline::RawSkeleton raw_skeleton;
  if (!Load(OPTIONS_skeleton, &raw_skeleton)) {
    return EXIT_FAILURE;
  }
  FactorMap factors;
  BakeJoints(raw_skeleton.roots, OPTIONS_scale, &factors);

  auto skeleton = offline::SkeletonBuilder()(raw_skeleton);
  if (!skeleton) {
    ozz::log::Err() << "Failed to build skeleton." << std::endl;
    return EXIT_FAILURE;
  }
  if (OPTIONS_skeleton_output.value()[0] &&
      !Save(OPTIONS_skeleton_output, *skeleton)) {
    return EXIT_FAILURE;
  }

  if (!OPTIONS_animation.value()[0]) {
    return EXIT_SUCCESS;
  }
  if (!OPTIONS_animation_output.value()[0]) {
    ozz::log::Err() << "--animation_output is required with --animation." << std::endl;
    return EXIT_FAILURE;
  }

  offline::RawAnimation raw_animation;
  if (!Load(OPTIONS_animation, &raw_animation)) {
    return EXIT_FAILURE;
  }
  if (raw_animation.num_tracks() != skeleton->num_joints()) {
    ozz::log::Err() << "Animation has " << raw_animation.num_tracks()
                    << " tracks, skeleton has " << skeleton->num_joints()
                    << " joints." << std::endl;
    return EXIT_FAILURE;
  }

  // Tracks are ordered like the runtime skeleton joints.
  for (int i = 0; i < skeleton->num_joints(); ++i) {
    const JointFactors& f = factors[skeleton->joint_names()[i]];
    auto& track = raw_animation.tracks[i];
    for (auto& key : track.translations) {
      key.value = key.value * f.parent;
    }
    for (auto& key : track.scales) {
      key.value = key.value * (1.f / f.self);
    }
  }

  if (OPTIONS_optimize) {
    offline::AnimationOptimizer optimizer;
    optimizer.setting.tolerance = OPTIONS_tolerance;
    optimizer.setting.distance = OPTIONS_distance;
    offline::RawAnimation optimized;
    if (!optimizer(raw_animation, *skeleton, &optimized)) {
      ozz::log::Err() << "Failed to optimize animation." << std::endl;
      return EXIT_FAILURE;
    }
    raw_animation = std::move(optimized);
  }

  auto animation = offline::AnimationBuilder()(raw_animation);
  if (!animation) {
    ozz::log::Err() << "Failed to build animation." << std::endl;
    return EXIT_FAILURE;
  }
  return Save(OPTIONS_animation_output, *animation) ? EXIT_SUCCESS : EXIT_FAILURE;
}
