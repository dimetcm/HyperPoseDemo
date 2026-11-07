#include "common/retarget.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>

#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/maths/transform.h"

namespace math = ozz::math;

namespace {

// Canonical humanoid bones, with the names used by each supported rig and the
// bone used to measure the rest direction (aim). Left side only; right side is
// derived by swapping the side prefixes.
struct BoneDef {
  const char* name;
  const char* aim;
  const char* aliases[4];
};

// "%s" is replaced by the side prefix: l/r (CMU) or Left/Right (Mixamo).
const BoneDef kCenterBones[] = {
    {"Hips", nullptr, {"hip", "hips", "pelvis"}},
    {"Spine", "Chest", {"abdomen", "spine"}},
    {"Chest", "Neck", {"chest", "spine2"}},
    {"Neck", "Head", {"neck"}},
    {"Head", nullptr, {"head"}},
};

const BoneDef kSideBones[] = {
    {"Shoulder", "UpperArm", {"%sCollar", "%sShoulder"}},
    {"UpperArm", "LowerArm", {"%sShldr", "%sArm"}},
    {"LowerArm", "Hand", {"%sForeArm"}},
    {"Hand", "Middle1", {"%sHand"}},
    {"UpperLeg", "LowerLeg", {"%sThigh", "%sUpLeg"}},
    {"LowerLeg", "Foot", {"%sShin", "%sLeg"}},
    {"Foot", "Toes", {"%sFoot"}},
    {"Toes", nullptr, {"%sToe", "%sToeBase"}},
    {"Thumb1", "Thumb2", {"%sThumb1", "%sHandThumb1"}},
    {"Thumb2", nullptr, {"%sThumb2", "%sHandThumb2"}},
    {"Index1", "Index2", {"%sIndex1", "%sHandIndex1"}},
    {"Index2", nullptr, {"%sIndex2", "%sHandIndex2"}},
    {"Middle1", "Middle2", {"%sMid1", "%sHandMiddle1"}},
    {"Middle2", nullptr, {"%sMid2", "%sHandMiddle2"}},
    {"Ring1", "Ring2", {"%sRing1", "%sHandRing1"}},
    {"Ring2", nullptr, {"%sRing2", "%sHandRing2"}},
    {"Pinky1", "Pinky2", {"%sPinky1", "%sHandPinky1"}},
    {"Pinky2", nullptr, {"%sPinky2", "%sHandPinky2"}},
};

struct Canonical {
  std::string name;
  std::string aim;
  ozz::vector<std::string> aliases;  // Lower case.
};

std::string Lower(std::string _s) {
  for (char& c : _s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return _s;
}

std::string Expand(const char* _pattern, const char* _side) {
  std::string s = _pattern;
  const size_t pos = s.find("%s");
  if (pos != std::string::npos) {
    s.replace(pos, 2, _side);
  }
  return s;
}

const ozz::vector<Canonical>& CanonicalBones() {
  static const ozz::vector<Canonical> bones = [] {
    ozz::vector<Canonical> out;
    for (const BoneDef& def : kCenterBones) {
      Canonical& bone = out.emplace_back();
      bone.name = def.name;
      bone.aim = def.aim ? def.aim : "";
      for (const char* alias : def.aliases) {
        if (alias) bone.aliases.push_back(Lower(alias));
      }
    }
    const char* sides[2][3] = {{"Left", "l", "Left"}, {"Right", "r", "Right"}};
    for (const auto& side : sides) {
      for (const BoneDef& def : kSideBones) {
        Canonical& bone = out.emplace_back();
        bone.name = std::string(side[0]) + def.name;
        bone.aim = def.aim ? std::string(side[0]) + def.aim : "";
        for (const char* alias : def.aliases) {
          if (!alias) continue;
          bone.aliases.push_back(Lower(Expand(alias, side[1])));
          bone.aliases.push_back(Lower(Expand(alias, side[2])));
        }
      }
    }
    return out;
  }();
  return bones;
}

// Maps each canonical bone to a joint index (-1 if absent). Joint names are
// matched case-insensitively, ignoring namespaces such as "mixamorig5:".
ozz::vector<int> MapCanonical(const ozz::animation::Skeleton& _skeleton) {
  const auto& bones = CanonicalBones();
  ozz::vector<int> map(bones.size(), -1);
  for (int j = 0; j < _skeleton.num_joints(); ++j) {
    std::string name = _skeleton.joint_names()[j];
    const size_t colon = name.rfind(':');
    if (colon != std::string::npos) {
      name = name.substr(colon + 1);
    }
    name = Lower(name);
    for (size_t b = 0; b < bones.size(); ++b) {
      if (map[b] >= 0) continue;
      for (const auto& alias : bones[b].aliases) {
        if (alias == name) {
          map[b] = j;
          break;
        }
      }
    }
  }
  return map;
}

int FindCanonical(const std::string& _name) {
  const auto& bones = CanonicalBones();
  for (size_t b = 0; b < bones.size(); ++b) {
    if (bones[b].name == _name) return static_cast<int>(b);
  }
  return -1;
}

// Rest pose model-space rotations and translations.
void RestModel(const ozz::animation::Skeleton& _skeleton,
               ozz::vector<math::Quaternion>* _rotations,
               ozz::vector<math::Float3>* _translations) {
  ozz::vector<math::Float4x4> models(_skeleton.num_joints());
  ozz::animation::LocalToModelJob job;
  job.skeleton = &_skeleton;
  job.input = _skeleton.joint_rest_poses();
  job.output = make_span(models);
  job.Run();

  _rotations->resize(models.size());
  _translations->resize(models.size());
  for (size_t i = 0; i < models.size(); ++i) {
    math::Transform t;
    if (!math::ToAffine(models[i], &t)) {
      t = math::Transform::identity();
      math::Store3PtrU(models[i].cols[3], &t.translation.x);
    }
    (*_rotations)[i] = Normalize(t.rotation);
    (*_translations)[i] = t.translation;
  }
}

math::Float3 FlattenY(const math::Float3& _v) {
  return math::Float3(_v.x, 0.f, _v.z);
}

math::Quaternion ExtractRotation(const math::Float4x4& _m) {
  math::Transform t;
  if (!math::ToAffine(_m, &t)) {
    return math::Quaternion::identity();
  }
  return Normalize(t.rotation);
}

}  // namespace

bool Retargeter::Setup(const ozz::animation::Skeleton& _source,
                       const ozz::animation::Skeleton& _target) {
  joints_.clear();
  num_mapped_ = 0;

  const auto& bones = CanonicalBones();
  const ozz::vector<int> source_map = MapCanonical(_source);
  const ozz::vector<int> target_map = MapCanonical(_target);

  ozz::vector<math::Float3> source_positions, target_positions;
  ozz::vector<math::Quaternion> source_rotations, target_rotations;
  RestModel(_source, &source_rotations, &source_positions);
  RestModel(_target, &target_rotations, &target_positions);

  // Requires hips and both thighs to align facing and scale.
  const int hips = FindCanonical("Hips");
  const int left_leg = FindCanonical("LeftUpperLeg");
  const int right_leg = FindCanonical("RightUpperLeg");
  for (int b : {hips, left_leg, right_leg}) {
    if (source_map[b] < 0 || target_map[b] < 0) {
      return false;
    }
  }

  // Facing: rotates the source left-to-right thigh axis onto the target's.
  const math::Float3 source_side = FlattenY(
      source_positions[source_map[right_leg]] - source_positions[source_map[left_leg]]);
  const math::Float3 target_side = FlattenY(
      target_positions[target_map[right_leg]] - target_positions[target_map[left_leg]]);
  facing_ = math::Quaternion::FromVectors(source_side, target_side);

  // Proportions: hip heights above the lowest point of each rest pose.
  auto hip_height = [&](const ozz::vector<math::Float3>& _positions, int _hips) {
    float lowest = _positions[_hips].y;
    for (const auto& p : _positions) lowest = std::min(lowest, p.y);
    return _positions[_hips].y - lowest;
  };
  const float source_height = hip_height(source_positions, source_map[hips]);
  const float target_height = hip_height(target_positions, target_map[hips]);
  hips_scale_ = source_height > 1e-4f ? target_height / source_height : 1.f;

  // Target joints.
  const int num_joints = _target.num_joints();
  joints_.resize(num_joints);
  const auto parents = _target.joint_parents();
  for (int j = 0; j < num_joints; ++j) {
    Joint& joint = joints_[j];
    joint.parent = parents[j];
    if (joint.parent >= 0) {
      const math::Quaternion inv_parent = Conjugate(target_rotations[joint.parent]);
      joint.rest_local_rotation = Normalize(inv_parent * target_rotations[j]);
      joint.rest_local_translation = TransformVector(
          inv_parent, target_positions[j] - target_positions[joint.parent]);
    } else {
      joint.rest_local_rotation = target_rotations[j];
      joint.rest_local_translation = target_positions[j];
    }
  }

  const math::Quaternion inv_facing = Conjugate(facing_);
  for (size_t b = 0; b < bones.size(); ++b) {
    const int s = source_map[b], t = target_map[b];
    if (s < 0 || t < 0) {
      continue;
    }
    Joint& joint = joints_[t];
    joint.source = s;
    joint.hips = static_cast<int>(b) == hips;
    ++num_mapped_;

    // Aligns the rest bone direction, when the aim bone exists on both sides.
    math::Quaternion rest_alignment = math::Quaternion::identity();
    const int aim = bones[b].aim.empty() ? -1 : FindCanonical(bones[b].aim);
    if (aim >= 0 && source_map[aim] >= 0 && target_map[aim] >= 0) {
      const math::Float3 source_dir = TransformVector(
          facing_, source_positions[source_map[aim]] - source_positions[s]);
      const math::Float3 target_dir =
          target_positions[target_map[aim]] - target_positions[t];
      if (LengthSqr(source_dir) > 1e-8f && LengthSqr(target_dir) > 1e-8f) {
        rest_alignment = math::Quaternion::FromVectors(target_dir, source_dir);
      }
    }
    joint.offset = Normalize(Conjugate(source_rotations[s]) * inv_facing *
                             rest_alignment * target_rotations[t]);
  }

  model_rotations_.resize(num_joints);
  model_translations_.resize(num_joints);
  return num_mapped_ >= 8;
}

void Retargeter::Run(
    ozz::span<const math::Float4x4> _source_joint_model_transforms,
    ozz::span<math::Float4x4> _target_joint_model_transforms) const {
  for (size_t j = 0; j < joints_.size(); ++j) {
    const Joint& joint = joints_[j];
    const bool has_parent = joint.parent >= 0;
    const math::Quaternion parent_rotation =
        has_parent ? model_rotations_[joint.parent] : math::Quaternion::identity();
    const math::Float3 parent_translation =
        has_parent ? model_translations_[joint.parent] : math::Float3::zero();

    math::Quaternion rotation;
    if (joint.source >= 0) {
      // World-space rotation of the source bone since its rest pose, expressed
      // in the target facing, applied on top of the aligned target rest bone
      // (constant parts folded into offset, see Joint).
      rotation = Normalize(
          facing_ * ExtractRotation(_source_joint_model_transforms[joint.source]) *
          joint.offset);
    } else {
      rotation = Normalize(parent_rotation * joint.rest_local_rotation);
    }

    math::Float3 translation;
    if (joint.hips) {
      math::Float3 source_hips;
      math::Store3PtrU(_source_joint_model_transforms[joint.source].cols[3],
                       &source_hips.x);
      translation = TransformVector(facing_, source_hips * hips_scale_);
    } else {
      translation = parent_translation +
                    TransformVector(parent_rotation, joint.rest_local_translation);
    }

    model_rotations_[j] = rotation;
    model_translations_[j] = translation;
    _target_joint_model_transforms[j] = math::Float4x4::FromAffine(
        math::simd_float4::Load3PtrU(&translation.x),
        math::simd_float4::LoadPtrU(&rotation.x), math::simd_float4::one());
  }
}
