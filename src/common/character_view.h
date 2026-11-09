#pragma once

// A skinned character driven by clip poses, shared by the apps. Characters
// are folders holding skeleton.ozz + mesh.ozz. Clips made for another skeleton
// are retargeted onto the character.

#include <filesystem>
#include <string>
#include <vector>

#include "common/retarget.h"
#include "framework/mesh.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/box.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/span.h"

namespace ozz::sample {
class Renderer;
}

namespace hyperpose {

class CharacterView {
 public:
  // Finds characters below _root and selects the first one.
  void Scan(const std::filesystem::path& _root);

  // Selects a character (-1 for none). Returns false if loading failed.
  bool Select(int _index);

  // Sets the skeleton of the clips driving the character (nullptr for none),
  // and sets up retargeting when it differs from the character skeleton.
  void SetClipSkeleton(const ozz::animation::Skeleton* _skeleton,
                       const std::filesystem::path& _path);

  // The view holds no pose: every call below takes a clip pose (clip skeleton
  // joint model-space transforms), so one view can draw many characters.

  // Character skeleton joint model-space transforms for a clip pose
  // (retargeted when the skeletons differ). Returns false if not posed().
  bool Pose(ozz::span<const ozz::math::Float4x4> _clip_models,
            ozz::vector<ozz::math::Float4x4>* _character_models) const;

  // Joint bounds of the character in a clip pose, or of the clip pose itself
  // when not posed(), for camera framing.
  ozz::math::Box Bounds(ozz::span<const ozz::math::Float4x4> _clip_models,
                        const ozz::math::Float4x4& _transform) const;

  // Draws the skinned meshes in a clip pose.
  bool Draw(ozz::sample::Renderer* _renderer,
            ozz::span<const ozz::math::Float4x4> _clip_models,
            const ozz::math::Float4x4& _transform);

  // Skins the meshes in a clip pose once, into static meshes (no skinning
  // data) that DrawBaked renders without per frame skinning.
  bool Bake(ozz::span<const ozz::math::Float4x4> _clip_models,
            ozz::vector<ozz::sample::Mesh>* _baked);
  static bool DrawBaked(ozz::sample::Renderer* _renderer,
                        const ozz::vector<ozz::sample::Mesh>& _baked,
                        const ozz::math::Float4x4& _transform);

  // Character selection combo and status. Returns true on selection change.
  bool DrawGui();

  // True when a character is selected and can follow the clip skeleton.
  bool posed() const;

  const ozz::vector<ozz::sample::Mesh>& meshes() const { return meshes_; }

  // Incremented whenever the selected character changes.
  int generation() const { return generation_; }

 private:
  struct Character {
    std::string name;
    std::filesystem::path skeleton;
    std::filesystem::path mesh;
  };
  void ScanFolder(const std::filesystem::path& _path);
  bool SameSkeleton() const { return clip_skeleton_path_ == skeleton_path_; }
  void SetupRetarget();

  std::vector<Character> characters_;
  int index_ = -1;
  int generation_ = 0;
  std::string status_;

  std::filesystem::path skeleton_path_;
  ozz::animation::Skeleton skeleton_;
  ozz::vector<ozz::sample::Mesh> meshes_;
  // Scratch: character pose and skinning matrices of the pose being drawn.
  ozz::vector<ozz::math::Float4x4> models_;
  ozz::vector<ozz::math::Float4x4> skinning_matrices_;

  const ozz::animation::Skeleton* clip_skeleton_ = nullptr;
  std::filesystem::path clip_skeleton_path_;
  Retargeter retargeter_;
  bool retarget_ok_ = false;
};

}  // namespace hyperpose
