#pragma once

// Gallery of common poses: one character per common pose, in rows behind the
// origin, largest pose first. Animated, each plays its clips from weighted
// random frames of its pose; still, each shows its most typical frame, skinned
// once.

#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/clip_player.h"
#include "common/common_pose.h"
#include "framework/mesh.h"
#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/box.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/span.h"

namespace ozz::animation {
class Skeleton;
}
namespace ozz::sample {
class Renderer;
}

namespace hyperpose {

class CharacterView;

class PoseGallery {
 public:
  PoseGallery();
  ~PoseGallery();

  // Settings.
  bool animate = false;
  float spacing = 1.f;    // Between characters (m).
  float play_time = 1.f;  // Animated: play time of each picked frame (s).

  // Rebuilds the gallery for the common poses of _result, which must have
  // been found for _skeleton (that must outlive the gallery). Returns false,
  // with error() set, if a clip failed to load.
  bool Build(const CommonPoseResult& _result,
             const ozz::animation::Skeleton* _skeleton, int _heading_axis);
  void Clear();

  int size() const { return static_cast<int>(still_.size()); }
  bool empty() const { return still_.empty(); }
  const std::string& error() const { return error_; }

  // Animated: advances the clips. Still: skins _character once in each pose
  // when needed (when not done yet, or the character changed).
  void Update(float _dt, CharacterView* _character);

  // Draws every pose, as the character (if _draw_mesh and posed) and/or the
  // clip skeleton.
  bool Draw(ozz::sample::Renderer* _renderer, CharacterView* _character,
            bool _draw_mesh, bool _draw_skeleton);

  // What pose _c shows: the clip and time, and duration (negative for a still
  // frame).
  struct Shown {
    ozz::span<const ozz::math::Float4x4> models;
    const std::filesystem::path* clip;
    float time;
    float duration;
  };
  Shown shown(int _c) const;

  // Placement of pose _c.
  ozz::math::Float4x4 Transform(int _c) const;

  // Bounds of the still poses (close enough to animated ones for framing).
  ozz::math::Box Bounds() const;

 private:
  // A character playing random frames of a common pose.
  struct Slot {
    std::discrete_distribution<int> picker;  // Frame index, weighted.
    int frame = -1;
    float timer = 0.f;  // Until the next pick (s).
    ClipPlayer clip;
    ozz::vector<ozz::math::Float4x4> models;
  };

  bool PickFrame(int _c);
  bool Baked(const CharacterView& _character) const;
  void Bake(CharacterView* _character);

  CommonPoseResult result_;
  const ozz::animation::Skeleton* skeleton_ = nullptr;
  int heading_axis_ = 0;
  std::string error_;

  // Most typical frame of each pose, aligned to the origin, and the character
  // skinned in it.
  std::vector<ozz::vector<ozz::math::Float4x4>> still_;
  std::vector<ozz::vector<ozz::sample::Mesh>> baked_;
  int baked_generation_ = -1;  // Character generation of baked_.

  std::vector<std::unique_ptr<Slot>> slots_;
  std::mt19937 rng_;
};

}  // namespace hyperpose
