#pragma once

// Gallery of common poses: one slot per common pose, in rows behind the
// origin, largest pose first. Still, each shows its most typical frame,
// skinned once; random frames, each plays its clips from weighted random
// frames of its pose; clips, each overlays every stay of a clip in its pose
// (from where the clip enters it to where it leaves), in sync: each played
// around its frame closest to the pose, over the same window, looping.

#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/core/clip_player.h"
#include "common/core/common_pose.h"
#include "common/gui/character_view.h"
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

  enum class Mode { kStill, kFrames, kClips };

  // Settings.
  Mode mode = Mode::kStill;
  float spacing = 1.f;    // Between characters (m).
  float play_time = 1.f;  // Frames: play time of each picked frame (s).
  float speed = .2f;      // Animated: clips' playback speed (time scale).
  // Clips: played before and after each stay's frame closest to the pose
  // (s, clip time).
  float sync_window = .5f;

  bool animated() const { return mode != Mode::kStill; }

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
  // frame). Clips: the first stay, and how many are overlaid.
  struct Shown {
    ozz::span<const ozz::math::Float4x4> models;
    const std::filesystem::path* clip;
    float time;
    float duration;
    int count = 1;
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

  // A clip's stay in a pose, played around anchor (clips mode).
  struct Stay {
    int clip = -1;  // Index in result_.clips.
    float anchor = 0.f;  // Its frame closest to the pose (s).
    float time = 0.f;    // Playing.
    ozz::vector<ozz::math::Float4x4> models;
  };

  bool PickFrame(int _c);
  // Where clip _clip's motion starts (s): after a leading bind pose, if any
  // (see PoseFeatures::first_frame).
  float MotionStart(int _clip) const;
  // Clips mode: the stays of every pose, and their clips loaded, on first
  // use.
  bool BuildStays();
  void UpdateStays(float _dt);
  bool Baked(const CharacterView& _character) const;
  void Bake(CharacterView* _character);

  CommonPoseResult result_;
  const ozz::animation::Skeleton* skeleton_ = nullptr;
  int heading_axis_ = 0;
  std::string error_;

  // Most typical frame of each pose, aligned to the origin, and the character
  // skinned in it.
  std::vector<ozz::vector<ozz::math::Float4x4>> still_;
  std::vector<CharacterView::Baked> baked_;  // Per still pose.
  int baked_generation_ = -1;  // Character generation of baked_.

  std::vector<std::unique_ptr<Slot>> slots_;
  std::vector<std::vector<Stay>> stays_;  // Per pose.
  std::vector<std::unique_ptr<ClipPlayer>> players_;  // Per clip, shared by stays.
  bool stays_built_ = false;
  float phase_ = 0.f;  // Shared by the stays: time from their anchors (s).
  std::mt19937 rng_;
};

}  // namespace hyperpose
