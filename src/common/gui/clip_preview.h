#pragma once

// The main character's clip preview: a clip played or scrubbed, with optional
// root motion and playback limited to the trim range of its pose set entry,
// and its playback and trim controls (Dear ImGui).

#include "common/core/clip_player.h"
#include "common/core/common_pose.h"
#include "common/core/pose_match.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/span.h"

namespace hyperpose {

class ClipPreview {
 public:
  ClipPlayer clip;         // Also shows still frames while not active.
  bool active = false;     // A clip is being previewed.
  bool play_trim = true;   // Plays only the trim range of a set clip.

  // Root motion: places the clip so that its root at _time stands at the
  // origin facing the default heading (_character_root nullptr), or where
  // _character_root (model-space root of the last pose) stands.
  void Anchor(float _time, const ozz::math::Float4x4* _character_root, int _heading_axis);
  // Moves model-space joint transforms of the clip to its anchor.
  void ApplyAnchor(ozz::span<ozz::math::Float4x4> _models) const;

  // Advances playback by _dt, wrapping (looping) or stopping at the ends of
  // _entry's trim range when play_trim (_entry nullptr: not in the set).
  // Returns true if playback wrapped around (clip loop or trim range).
  bool Advance(float _dt, const PoseSetEntry* _entry);

  struct GuiResult {
    bool close = false;       // Close pressed.
    bool add_to_set = false;  // Add to pose set pressed.
  };
  // Playback and trim controls of the previewed clip. _entry: its pose set
  // entry, trimmed here (nullptr: not in the set, offers to add it); _busy:
  // trims and adding locked.
  GuiResult Gui(PoseSetEntry* _entry, bool _busy);

 private:
  RootPlacement placement_;  // Root motion anchor.
};

}  // namespace hyperpose
