#pragma once

// Transition graph window (Dear ImGui): a node per common pose around a
// circle, largest first at the top, clockwise, sized by share. Arrows out of
// (orange) and into (blue) the selected pose, thicker for more frequent; with
// all_poses, the other poses' transitions too, grey. Clip ends are short
// "end" arrows pointing out of the circle. Nodes without a drawn arrow are
// dimmed.

#include <string>
#include <vector>

#include "common/core/common_pose.h"

namespace hyperpose {

class ClipLibrary;

class TransitionGraph {
 public:
  // Settings.
  bool open = true;        // Window shown.
  bool all_poses = true;   // Also the other poses' transitions, grey.
  float min_share = .05f;  // Arrows rarer than this share are hidden.

  // Draws the window (when open) for _result, with _selected highlighted (-1
  // for none); tooltips describe clips from _library. Poses _marked (per
  // pose, e.g. tagged for the walk) are ringed green, and _path (poses, e.g.
  // the walk's way to its target) drawn green. Returns the clicked pose, -1
  // if none.
  int Draw(const CommonPoseResult& _result, int _selected, const ClipLibrary& _library,
           const std::vector<bool>& _marked = {}, const std::vector<int>& _path = {});
};

}  // namespace hyperpose
