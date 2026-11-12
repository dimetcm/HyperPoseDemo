#pragma once

// Project bar (Dear ImGui): the project name (* when changed) and New /
// Open... / Save (Ctrl+S) / Save as..., with a popup listing the projects
// folder, a name dialog for Save as, and a confirmation before New or Open
// drop unsaved changes. It only asks: the caller carries out the action.

#include <filesystem>
#include <vector>

namespace hyperpose {

class ProjectBar {
 public:
  struct Action {
    enum Type { kNone, kNew, kOpen, kSave };
    Type type = kNone;
    std::filesystem::path path;  // kOpen: the project, kSave: where to.
  };

  // Draws the bar for the project file _path (empty: untitled) in
  // _projects_dir; _dirty: it has unsaved changes; _busy: buttons disabled.
  // Returns what the user asked for this frame.
  Action Draw(const std::filesystem::path& _projects_dir,
              const std::filesystem::path& _path, bool _dirty, bool _busy);

 private:
  // The confirmed New / Open, cleared.
  Action TakePending();

  std::vector<std::filesystem::path> list_;  // Open popup.
  char save_name_[128] = {};
  bool pending_new_ = false;
  std::filesystem::path pending_open_;
};

}  // namespace hyperpose
