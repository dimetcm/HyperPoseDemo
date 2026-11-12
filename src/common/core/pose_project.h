#pragma once

// Pose database projects: the clips selected for the database (with the
// time range used from each, for trimming) and the common pose settings.
// Saved as a line based text file (.hpproject):
//
//   # HyperPoseDemo project
//   version 1
//   clusters 12
//   stride 1
//   seed 1
//   velocity_weight 0.1
//   clip <begin> <end> <path>
//
// In the file, clip paths are relative to the clip root (absolute if outside
// it), with '/' separators; in memory, they're full paths. end < 0 means the
// clip end.

#include <filesystem>
#include <string>
#include <vector>

#include "common/core/common_pose.h"

namespace hyperpose {

// A pose set and the settings to find its common poses with.
struct PoseProject {
  std::vector<PoseSetEntry> clips;  // Durations aren't saved.
  CommonPoseSettings settings;

  bool operator==(const PoseProject& _o) const {
    return clips == _o.clips && settings == _o.settings;
  }
  bool operator!=(const PoseProject& _o) const { return !(*this == _o); }
};

constexpr const char* kProjectExtension = ".hpproject";

// Saves _project to _path, with clip paths relative to _clip_root.
bool SavePoseProject(const std::filesystem::path& _path,
                     const PoseProject& _project,
                     const std::filesystem::path& _clip_root, std::string* _error);
// Loads _project from _path, resolving relative clip paths against
// _clip_root. Settings are clamped to their ranges; clip durations are left
// at 0 (see PoseSetEntry).
bool LoadPoseProject(const std::filesystem::path& _path,
                     const std::filesystem::path& _clip_root,
                     PoseProject* _project, std::string* _error);

// Project files in _folder, most recently modified first.
std::vector<std::filesystem::path> ListPoseProjects(
    const std::filesystem::path& _folder);

// Most recently opened / saved project (per user, kept in
// %LOCALAPPDATA%/HyperPoseDemo), empty if none or gone.
std::filesystem::path RecentPoseProject();
void SetRecentPoseProject(const std::filesystem::path& _path);

}  // namespace hyperpose
