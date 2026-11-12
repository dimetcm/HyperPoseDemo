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
// Clip paths are relative to the clip root (absolute if outside it), with
// '/' separators; end < 0 means the clip end.

#include <filesystem>
#include <string>
#include <vector>

namespace hyperpose {

struct PoseProject {
  struct Clip {
    std::filesystem::path path;
    float begin = 0.f;
    float end = -1.f;
    bool operator==(const Clip& _o) const {
      return path == _o.path && begin == _o.begin && end == _o.end;
    }
  };
  std::vector<Clip> clips;
  int clusters = 12;
  int stride = 1;  // Clusters every stride-th frame of the clips.
  int seed = 1;
  // Joint velocities against positions when clustering (seconds, 0: off).
  float velocity_weight = .1f;

  bool operator==(const PoseProject& _o) const {
    return clips == _o.clips && clusters == _o.clusters && stride == _o.stride &&
           seed == _o.seed && velocity_weight == _o.velocity_weight;
  }
  bool operator!=(const PoseProject& _o) const { return !(*this == _o); }
};

constexpr const char* kProjectExtension = ".hpproject";

bool SavePoseProject(const std::filesystem::path& _path,
                     const PoseProject& _project, std::string* _error);
bool LoadPoseProject(const std::filesystem::path& _path,
                     PoseProject* _project, std::string* _error);

// Project files in _folder, most recently modified first.
std::vector<std::filesystem::path> ListPoseProjects(
    const std::filesystem::path& _folder);

// Most recently opened / saved project (per user, kept in
// %LOCALAPPDATA%/HyperPoseDemo), empty if none or gone.
std::filesystem::path RecentPoseProject();
void SetRecentPoseProject(const std::filesystem::path& _path);

}  // namespace hyperpose
