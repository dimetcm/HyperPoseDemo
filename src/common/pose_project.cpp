#include "common/pose_project.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace hyperpose {
namespace {

const int kVersion = 1;

fs::path SettingsFolder() {
  const char* local = std::getenv("LOCALAPPDATA");
  return local ? fs::path(local) / "HyperPoseDemo" : fs::path();
}

}  // namespace

bool SavePoseProject(const fs::path& _path, const PoseProject& _project,
                     const fs::path& _clip_root, std::string* _error) {
  std::error_code ec;
  if (_path.has_parent_path()) {
    fs::create_directories(_path.parent_path(), ec);
  }
  // Written aside then swapped in, so a failed save keeps the old file.
  const fs::path temp = fs::path(_path).concat(".tmp");
  {
    std::ofstream file(temp);
    if (!file) {
      *_error = "Can't write " + temp.string();
      return false;
    }
    file << "# HyperPoseDemo project\n"
         << "version " << kVersion << "\n"
         << "clusters " << _project.settings.clusters << "\n"
         << "stride " << _project.settings.stride << "\n"
         << "seed " << _project.settings.seed << "\n"
         << "velocity_weight " << _project.settings.velocity_weight << "\n";
    for (const PoseSetEntry& clip : _project.clips) {
      // Relative to the clip root, so projects survive moving the assets.
      fs::path path = clip.path.lexically_relative(_clip_root);
      if (path.empty() || *path.begin() == "..") {
        path = clip.path;
      }
      file << "clip " << clip.begin << " " << clip.end << " "
           << path.generic_string() << "\n";
    }
    if (!file) {
      *_error = "Failed writing " + temp.string();
      return false;
    }
  }
  fs::rename(temp, _path, ec);
  if (ec) {
    *_error = "Can't replace " + _path.string() + ": " + ec.message();
    fs::remove(temp, ec);
    return false;
  }
  return true;
}

bool LoadPoseProject(const fs::path& _path, const fs::path& _clip_root,
                     PoseProject* _project, std::string* _error) {
  std::ifstream file(_path);
  if (!file) {
    *_error = "Can't open " + _path.string();
    return false;
  }
  PoseProject project;
  std::string line;
  int number = 0;
  while (std::getline(file, line)) {
    ++number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream in(line);
    std::string key;
    in >> key;
    bool ok = true;
    if (key == "version") {
      int version = 0;
      ok = static_cast<bool>(in >> version);
      if (ok && version > kVersion) {
        *_error = _path.filename().string() + " is from a newer version.";
        return false;
      }
    } else if (key == "clusters") {
      ok = static_cast<bool>(in >> project.settings.clusters);
    } else if (key == "stride") {
      ok = static_cast<bool>(in >> project.settings.stride);
    } else if (key == "rate") {
      // Older projects: frames sampled per second, from 30 Hz clips.
      float rate = 0.f;
      ok = static_cast<bool>(in >> rate) && rate > 0.f;
      project.settings.stride = std::max(1, static_cast<int>(std::lround(30.f / rate)));
    } else if (key == "seed") {
      ok = static_cast<bool>(in >> project.settings.seed);
    } else if (key == "velocity_weight") {
      ok = static_cast<bool>(in >> project.settings.velocity_weight);
    } else if (key == "clip") {
      PoseSetEntry clip;
      ok = static_cast<bool>(in >> clip.begin >> clip.end);
      std::string path;
      std::getline(in >> std::ws, path);  // The rest: may hold spaces.
      ok = ok && !path.empty();
      clip.path = fs::path(path).make_preferred();
      if (clip.path.is_relative()) {
        clip.path = _clip_root / clip.path;
      }
      project.clips.push_back(clip);
    }  // Unknown keys are skipped, for newer minor additions.
    if (!ok) {
      *_error = _path.filename().string() + ": bad line " + std::to_string(number);
      return false;
    }
  }
  project.settings = project.settings.Clamped();
  *_project = std::move(project);
  return true;
}

std::vector<fs::path> ListPoseProjects(const fs::path& _folder) {
  std::vector<std::pair<fs::file_time_type, fs::path>> found;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(_folder, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == kProjectExtension) {
      found.push_back({entry.last_write_time(ec), entry.path()});
    }
  }
  std::sort(found.begin(), found.end(),
            [](const auto& _a, const auto& _b) { return _a.first > _b.first; });
  std::vector<fs::path> paths;
  for (auto& [time, path] : found) {
    paths.push_back(std::move(path));
  }
  return paths;
}

fs::path RecentPoseProject() {
  const fs::path folder = SettingsFolder();
  if (folder.empty()) {
    return {};
  }
  std::ifstream file(folder / "recent_project.txt");
  std::string line;
  if (!std::getline(file, line) || line.empty()) {
    return {};
  }
  const fs::path path = fs::u8path(line);
  std::error_code ec;
  return fs::exists(path, ec) ? path : fs::path();
}

void SetRecentPoseProject(const fs::path& _path) {
  const fs::path folder = SettingsFolder();
  if (folder.empty()) {
    return;
  }
  std::error_code ec;
  fs::create_directories(folder, ec);
  std::ofstream file(folder / "recent_project.txt");
  file << fs::absolute(_path, ec).u8string() << "\n";
}

}  // namespace hyperpose
