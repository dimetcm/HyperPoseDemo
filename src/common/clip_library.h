#pragma once

// Clip library browsing, shared by the apps: a folder tree of .ozz animation
// clips with durations and descriptions, drawn as a Dear ImGui table.

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <imgui.h>

namespace hyperpose {

constexpr const char* kSkeletonFileName = "skeleton.ozz";
constexpr const char* kMeshFileName = "mesh.ozz";

// Finds the nearest skeleton.ozz, from the clip folder up to _root.
std::filesystem::path FindClipSkeleton(const std::filesystem::path& _clip,
                                       const std::filesystem::path& _root);

class ClipLibrary {
 public:
  // Scans _root for clips, with descriptions from a "file_name,description"
  // CSV (optional). Returns false if no clip was found.
  bool Scan(const std::filesystem::path& _root,
            const std::filesystem::path& _descriptions_csv);

  // Filter text box (matches clip names and descriptions).
  void DrawFilter();

  // Clip tree table, filling the remaining height. Rows in _highlighted are
  // drawn selected. Returns true and sets _clicked when a clip row is clicked.
  bool DrawTable(const std::vector<std::filesystem::path>& _highlighted,
                 std::filesystem::path* _clicked);

  // Description of a clip, by file name without extension ("" if none).
  const std::string& Description(const std::string& _clip_name) const;

  const std::filesystem::path& root() const { return root_; }

 private:
  struct Clip {
    std::filesystem::path path;
    std::string name;  // File name without extension.
    float duration;
    std::string description;
  };

  struct Folder {
    std::string name;
    std::vector<Folder> folders;
    std::vector<Clip> clips;
    int num_clips = 0;  // Including sub-folders.
  };

  bool ScanFolder(const std::filesystem::path& _path, Folder* _folder) const;
  bool Matches(const Clip& _clip) const;
  bool HasMatch(const Folder& _folder) const;
  void DrawFolder(const Folder& _folder, const std::filesystem::path& _path,
                  bool _is_root,
                  const std::vector<std::filesystem::path>& _highlighted,
                  std::filesystem::path* _clicked);

  std::filesystem::path root_;
  std::unordered_map<std::string, std::string> descriptions_;
  Folder tree_;
  ImGuiTextFilter filter_;
  std::vector<const Clip*> visible_;
};

}  // namespace hyperpose
