#include "common/clip_library.h"

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "common/anim_utils.h"

namespace fs = std::filesystem;

namespace hyperpose {

namespace {

// Reads "file_name,description" rows (header skipped). Descriptions may be
// quoted, with "" escaping quotes.
std::unordered_map<std::string, std::string> LoadDescriptions(
    const fs::path& _path) {
  std::unordered_map<std::string, std::string> descriptions;
  std::ifstream file(_path);
  std::string line;
  std::getline(file, line);  // Header.
  while (std::getline(file, line)) {
    const size_t comma = line.find(',');
    if (comma == std::string::npos) {
      continue;
    }
    std::string description;
    size_t i = comma + 1;
    if (i < line.size() && line[i] == '"') {
      for (++i; i < line.size(); ++i) {
        if (line[i] == '"') {
          if (i + 1 < line.size() && line[i + 1] == '"') {
            description += '"';
            ++i;
          } else {
            break;
          }
        } else {
          description += line[i];
        }
      }
    } else {
      description = line.substr(i);
    }
    descriptions[fs::path(line.substr(0, comma)).stem().string()] = description;
  }
  return descriptions;
}

}  // namespace

fs::path FindClipSkeleton(const fs::path& _clip, const fs::path& _root) {
  for (fs::path dir = _clip.parent_path(); !dir.empty();
       dir = dir.parent_path()) {
    if (fs::exists(dir / kSkeletonFileName)) {
      return dir / kSkeletonFileName;
    }
    if (dir == _root || dir == dir.parent_path()) {
      break;
    }
  }
  return {};
}

bool ClipLibrary::Scan(const fs::path& _root, const fs::path& _descriptions_csv) {
  root_ = _root;
  descriptions_ = LoadDescriptions(_descriptions_csv);
  tree_ = Folder();
  return ScanFolder(_root, &tree_);
}

// Recursively collects animation clips, skipping other archives (skeletons,
// meshes), hidden entries and folders without clips.
bool ClipLibrary::ScanFolder(const fs::path& _path, Folder* _folder) const {
  _folder->name = _path.filename().string();
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(_path, ec)) {
    const auto name = entry.path().filename().string();
    if (name.empty() || name[0] == '.') {
      continue;
    }
    if (entry.is_directory()) {
      Folder child;
      if (ScanFolder(entry.path(), &child)) {
        _folder->folders.push_back(std::move(child));
      }
    } else if (entry.path().extension() == ".ozz") {
      float duration;
      if (ReadClipDuration(entry.path().string().c_str(), &duration)) {
        const std::string stem = entry.path().stem().string();
        _folder->clips.push_back({entry.path(), stem, duration, Description(stem)});
      }
    }
  }
  std::sort(_folder->folders.begin(), _folder->folders.end(),
            [](const Folder& _a, const Folder& _b) { return _a.name < _b.name; });
  std::sort(_folder->clips.begin(), _folder->clips.end(),
            [](const Clip& _a, const Clip& _b) { return _a.name < _b.name; });
  _folder->num_clips = static_cast<int>(_folder->clips.size());
  for (const Folder& folder : _folder->folders) {
    _folder->num_clips += folder.num_clips;
  }
  return _folder->num_clips > 0;
}

const std::string& ClipLibrary::Description(const std::string& _clip_name) const {
  static const std::string kNone;
  const auto it = descriptions_.find(_clip_name);
  return it != descriptions_.end() ? it->second : kNone;
}

void ClipLibrary::DrawFilter() {
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::InputTextWithHint("##filter", "Filter clips", filter_.InputBuf,
                               IM_ARRAYSIZE(filter_.InputBuf))) {
    filter_.Build();
  }
}

void ClipLibrary::SetFilter(const char* _text) {
  std::snprintf(filter_.InputBuf, IM_ARRAYSIZE(filter_.InputBuf), "%s", _text);
  filter_.Build();
}

bool ClipLibrary::DrawTable(const std::vector<fs::path>& _highlighted,
                            fs::path* _clicked,
                            const std::vector<fs::path>* _checked,
                            fs::path* _toggled) {
  _clicked->clear();
  if (_toggled) {
    _toggled->clear();
  }
  const ImGuiTableFlags table_flags =
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
      ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;
  if (ImGui::BeginTable("clips", _checked ? 4 : 3, table_flags)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    if (_checked) {
      ImGui::TableSetupColumn("Set", ImGuiTableColumnFlags_WidthFixed,
                              ImGui::CalcTextSize("Set").x);
    }
    ImGui::TableSetupColumn("Clip", ImGuiTableColumnFlags_WidthStretch, 1.f);
    ImGui::TableSetupColumn("Duration", ImGuiTableColumnFlags_WidthFixed,
                            ImGui::CalcTextSize("Duration").x);
    ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch,
                            3.f);
    ImGui::TableHeadersRow();
    DrawFolder(tree_, root_, true, _highlighted, _clicked, _checked, _toggled);
    ImGui::EndTable();
  }
  return !_clicked->empty();
}

std::vector<fs::path> ClipLibrary::FilteredClips() const {
  std::vector<fs::path> clips;
  CollectFiltered(tree_, &clips);
  return clips;
}

void ClipLibrary::CollectFiltered(const Folder& _folder,
                                  std::vector<fs::path>* _clips) const {
  for (const auto& folder : _folder.folders) {
    CollectFiltered(folder, _clips);
  }
  for (const auto& clip : _folder.clips) {
    if (!filter_.IsActive() || Matches(clip)) {
      _clips->push_back(clip.path);
    }
  }
}

bool ClipLibrary::Matches(const Clip& _clip) const {
  return filter_.PassFilter(_clip.name.c_str()) ||
         filter_.PassFilter(_clip.description.c_str());
}

bool ClipLibrary::HasMatch(const Folder& _folder) const {
  for (const auto& clip : _folder.clips) {
    if (Matches(clip)) {
      return true;
    }
  }
  for (const auto& folder : _folder.folders) {
    if (HasMatch(folder)) {
      return true;
    }
  }
  return false;
}

void ClipLibrary::DrawFolder(const Folder& _folder, const fs::path& _path,
                             bool _is_root,
                             const std::vector<fs::path>& _highlighted,
                             fs::path* _clicked,
                             const std::vector<fs::path>* _checked,
                             fs::path* _toggled) {
  const int first = _checked ? 1 : 0;  // Name column index.
  const bool filtering = filter_.IsActive();
  if (filtering && !HasMatch(_folder)) {
    return;
  }
  if (filtering) {
    ImGui::SetNextItemOpen(true);
  }
  const ImGuiTreeNodeFlags folder_flags =
      ImGuiTreeNodeFlags_SpanAllColumns |
      (_is_root ? ImGuiTreeNodeFlags_DefaultOpen : 0);
  ImGui::TableNextRow();
  if (_checked && _is_root) {
    // Not indented yet: remembers where the checkbox column starts.
    ImGui::TableSetColumnIndex(0);
    check_column_x_ = ImGui::GetCursorScreenPos().x;
  }
  ImGui::TableSetColumnIndex(first);
  const bool open = ImGui::TreeNodeEx(_path.string().c_str(), folder_flags, "%s",
                                      _folder.name.c_str());
  ImGui::TableNextColumn();
  ImGui::TextDisabled("%d", _folder.num_clips);
  if (!open) {
    return;
  }
  for (const auto& folder : _folder.folders) {
    DrawFolder(folder, _path / folder.name, false, _highlighted, _clicked,
               _checked, _toggled);
  }

  // Clips, clipped to the visible range as folders can hold thousands.
  visible_.clear();
  for (const auto& clip : _folder.clips) {
    if (!filtering || Matches(clip)) {
      visible_.push_back(&clip);
    }
  }
  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(visible_.size()));
  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      const Clip& clip = *visible_[i];
      const bool selected =
          std::find(_highlighted.begin(), _highlighted.end(), clip.path) !=
          _highlighted.end();
      const ImGuiTreeNodeFlags flags =
          ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
          ImGuiTreeNodeFlags_SpanAllColumns |
          (_checked ? ImGuiTreeNodeFlags_AllowOverlap : 0) |
          (selected ? ImGuiTreeNodeFlags_Selected : 0);
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(first);
      ImGui::TreeNodeEx(clip.path.string().c_str(), flags, "%s",
                        clip.name.c_str());
      if (ImGui::IsItemClicked()) {
        *_clicked = clip.path;
      }
      ImGui::TableNextColumn();
      ImGui::Text("%.1fs", clip.duration);
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(clip.description.c_str());
      if (!clip.description.empty()) {
        ImGui::SetItemTooltip("%s", clip.description.c_str());
      }
      // Checkbox last, so it takes the click over the row spanning node.
      if (_checked) {
        // Tree indentation applies to every column: draw at the column start.
        ImGui::TableSetColumnIndex(0);
        ImGui::SetCursorScreenPos(
            ImVec2(check_column_x_, ImGui::GetCursorScreenPos().y));
        bool checked = std::find(_checked->begin(), _checked->end(),
                                 clip.path) != _checked->end();
        ImGui::PushID(clip.path.string().c_str());
        // No vertical padding: keeps the row at text height.
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                            ImVec2(ImGui::GetStyle().FramePadding.x, 0.f));
        if (ImGui::Checkbox("##set", &checked) && _toggled) {
          *_toggled = clip.path;
        }
        ImGui::PopStyleVar();
        ImGui::PopID();
      }
    }
  }
  ImGui::TreePop();
}

}  // namespace hyperpose
