#include "common/gui/project_bar.h"

#include <algorithm>
#include <cstdio>
#include <string>

#include <imgui.h>

#include "common/core/pose_project.h"

namespace fs = std::filesystem;

namespace hyperpose {

ProjectBar::Action ProjectBar::Draw(const fs::path& _projects_dir, const fs::path& _path,
                                    bool _dirty, bool _busy) {
  Action action;
  const std::string name = _path.empty() ? "Untitled" : _path.stem().string();
  ImGui::AlignTextToFramePadding();
  ImGui::Text("Project: %s%s", name.c_str(), _dirty ? "*" : "");
  if (!_path.empty()) {
    ImGui::SetItemTooltip("%s", _path.string().c_str());
  }
  ImGui::BeginDisabled(_busy);
  const float right = ImGui::GetContentRegionMax().x;
  const ImGuiStyle& style = ImGui::GetStyle();
  const float buttons = ImGui::CalcTextSize("NewOpen...SaveSave as...").x +
                        style.FramePadding.x * 8.f + style.ItemSpacing.x * 3.f;
  ImGui::SameLine(std::max(ImGui::GetCursorPosX(), right - buttons));
  bool open_save_as = false;
  if (ImGui::Button("New")) {
    pending_open_.clear();
    pending_new_ = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Open...")) {
    list_ = ListPoseProjects(_projects_dir);
    ImGui::OpenPopup("Open project");
  }
  ImGui::SameLine();
  const bool save = ImGui::Button("Save") ||
                    ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S, ImGuiInputFlags_RouteGlobal);
  if (save) {
    if (_path.empty()) {
      open_save_as = true;
    } else {
      action = {Action::kSave, _path};
    }
  }
  ImGui::SetItemTooltip("Ctrl+S");
  ImGui::SameLine();
  if (ImGui::Button("Save as...")) {
    open_save_as = true;
  }
  ImGui::EndDisabled();

  // Open: projects of the projects folder, most recent first.
  if (ImGui::BeginPopup("Open project")) {
    ImGui::TextDisabled("%s", _projects_dir.string().c_str());
    if (list_.empty()) {
      ImGui::TextUnformatted("No projects yet: Save as... to make one.");
    }
    for (const fs::path& path : list_) {
      if (ImGui::Selectable(path.stem().string().c_str(), path == _path)) {
        pending_open_ = path;
      }
    }
    ImGui::EndPopup();
  }

  // Save as: a name in the projects folder.
  if (open_save_as) {
    std::snprintf(save_name_, sizeof(save_name_), "%s",
                  _path.empty() ? "" : name.c_str());
    ImGui::OpenPopup("Save project as");
  }
  if (ImGui::BeginPopupModal("Save project as", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    if (ImGui::IsWindowAppearing()) {
      ImGui::SetKeyboardFocusHere();
    }
    const bool enter = ImGui::InputText("Name", save_name_, sizeof(save_name_),
                                        ImGuiInputTextFlags_EnterReturnsTrue);
    const fs::path path = _projects_dir / (std::string(save_name_) + kProjectExtension);
    std::error_code ec;
    const bool valid = save_name_[0] && fs::path(save_name_).filename() == fs::path(save_name_);
    if (valid && fs::exists(path, ec) && path != _path) {
      ImGui::TextColored(ImVec4(1.f, .7f, .3f, 1.f), "Replaces the existing %s.", save_name_);
    }
    ImGui::BeginDisabled(!valid);
    if (ImGui::Button("Save") || (enter && valid)) {
      action = {Action::kSave, path};
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  // New / Open, confirmed when that drops changes.
  if (pending_new_ || !pending_open_.empty()) {
    if (!_dirty) {
      action = TakePending();
    } else if (!ImGui::IsPopupOpen("Discard changes?")) {
      ImGui::OpenPopup("Discard changes?");
    }
  }
  if (ImGui::BeginPopupModal("Discard changes?", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("%s has unsaved changes.", name.c_str());
    if (ImGui::Button("Discard")) {
      action = TakePending();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      pending_new_ = false;
      pending_open_.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  return action;
}

ProjectBar::Action ProjectBar::TakePending() {
  Action action;
  if (pending_new_) {
    action.type = Action::kNew;
  } else if (!pending_open_.empty()) {
    action = {Action::kOpen, pending_open_};
  }
  pending_new_ = false;
  pending_open_.clear();
  return action;
}

}  // namespace hyperpose
