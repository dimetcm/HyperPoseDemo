// Common Poses: finds the most common poses in a set of clips. Every frame of
// the selected clips is turned into a pose feature (root relative, heading
// aligned joint positions), then grouped with k-means. Clusters are listed by
// size; each shows its most typical frame, and its members can be browsed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

#include <imgui.h>

#include "common/anim_utils.h"
#include "common/character_view.h"
#include "common/clip_library.h"
#include "common/clip_player.h"
#include "common/common_pose.h"
#include "common/pose_gallery.h"
#include "common/pose_project.h"
#include "common/pose_walk.h"
#include "common/pose_match.h"
#include "framework/application.h"
#include "framework/imgui.h"
#include "framework/renderer.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/log.h"
#include "ozz/base/maths/box.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/options/options.h"

namespace fs = std::filesystem;
namespace math = ozz::math;

OZZ_OPTIONS_DECLARE_STRING(root, "Folder to browse for .ozz animations.",
                           HYPERPOSE_ASSETS_DIR "/ozz", false)
OZZ_OPTIONS_DECLARE_STRING(
    metadata, "CSV of clip descriptions (file_name,description).",
    HYPERPOSE_ASSETS_DIR "/HuggingFace/gbionics__cmu-fbx/metadata.csv", false)
OZZ_OPTIONS_DECLARE_STRING(
    auto_set, "Filter text: adds matching clips and finds common poses at startup.",
    "", false)
OZZ_OPTIONS_DECLARE_STRING(clip, "Clip to preview at startup (path).", "",
                           false)
OZZ_OPTIONS_DECLARE_BOOL(root_motion, "Root motion for the preview and pose walk.", false, false)
OZZ_OPTIONS_DECLARE_STRING(projects, "Folder of project files.",
                           HYPERPOSE_PROJECTS_DIR, false)
OZZ_OPTIONS_DECLARE_STRING(
    project, "Project to open at startup (default: the most recent one).", "", false)
OZZ_OPTIONS_DECLARE_BOOL(walk, "Starts the pose walk once common poses are found.",
                         false, false)

namespace {

using hyperpose::CommonPose;
using hyperpose::PoseSetEntry;

// Frame rate the clips were converted at (Tools/convert_fbx.py), for frame
// numbers in debug text.
const float kClipFrameRate = 30.f;

const float kRates[] = {5.f, 10.f, 15.f};
const char* kRateLabels[] = {"5 Hz", "10 Hz", "15 Hz"};

}  // namespace

class CommonPosesApplication : public ozz::sample::Application {
 protected:
  bool OnInitialize() override {
    if (!library_.Scan(OPTIONS_root.value(), OPTIONS_metadata.value())) {
      status_ = "No .ozz clips found in " + std::string(OPTIONS_root.value());
    }
    character_.Scan(OPTIONS_root.value());
    projects_dir_ = OPTIONS_projects.value();
    NewProject();
    // A given project, else (unless building a set from a filter) the most
    // recent one.
    if (OPTIONS_project.value()[0]) {
      OpenProject(OPTIONS_project.value());
    } else if (!OPTIONS_auto_set.value()[0]) {
      const fs::path recent = hyperpose::RecentPoseProject();
      if (!recent.empty()) {
        OpenProject(recent);
      }
    }
    if (OPTIONS_auto_set.value()[0]) {
      library_.SetFilter(OPTIONS_auto_set);
      for (const fs::path& clip : library_.FilteredClips()) {
        if (!Add(clip)) break;
      }
      StartClustering();
    }
    walk_on_result_ = OPTIONS_walk;
    walk_.root_motion = OPTIONS_root_motion;
    if (OPTIONS_clip.value()[0]) {
      StartPreview(OPTIONS_clip.value(), 0.f);
    }
    return true;
  }

  bool OnUpdate(float _dt, float) override {
    // Picks up the clustering result when the worker is done.
    if (job_.valid() &&
        job_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      walk_.Stop();  // Walks the previous result.
      result_ = job_.get();
      status_ = result_.error;
      LogResult();
      if (!gallery_.Build(result_, &skeleton(), clip_skeleton_.heading_axis())) {
        status_ = gallery_.error();
      }
      if (previewing_) {
        // Selects the first pose without interrupting the preview.
        pose_ = result_.poses.empty() ? -1 : 0;
        frame_ = 0;
        transition_example_.clear();
      } else {
        SelectPose(result_.poses.empty() ? -1 : 0);
      }
      if (walk_on_result_ && !result_.poses.empty()) {
        StartWalk();  // First result only.
        walk_on_result_ = false;
      }
    }
    if (previewing_) {
      UpdatePreview(_dt);
    }
    if (walk_.active()) {
      UpdateWalk(_dt);
    }
    if (show_gallery_) {
      gallery_.Update(_dt, &character_);
    }
    return true;
  }

  bool OnDisplay(ozz::sample::Renderer* _renderer) override {
    bool success = true;
    // Debug labels are placed here, while the camera is set for the 3D view,
    // and drawn with Dear ImGui later in the frame.
    labels_.clear();
    if (pose_valid_) {
      success &= DrawPose(_renderer, make_span(models_), math::Float4x4::identity());
      if (walk_.active()) {
        const hyperpose::PoseWalk::Segment& segment = walk_.segment();
        AddLabel(math::Float4x4::identity(), make_span(models_),
                 (segment.jump ? "jump #" : "#") + std::to_string(segment.from + 1) +
                     " -> " + PoseName(segment.to) + "\n",
                 result_.clips[segment.clip], walk_.clip_time(), walk_.clip_duration());
      } else {
        AddLabel(math::Float4x4::identity(), make_span(models_), "", clip_.path(),
                 clip_.time, clip_.duration());
      }
    }
    // Every common pose, in rows behind the displayed pose.
    if (show_gallery_ && !gallery_.empty()) {
      success &= gallery_.Draw(_renderer, &character_, draw_mesh_, draw_skeleton_);
      for (int c = 0; c < gallery_.size(); ++c) {
        const hyperpose::PoseGallery::Shown pose = gallery_.shown(c);
        AddLabel(gallery_.Transform(c), pose.models,
                 "#" + std::to_string(c + 1) + " ", *pose.clip, pose.time,
                 pose.duration);
      }
    }
    return success;
  }


  // Queues a debug label above a pose: clip, time / duration (if known, >= 0)
  // and frame number at the clips frame rate.
  void AddLabel(const math::Float4x4& _transform,
                ozz::span<const math::Float4x4> _models, const std::string& _prefix,
                const fs::path& _clip, float _time, float _duration) {
    if (!debug_text_ || _models.empty() || _clip.empty()) {
      return;
    }
    // Anchor above the highest joint, at the root's ground position.
    float top = -std::numeric_limits<float>::max();
    for (const math::Float4x4& model : _models) {
      top = std::max(top, math::GetY(model.cols[3]));
    }
    math::Float3 root;
    math::Store3PtrU(_models[0].cols[3], &root.x);
    math::Float3 anchor;
    math::Store3PtrU(TransformPoint(_transform, math::simd_float4::Load(
                                                    root.x, top + .25f, root.z, 1.f)),
                     &anchor.x);

    char text[256];
    const int frame = static_cast<int>(_time * kClipFrameRate + .5f);
    if (_duration >= 0.f) {
      std::snprintf(text, sizeof(text), "%s%s\n%.1f/%.1fs\nf %d/%d",
                    _prefix.c_str(), _clip.stem().string().c_str(), _time,
                    _duration, frame,
                    static_cast<int>(_duration * kClipFrameRate + .5f));
    } else {
      std::snprintf(text, sizeof(text), "%s%s\n%.1fs\nf %d", _prefix.c_str(),
                    _clip.stem().string().c_str(), _time, frame);
    }
    AddText(anchor, text);
  }

  // Queues a text centered above a world point, drawn by DrawLabels.
  void AddText(const math::Float3& _anchor, const std::string& _text) {
    // WorldToScreen maps the projection to the whole window, bottom up; the
    // 3D view is the part right of the panel.
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    const math::Float2 screen = WorldToScreen(_anchor);
    const float ndc_x = 2.f * screen.x / display.x - 1.f;
    const ImVec2 pos(gui_left_inset_ + (display.x - gui_left_inset_) * (ndc_x + 1.f) * .5f,
                     display.y - screen.y);
    labels_.push_back({pos, _text});
  }

  // Draws the queued debug labels over the 3D view, centered on their anchor.
  void DrawLabels() {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImFont* font = ImGui::GetFont();
    const float font_size = ImGui::GetFontSize() * .8f;
    for (const Label& label : labels_) {
      const ImVec2 size =
          font->CalcTextSizeA(font_size, FLT_MAX, 0.f, label.text.c_str());
      const ImVec2 pos(label.pos.x - size.x * .5f, label.pos.y - size.y);
      if (pos.x + size.x < gui_left_inset_) {
        continue;  // Behind the panel.
      }
      draw_list->AddRectFilled(ImVec2(pos.x - 3.f, pos.y - 2.f),
                               ImVec2(pos.x + size.x + 3.f, pos.y + size.y + 2.f),
                               IM_COL32(0, 0, 0, 140), 3.f);
      draw_list->AddText(font, font_size, pos, IM_COL32(255, 255, 255, 255),
                         label.text.c_str());
    }
  }

  // Draws a clip skeleton pose, as the character (if shown) and/or skeleton.
  bool DrawPose(ozz::sample::Renderer* _renderer,
                ozz::span<const math::Float4x4> _models,
                const math::Float4x4& _transform) {
    bool success = true;
    const bool show_character = draw_mesh_ && character_.posed();
    if (show_character) {
      success &= character_.Draw(_renderer, _models, _transform);
    }
    if (draw_skeleton_ || !show_character) {
      success &= _renderer->DrawPosture(skeleton(), _models, _transform);
    }
    return success;
  }

  bool OnGui(ozz::sample::ImGui* _im_gui) override {
    if (!character_.meshes().empty()) {
      static bool display_open = true;
      ozz::sample::ImGui::OpenClose doc(_im_gui, "Display", &display_open);
      if (display_open) {
        _im_gui->DoCheckBox("Draw character", &draw_mesh_);
        _im_gui->DoCheckBox("Draw clip skeleton", &draw_skeleton_);
      }
    }
    return true;
  }

  bool OnDearImGui() override {
    const ImGuiIO& io = ImGui::GetIO();
    const float dpi = ImGui::GetStyle().FontScaleDpi;
    ImGui::SetNextWindowPos(ImVec2(0.f, 0.f));
    ImGui::SetNextWindowSize(ImVec2(560.f * dpi, io.DisplaySize.y),
                             ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(150.f * dpi, io.DisplaySize.y),
        ImVec2(io.DisplaySize.x * .6f, io.DisplaySize.y));
    ImGui::Begin("Clips", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    gui_left_inset_ = ImGui::GetWindowWidth();

    ProjectGui();
    ImGui::Separator();
    library_.DrawFilter();
    if (!status_.empty()) {
      ImGui::TextWrapped("%s", status_.c_str());
    }
    ImGui::Separator();
    if (character_.DrawGui()) {
      RefreshPose();
    }
    ImGui::Separator();

    // Clip tree first, with a fixed share of the panel, so the sections below
    // can't squeeze it out. Row click previews, checkbox adds to / removes
    // from the set.
    const bool running = job_.valid();
    ImGui::BeginChild("browser", ImVec2(0.f, ImGui::GetContentRegionAvail().y * .4f),
                      ImGuiChildFlags_ResizeY);
    std::vector<fs::path> in_set;
    for (const PoseSetEntry& entry : set_) {
      in_set.push_back(entry.path);
    }
    std::vector<fs::path> highlighted;
    if (previewing_) {
      highlighted.push_back(clip_.path());
    }
    fs::path clicked, toggled;
    library_.DrawTable(highlighted, &clicked, &in_set, &toggled);
    if (!clicked.empty()) {
      PreviewClip(clicked);
    }
    if (!toggled.empty() && !running) {
      Toggle(toggled);
    }
    ImGui::EndChild();

    // Pose set, preview and clustering, scrolling below.
    ImGui::BeginChild("sections");
    ImGui::BeginDisabled(running);
    PoseSetGui();
    ImGui::EndDisabled();
    PreviewGui();
    ClustersGui();
    ImGui::EndChild();
    ImGui::End();
    TransitionGraphWindow();
    DrawLabels();
    return true;
  }

  ozz::math::Box GetSceneBounds() const override {
    ozz::math::Box bounds;
    if (pose_valid_) {
      bounds = character_.Bounds(make_span(models_), math::Float4x4::identity());
    }
    if (show_gallery_ && !gallery_.empty()) {
      // Clip skeleton poses: close enough to the character for framing.
      const ozz::math::Box box = gallery_.Bounds();
      bounds = bounds.is_valid() ? Merge(bounds, box) : box;
    }
    return bounds;
  }

 private:
  // Pose set entry of _clip, nullptr if not in the set.
  PoseSetEntry* FindEntry(const fs::path& _clip) {
    const auto it = std::find_if(set_.begin(), set_.end(),
                                 [&](const PoseSetEntry& _e) { return _e.path == _clip; });
    return it == set_.end() ? nullptr : &*it;
  }

  // Trim range of an entry: the part of the clip used (end < 0: clip end).
  static float TrimEnd(const PoseSetEntry& _entry) {
    return _entry.end < 0.f ? _entry.duration : std::min(_entry.end, _entry.duration);
  }
  static bool Trimmed(const PoseSetEntry& _entry) {
    return _entry.begin > 0.f || TrimEnd(_entry) < _entry.duration;
  }
  static void SetTrim(PoseSetEntry* _entry, float _begin, float _end) {
    const float duration = _entry->duration;
    _entry->begin = std::clamp(_begin, 0.f, duration);
    _end = std::clamp(_end, _entry->begin, duration);
    _entry->end = _end >= duration - 1e-3f ? -1.f : _end;  // Clip end.
  }

  // Starts clustering the pose set on a worker thread.
  void StartClustering() {
    if (set_.empty() || job_.valid()) {
      return;
    }
    status_.clear();
    clustered_project_ = CurrentProject();
    job_ = std::async(std::launch::async, hyperpose::FindCommonPoses, set_, &skeleton(),
                      kRates[rate_], k_, static_cast<uint32_t>(seed_));
  }

  // "#3", or "end" for -1 (a clip end).
  static std::string PoseName(int _pose) {
    return _pose < 0 ? "end" : "#" + std::to_string(_pose + 1);
  }

  // The first _max edges out of a pose with their shares: " #3 60% end 20%".
  static std::string ExitsText(const hyperpose::PoseExits& _exits, size_t _max) {
    std::string text;
    for (size_t e = 0; e < std::min(_exits.edges.size(), _max); ++e) {
      const hyperpose::PoseEdge& edge = _exits.edges[e];
      char item[32];
      std::snprintf(item, sizeof(item), " %s %.0f%%", PoseName(edge.to).c_str(),
                    _exits.share(edge) * 100.f);
      text += item;
    }
    return text;
  }

  void LogResult() const {
    if (!result_.error.empty()) {
      ozz::log::Err() << result_.error << std::endl;
      return;
    }
    ozz::log::Out() << result_.poses.size() << " common poses of "
                    << result_.num_frames << " frames from "
                    << result_.clips.size() << " clips in " << result_.seconds
                    << "s:" << std::endl;
    for (size_t c = 0; c < result_.poses.size(); ++c) {
      const CommonPose& pose = result_.poses[c];
      const CommonPose::Frame& frame = pose.frames.front();
      const std::string name = result_.clips[frame.clip].stem().string();
      char line[256];
      std::snprintf(line, sizeof(line),
                    "  #%-2d %5.1f%%  spread %4.1f cm  typical %s @ %.1fs (%s)",
                    static_cast<int>(c + 1),
                    100.f * pose.frames.size() / result_.num_frames,
                    pose.spread * 100.f, name.c_str(), frame.time,
                    library_.Description(name).c_str());
      ozz::log::Out() << line << std::endl;
      // Top transitions.
      const hyperpose::PoseExits& exits = result_.exits[c];
      ozz::log::Out() << "      ->" << ExitsText(exits, 4) << "  (" << exits.count
                      << " transitions)" << std::endl;
    }
  }

  // Project: the pose set and common pose settings, as saved.
  hyperpose::PoseProject CurrentProject() const {
    hyperpose::PoseProject project;
    for (const PoseSetEntry& entry : set_) {
      // Relative to the clip root, so projects survive moving the assets.
      fs::path path = entry.path.lexically_relative(library_.root());
      if (path.empty() || *path.begin() == "..") {
        path = entry.path;
      }
      project.clips.push_back({path, entry.begin, entry.end});
    }
    project.clusters = k_;
    project.rate = kRates[rate_];
    project.seed = seed_;
    return project;
  }

  bool ProjectDirty() const { return CurrentProject() != saved_project_; }

  std::string ProjectName() const {
    return project_path_.empty() ? "Untitled" : project_path_.stem().string();
  }

  // Empty, untitled project with default settings.
  void NewProject() {
    set_.clear();
    k_ = 12;
    rate_ = 1;
    seed_ = 1;
    project_path_.clear();
    saved_project_ = CurrentProject();
  }

  // Loads a project's set and settings, then finds its common poses. Clips
  // that are missing or don't share the skeleton are left out (the project
  // then shows as changed).
  bool OpenProject(const fs::path& _path) {
    if (job_.valid()) {
      return false;
    }
    hyperpose::PoseProject project;
    std::string error;
    if (!hyperpose::LoadPoseProject(_path, &project, &error)) {
      status_ = error;
      return false;
    }
    set_.clear();
    int missing = 0;
    for (const hyperpose::PoseProject::Clip& clip : project.clips) {
      const fs::path path = clip.path.is_absolute() ? clip.path : library_.root() / clip.path;
      const size_t size = set_.size();
      std::error_code ec;
      if (!fs::exists(path, ec) || !Add(path) || set_.size() == size) {
        ++missing;
        continue;
      }
      set_.back().begin = clip.begin;
      set_.back().end = clip.end;
    }
    k_ = std::clamp(project.clusters, 2, 64);
    rate_ = 0;
    for (int i = 0; i < IM_ARRAYSIZE(kRates); ++i) {
      if (std::fabs(kRates[i] - project.rate) < std::fabs(kRates[rate_] - project.rate)) {
        rate_ = i;
      }
    }
    seed_ = project.seed;
    project_path_ = _path;
    saved_project_ = missing ? project : CurrentProject();
    hyperpose::SetRecentPoseProject(_path);
    status_ = "Opened " + ProjectName() + ": " + std::to_string(set_.size()) + " clips";
    if (missing) {
      status_ += ", " + std::to_string(missing) + " missing or not loaded";
    }
    walk_.Stop();
    StartClustering();
    return true;
  }

  bool SaveProject(const fs::path& _path) {
    const hyperpose::PoseProject project = CurrentProject();
    std::string error;
    if (!hyperpose::SavePoseProject(_path, project, &error)) {
      status_ = error;
      return false;
    }
    project_path_ = _path;
    saved_project_ = project;
    hyperpose::SetRecentPoseProject(_path);
    status_ = "Saved " + _path.string();
    return true;
  }

  // Project name (* when changed) and New / Open / Save / Save as. Changes
  // are confirmed before New or Open drop them.
  void ProjectGui() {
    const bool dirty = ProjectDirty();
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Project: %s%s", ProjectName().c_str(), dirty ? "*" : "");
    if (!project_path_.empty()) {
      ImGui::SetItemTooltip("%s", project_path_.string().c_str());
    }
    ImGui::BeginDisabled(job_.valid());
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
      project_list_ = hyperpose::ListPoseProjects(projects_dir_);
      ImGui::OpenPopup("Open project");
    }
    ImGui::SameLine();
    const bool save = ImGui::Button("Save") ||
                      ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S, ImGuiInputFlags_RouteGlobal);
    if (save) {
      if (project_path_.empty()) {
        open_save_as = true;
      } else {
        SaveProject(project_path_);
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
      ImGui::TextDisabled("%s", projects_dir_.string().c_str());
      if (project_list_.empty()) {
        ImGui::TextUnformatted("No projects yet: Save as... to make one.");
      }
      for (const fs::path& path : project_list_) {
        if (ImGui::Selectable(path.stem().string().c_str(), path == project_path_)) {
          pending_open_ = path;
        }
      }
      ImGui::EndPopup();
    }

    // Save as: a name in the projects folder.
    if (open_save_as) {
      const std::string name = project_path_.empty() ? "" : ProjectName();
      std::snprintf(save_name_, sizeof(save_name_), "%s", name.c_str());
      ImGui::OpenPopup("Save project as");
    }
    if (ImGui::BeginPopupModal("Save project as", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
      if (ImGui::IsWindowAppearing()) {
        ImGui::SetKeyboardFocusHere();
      }
      const bool enter = ImGui::InputText("Name", save_name_, sizeof(save_name_),
                                          ImGuiInputTextFlags_EnterReturnsTrue);
      const fs::path path = projects_dir_ / (std::string(save_name_) + hyperpose::kProjectExtension);
      std::error_code ec;
      const bool valid = save_name_[0] && fs::path(save_name_).filename() == fs::path(save_name_);
      if (valid && fs::exists(path, ec) && path != project_path_) {
        ImGui::TextColored(ImVec4(1.f, .7f, .3f, 1.f), "Replaces the existing %s.", save_name_);
      }
      ImGui::BeginDisabled(!valid);
      if (ImGui::Button("Save") || (enter && valid)) {
        SaveProject(path);
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
      if (!dirty) {
        ApplyPending();
      } else if (!ImGui::IsPopupOpen("Discard changes?")) {
        ImGui::OpenPopup("Discard changes?");
      }
    }
    if (ImGui::BeginPopupModal("Discard changes?", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text("%s has unsaved changes.", ProjectName().c_str());
      if (ImGui::Button("Discard")) {
        ApplyPending();
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
  }

  void ApplyPending() {
    if (pending_new_) {
      NewProject();
      status_.clear();
    } else if (!pending_open_.empty()) {
      OpenProject(pending_open_);
    }
    pending_new_ = false;
    pending_open_.clear();
  }

  // Adds a clip to the pose set, or removes it if already in. All clips must
  // share a skeleton.
  void Toggle(const fs::path& _clip) {
    const auto it = std::find_if(set_.begin(), set_.end(), [&](const PoseSetEntry& _e) {
      return _e.path == _clip;
    });
    if (it != set_.end()) {
      set_.erase(it);
      return;
    }
    Add(_clip);
  }

  bool Add(const fs::path& _clip) {
    if (std::any_of(set_.begin(), set_.end(),
                    [&](const PoseSetEntry& _e) { return _e.path == _clip; })) {
      return true;
    }
    if (!UseSkeleton(_clip)) {
      return false;
    }
    float duration = 0.f;
    hyperpose::ReadClipDuration(_clip.string().c_str(), &duration);
    set_.push_back({_clip, duration});
    status_.clear();
    return true;
  }

  // Loads the skeleton of _clip if not current. Switching is only allowed
  // while the pose set is empty, as all set clips must share a skeleton.
  bool UseSkeleton(const fs::path& _clip) {
    const auto use = clip_skeleton_.Use(_clip, library_.root(),
                                        set_.empty() && !job_.valid());
    if (use == hyperpose::ClipSkeleton::Result::kFailed) {
      status_ = clip_skeleton_.error();
      return false;
    }
    if (use == hyperpose::ClipSkeleton::Result::kChanged) {
      // Everything loaded for the previous skeleton is stale.
      walk_.Stop();
      models_.resize(skeleton().num_joints());
      character_.SetClipSkeleton(&skeleton(), clip_skeleton_.path());
      result_ = hyperpose::CommonPoseResult();
      gallery_.Clear();
      pose_ = -1;
      previewing_ = false;
      clip_.Unload();
      pose_valid_ = false;
    }
    return true;
  }

  const ozz::animation::Skeleton& skeleton() const {
    return clip_skeleton_.skeleton();
  }

  void PoseSetGui() {
    float total = 0.f;  // Of the trimmed clips.
    for (const PoseSetEntry& entry : set_) {
      total += TrimEnd(entry) - entry.begin;
    }
    char header[96];
    std::snprintf(header, sizeof(header), "Pose set: %d clips, %.0fs###set",
                  static_cast<int>(set_.size()), total);
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }
    ImGui::TextDisabled("Tick clips below to add them, click any clip to preview (and trim) it.");
    const std::vector<fs::path> filtered = library_.FilteredClips();
    char add[64];
    std::snprintf(add, sizeof(add), "Add all filtered (%d)",
                  static_cast<int>(filtered.size()));
    if (ImGui::Button(add)) {
      for (const fs::path& clip : filtered) {
        if (!Add(clip)) {
          break;
        }
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
      set_.clear();
    }

    // Set list, with trim ranges.
    if (!set_.empty()) {
      const float line = ImGui::GetTextLineHeightWithSpacing();
      ImGui::BeginChild("set", ImVec2(0.f, std::min<float>(set_.size(), 6.f) * line + line * .5f),
                        ImGuiChildFlags_Borders);
      for (size_t i = 0; i < set_.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::SmallButton("x")) {
          set_.erase(set_.begin() + i);
          ImGui::PopID();
          break;
        }
        ImGui::SameLine();
        const std::string name = set_[i].path.stem().string();
        char label[128];
        if (Trimmed(set_[i])) {
          std::snprintf(label, sizeof(label), "%s  %.1f-%.1fs of %.1fs", name.c_str(),
                        set_[i].begin, TrimEnd(set_[i]), set_[i].duration);
        } else {
          std::snprintf(label, sizeof(label), "%s  %.1fs", name.c_str(), set_[i].duration);
        }
        if (ImGui::Selectable(label, PreviewedClip() == set_[i].path, 0,
                              ImVec2(ImGui::CalcTextSize(label).x, 0.f))) {
          PreviewClip(set_[i].path);
        }
        ImGui::SetItemTooltip("Click to preview and trim.");
        const std::string& description = library_.Description(name);
        if (!description.empty()) {
          ImGui::SameLine();
          ImGui::TextDisabled("%s", description.c_str());
        }
        ImGui::PopID();
      }
      ImGui::EndChild();
    }
  }

  void ClustersGui() {
    if (!ImGui::CollapsingHeader("Common poses", ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }
    const bool running = job_.valid();
    ImGui::BeginDisabled(running);
    const float half = ImGui::GetContentRegionAvail().x * .35f;
    ImGui::SetNextItemWidth(half);
    ImGui::SliderInt("Clusters", &k_, 2, 64);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(half * .5f);
    ImGui::Combo("Rate", &rate_, kRateLabels, IM_ARRAYSIZE(kRateLabels));
    ImGui::SetItemTooltip("Frames sampled per second of animation.");
    ImGui::SetNextItemWidth(half);
    ImGui::InputInt("Seed", &seed_);
    ImGui::BeginDisabled(set_.empty());
    if (ImGui::Button("Find common poses")) {
      StartClustering();
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    if (running) {
      ImGui::SameLine();
      ImGui::TextUnformatted("Working...");
    }

    if (result_.poses.empty()) {
      return;
    }
    if (!running && CurrentProject() != clustered_project_) {
      ImGui::TextColored(ImVec4(1.f, .7f, .3f, 1.f),
                         "Pose set, trims or settings changed: find common poses again.");
    }
    ImGui::Text("%d frames from %d clips in %.1fs", result_.num_frames,
                static_cast<int>(result_.clips.size()), result_.seconds);

    // Common poses, most frames first.
    const float rows = std::min<float>(result_.poses.size(), 8.f) + 1.5f;
    if (ImGui::BeginTable("poses", 4,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                              ImGuiTableFlags_BordersInnerV,
                          ImVec2(0.f, rows * ImGui::GetTextLineHeightWithSpacing()))) {
      ImGui::TableSetupScrollFreeze(0, 1);
      ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableSetupColumn("Share", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableSetupColumn("Spread", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableSetupColumn("Most typical frame", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      for (int c = 0; c < static_cast<int>(result_.poses.size()); ++c) {
        const CommonPose& pose = result_.poses[c];
        const CommonPose::Frame& frame = pose.frames.front();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        char label[16];
        std::snprintf(label, sizeof(label), "%d", c + 1);
        if (ImGui::Selectable(label, pose_ == c,
                              ImGuiSelectableFlags_SpanAllColumns)) {
          SelectPose(c);
        }
        ImGui::TableNextColumn();
        ImGui::Text("%.1f%%", 100.f * pose.frames.size() / result_.num_frames);
        ImGui::TableNextColumn();
        ImGui::Text("%.1f cm", pose.spread * 100.f);
        ImGui::TableNextColumn();
        const std::string name = result_.clips[frame.clip].stem().string();
        ImGui::Text("%s @ %.1fs", name.c_str(), frame.time);
        const std::string& description = library_.Description(name);
        if (!description.empty()) {
          ImGui::SetItemTooltip("%s", description.c_str());
        }
      }
      ImGui::EndTable();
    }

    // Above the selected pose's details, which the walk changes.
    WalkGui();

    // Frames of the selected common pose, most typical first.
    if (pose_ >= 0) {
      const CommonPose& pose = result_.poses[pose_];
      ImGui::SetNextItemWidth(-FLT_MIN);
      if (ImGui::SliderInt("##frame", &frame_, 0,
                           static_cast<int>(pose.frames.size()) - 1,
                           "frame %d (0 = most typical)")) {
        ShowFrame(frame_);
      }
      const CommonPose::Frame& frame = pose.frames[frame_];
      const std::string name = result_.clips[frame.clip].stem().string();
      if (ImGui::SmallButton("Play##frame")) {
        StartPreview(result_.clips[frame.clip], frame.time);
      }
      ImGui::SetItemTooltip("Previews the clip from this frame.");
      ImGui::SameLine();
      ImGui::Text("%s @ %.2fs, %.1f cm from pose  %s", name.c_str(), frame.time,
                  frame.distance * 100.f, library_.Description(name).c_str());
      TransitionsGui();
    }
    if (ImGui::Checkbox("Align pose to origin", &align_to_origin_)) {
      RefreshPose();
    }
    ImGui::Checkbox("Show all common poses", &show_gallery_);
    ImGui::SetItemTooltip("Every common pose, in rows behind, largest first.");
    ImGui::SameLine();
    ImGui::Checkbox("Animate", &gallery_.animate);
    ImGui::SetItemTooltip(
        "Each pose plays its clips from a random frame of the pose, switching\n"
        "after the play time. Frames closer to the pose are picked more often.\n"
        "Off: most typical frame, still (cheaper to draw).");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::SliderFloat("##spacing", &gallery_.spacing, .5f, 3.f, "spacing %.1f m");
    ImGui::BeginDisabled(!gallery_.animate);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * .5f);
    ImGui::SliderFloat("Play time", &gallery_.play_time, .1f, 10.f, "%.1f s",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("How long each picked frame's clip plays before the next pick.");
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Checkbox("Debug text", &debug_text_);
    ImGui::SetItemTooltip("Clip, time / duration and frame above each character.");
    ImGui::SameLine();
    ImGui::Checkbox("Transition graph", &show_transitions_);
    ImGui::SetItemTooltip("Window with the common poses and the transitions between them.");
    ImGui::Separator();
  }

  // Transition graph: a node per common pose around a circle, largest first
  // at the top, clockwise, sized by share. Arrows out of (orange) and into
  // (blue) the selected pose, thicker for more frequent; with "All poses",
  // the other poses' transitions too, grey. Clip ends are short "end" arrows
  // pointing out of the circle. Nodes without a drawn arrow are dimmed.
  // Click a node to select it.
  void TransitionGraphWindow() {
    if (!show_transitions_ || result_.poses.empty()) {
      return;
    }
    const ImGuiIO& io = ImGui::GetIO();
    const float dpi = ImGui::GetStyle().FontScaleDpi;
    ImGui::SetNextWindowSize(ImVec2(360.f * dpi, 400.f * dpi), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(
        ImVec2(io.DisplaySize.x - 370.f * dpi, io.DisplaySize.y - 410.f * dpi),
        ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Pose transitions", &show_transitions_)) {
      ImGui::End();
      return;
    }
    ImGui::Checkbox("All poses", &all_transitions_);
    ImGui::SetItemTooltip("Also the other poses' transitions, in grey.");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::SliderFloat("##min_share", &min_transition_share_, 0.f, .5f,
                       "hide below %.2f share");
    if (pose_ >= 0) {
      ImGui::TextColored(ImVec4(1.f, .67f, .24f, 1.f), "out of #%d", pose_ + 1);
      ImGui::SameLine();
      ImGui::TextColored(ImVec4(.35f, .67f, 1.f, 1.f), "into #%d", pose_ + 1);
      ImGui::SameLine();
      ImGui::TextDisabled("(%% of source's)");
      ImGui::SetItemTooltip("Arrow shares are of the source pose's transitions.");
    }

    // Canvas.
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 size(std::max(avail.x, 60.f), std::max(avail.y, 60.f));
    ImGui::InvisibleButton("graph", size);
    const bool hovered_canvas = ImGui::IsItemHovered();
    const bool clicked = ImGui::IsItemClicked();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), true);

    // Nodes.
    const int count = static_cast<int>(result_.poses.size());
    const float max_radius = 24.f * dpi;
    const float min_radius = 9.f * dpi;
    const ImVec2 center(origin.x + size.x * .5f, origin.y + size.y * .5f);
    // Room outside the circle for dead end arrows and their labels.
    const float kEndLength = 14.f * dpi;
    const float ring = std::max(
        10.f, std::min(size.x, size.y) * .5f - max_radius - kEndLength - 24.f * dpi);
    const float largest = static_cast<float>(result_.poses.front().frames.size());
    std::vector<ImVec2> positions(count);
    std::vector<float> radii(count);
    int hovered = -1;
    for (int p = 0; p < count; ++p) {
      const float angle = -1.5707963f + 6.2831853f * p / count;
      positions[p] = ImVec2(center.x + ring * std::cos(angle),
                            center.y + ring * std::sin(angle));
      radii[p] = min_radius + (max_radius - min_radius) *
                                  std::sqrt(result_.poses[p].frames.size() / largest);
      const float dx = io.MousePos.x - positions[p].x;
      const float dy = io.MousePos.y - positions[p].y;
      if (hovered_canvas && dx * dx + dy * dy <= radii[p] * radii[p]) {
        hovered = p;
      }
    }

    // Edges: curved to the right of the direction, so a->b and b->a don't
    // overlap, with an arrow head on the target and the share at the middle.
    // A dead end (_to -1, the clips end) is a short straight arrow out of the
    // circle, labeled "end".
    auto edge = [&](int _from, int _to, float _share, ImU32 _color, float _thickness,
                    bool _label) {
      auto toward = [](ImVec2 _p, ImVec2 _q, float _distance) {
        const float x = _q.x - _p.x, y = _q.y - _p.y;
        const float l = std::max(1e-3f, std::sqrt(x * x + y * y));
        return ImVec2(_p.x + x / l * _distance, _p.y + y / l * _distance);
      };
      const ImVec2 a = positions[_from];
      const ImVec2 b = _to >= 0 ? positions[_to]
                                : toward(center, a, ring + radii[_from] + kEndLength);
      const float dx = b.x - a.x, dy = b.y - a.y;
      const float length = std::max(1.f, std::sqrt(dx * dx + dy * dy));
      const float bend = _to >= 0 ? .15f : 0.f;
      const ImVec2 control((a.x + b.x) * .5f - dy / length * length * bend,
                           (a.y + b.y) * .5f + dx / length * length * bend);
      const ImVec2 start = toward(a, control, radii[_from]);
      const ImVec2 end = _to >= 0 ? toward(b, control, radii[_to] + 2.f) : b;
      // Arrow head along the curve tangent at the end.
      const float tx = end.x - control.x, ty = end.y - control.y;
      const float tl = std::max(1e-3f, std::sqrt(tx * tx + ty * ty));
      const ImVec2 dir(tx / tl, ty / tl);
      const float head = (6.f + _thickness * 1.5f) * dpi;
      const ImVec2 base(end.x - dir.x * head, end.y - dir.y * head);
      draw_list->AddBezierQuadratic(start, control, base, _color, _thickness);
      draw_list->AddTriangleFilled(
          end, ImVec2(base.x - dir.y * head * .5f, base.y + dir.x * head * .5f),
          ImVec2(base.x + dir.y * head * .5f, base.y - dir.x * head * .5f), _color);
      if (_label) {
        char text[16];
        std::snprintf(text, sizeof(text), _to >= 0 ? "%.0f%%" : "end %.0f%%",
                      _share * 100.f);
        const ImVec2 text_size = ImGui::CalcTextSize(text);
        // Past the arrow tip for a dead end, else mid curve.
        const ImVec2 mid =
            _to >= 0 ? ImVec2(.25f * start.x + .5f * control.x + .25f * base.x,
                              .25f * start.y + .5f * control.y + .25f * base.y)
                     : toward(end, ImVec2(2.f * end.x - start.x, 2.f * end.y - start.y),
                              .5f * std::max(text_size.x, text_size.y) + 3.f * dpi);
        const ImVec2 pos(mid.x - text_size.x * .5f, mid.y - text_size.y * .5f);
        draw_list->AddRectFilled(ImVec2(pos.x - 2.f, pos.y),
                                 ImVec2(pos.x + text_size.x + 2.f, pos.y + text_size.y),
                                 IM_COL32(20, 20, 20, 200), 3.f);
        draw_list->AddText(pos, _color, text);
      }
    };
    // Grey: other poses' transitions (with "All poses"). Blue: into the
    // selected pose, orange: out of it, drawn last to stay on top. Shares are
    // of the source pose's transitions.
    const ImU32 kOut = IM_COL32(255, 170, 60, 255);
    const ImU32 kIn = IM_COL32(90, 170, 255, 255);
    std::vector<bool> connected(count, false);  // Has a drawn arrow.
    std::vector<bool> targets(count, false), sources(count, false);
    for (int pass = 0; pass < 3; ++pass) {  // Others, in, out.
      for (int p = 0; p < count; ++p) {
        const hyperpose::PoseExits& exits = result_.exits[p];
        for (const hyperpose::PoseEdge& e : exits.edges) {
          const float share = exits.share(e);
          const bool out = p == pose_;
          const bool in = e.to >= 0 && e.to == pose_;
          if (share < min_transition_share_ ||
              (pass == 0 && (out || in || !all_transitions_)) ||
              (pass == 1 && !in) || (pass == 2 && !out)) {
            continue;
          }
          connected[p] = true;
          if (e.to >= 0) {
            connected[e.to] = true;
          }
          if (pass == 0) {
            edge(p, e.to, share, IM_COL32(170, 170, 170, 170),
                 (1.f + 4.f * share) * dpi, false);
          } else {
            if (e.to >= 0) {
              (out ? targets[e.to] : sources[p]) = true;
            }
            edge(p, e.to, share, out ? kOut : kIn, (1.f + 7.f * share) * dpi, true);
          }
        }
      }
    }

    // Nodes over the edges: selected orange, its targets / sources tinted,
    // nodes without a drawn arrow dimmed.
    for (int p = 0; p < count; ++p) {
      const bool dim = !connected[p] && p != pose_;
      const ImU32 fill = p == pose_     ? IM_COL32(230, 140, 40, 255)
                         : targets[p] ? IM_COL32(120, 90, 50, 255)
                         : sources[p] ? IM_COL32(50, 85, 125, 255)
                         : dim        ? IM_COL32(40, 40, 40, 255)
                                      : IM_COL32(75, 75, 75, 255);
      draw_list->AddCircleFilled(positions[p], radii[p], fill);
      draw_list->AddCircle(positions[p], radii[p],
                           p == hovered ? IM_COL32(255, 255, 255, 255)
                           : dim        ? IM_COL32(90, 90, 90, 255)
                                        : IM_COL32(160, 160, 160, 255),
                           0, p == hovered ? 2.f : 1.f);
      char text[8];
      std::snprintf(text, sizeof(text), "#%d", p + 1);
      const ImVec2 text_size = ImGui::CalcTextSize(text);
      draw_list->AddText(ImVec2(positions[p].x - text_size.x * .5f,
                                positions[p].y - text_size.y * .5f),
                         dim ? IM_COL32(120, 120, 120, 255) : IM_COL32(255, 255, 255, 255),
                         text);
    }
    draw_list->PopClipRect();

    if (hovered >= 0) {
      const CommonPose& pose = result_.poses[hovered];
      const CommonPose::Frame& frame = pose.frames.front();
      const std::string name = result_.clips[frame.clip].stem().string();
      const std::string next = ExitsText(result_.exits[hovered], 4);
      ImGui::SetTooltip("#%d: %.1f%% of frames, spread %.1f cm\n%s @ %.1fs  %s\nnext:%s",
                        hovered + 1, 100.f * pose.frames.size() / result_.num_frames,
                        pose.spread * 100.f, name.c_str(), frame.time,
                        library_.Description(name).c_str(),
                        next.empty() ? " none" : next.c_str());
      if (clicked) {
        SelectPose(hovered);
      }
    }
    ImGui::End();
  }

  // Transitions from the selected pose: where its clips go next (or end),
  // most frequent first. Click a pose to select it, Play to preview the clip
  // before the transition (cycles through the transitions).
  void TransitionsGui() {
    const hyperpose::PoseExits& exits = result_.exits[pose_];
    char header[64];
    std::snprintf(header, sizeof(header), "Transitions from #%d (%d)###transitions",
                  pose_ + 1, exits.count);
    if (!ImGui::TreeNodeEx(header, ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }
    ImGui::SetItemTooltip(
        "Next common pose along the clips, or \"end\" where they end in this pose.\n"
        "Bounces back within %.1fs are ignored.",
        hyperpose::kMinPoseStay);
    transition_example_.resize(exits.edges.size());
    // Fixed height (scrolls past 6 rows), so the UI below doesn't move when
    // the selected pose changes, as it does during the pose walk.
    const float rows = 6.f + 1.5f;
    if (ImGui::BeginTable("transitions", 4,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                              ImGuiTableFlags_BordersInnerV,
                          ImVec2(0.f, rows * ImGui::GetTextLineHeightWithSpacing()))) {
      ImGui::TableSetupScrollFreeze(0, 1);
      ImGui::TableSetupColumn("To", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableSetupColumn("Share", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableSetupColumn("Transition", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      if (exits.edges.empty()) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(3);
        ImGui::TextDisabled("None: only seen in bounces.");
      }
      int select = -1;
      for (size_t e = 0; e < exits.edges.size(); ++e) {
        const hyperpose::PoseEdge& edge = exits.edges[e];
        const int count = static_cast<int>(edge.transitions.size());
        ImGui::PushID(static_cast<int>(e));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if (edge.to >= 0) {
          if (ImGui::Selectable(PoseName(edge.to).c_str())) {
            select = edge.to;
          }
          ImGui::SetItemTooltip("Select pose #%d.", edge.to + 1);
        } else {
          ImGui::TextDisabled("end");
          ImGui::SetItemTooltip("The clips end in this pose.");
        }
        ImGui::TableNextColumn();
        ImGui::Text("%.0f%%", exits.share(edge) * 100.f);
        ImGui::TableNextColumn();
        ImGui::Text("%d", count);
        ImGui::TableNextColumn();
        int& shown = transition_example_[e];
        const hyperpose::PoseTransition& transition =
            result_.transitions[edge.transitions[shown % count]];
        if (ImGui::SmallButton("Play")) {
          // From a bit before, to see the transition happen.
          StartPreview(result_.clips[transition.clip], std::max(0.f, transition.time - 1.f));
          shown = (shown + 1) % count;
        }
        ImGui::SetItemTooltip(
            "Plays from 1s before the transition, the next one on each click.");
        ImGui::SameLine();
        ImGui::Text("next %d/%d: %s @ %.1fs", shown % count + 1, count,
                    result_.clips[transition.clip].stem().string().c_str(), transition.time);
        ImGui::PopID();
      }
      ImGui::EndTable();
      if (select >= 0) {
        SelectPose(select);
      }
    }
    ImGui::TreePop();
  }

  // Selects a common pose (-1 for none), showing its most typical frame.
  void SelectPose(int _pose) {
    pose_ = _pose;
    frame_ = 0;
    transition_example_.clear();
    if (pose_ >= 0) {
      ShowFrame(frame_);
    }
  }

  // Shows a frame of the selected pose, still.
  void ShowFrame(int _frame) {
    previewing_ = false;
    walk_.Stop();
    const CommonPose::Frame& frame = result_.poses[pose_].frames[_frame];
    SampleClip(result_.clips[frame.clip], frame.time);
  }

  // Previews a clip picked by the user.
  void PreviewClip(const fs::path& _clip) { StartPreview(_clip, 0.f); }

  // Clip being previewed (empty when showing a still frame).
  fs::path PreviewedClip() const { return previewing_ ? clip_.path() : fs::path(); }

  // Re-samples the displayed pose (preview or selected frame). The walk
  // samples every update.
  void RefreshPose() {
    if (walk_.active()) {
      return;
    }
    if (previewing_) {
      SampleClip(clip_.path(), clip_.time);
    } else if (pose_ >= 0) {
      ShowFrame(frame_);
    }
  }

  // Previews a clip, playing from _time.
  void StartPreview(const fs::path& _clip, float _time) {
    if (!UseSkeleton(_clip)) {
      return;
    }
    status_.clear();
    walk_.Stop();
    previewing_ = false;
    if (!SampleClip(_clip, _time)) {  // Loads the clip.
      return;
    }
    previewing_ = true;
    if (walk_.root_motion) {
      // Root motion from the origin.
      AnchorPreview(_time, true);
      SampleClip(_clip, _time);
    }
    clip_.playing = true;
  }

  // Pose walk: the main character walks the transition graph from the
  // selected pose (random if none), instead of the preview / still frame.
  void StartWalk() {
    previewing_ = false;
    if (!walk_.Start(&result_, &skeleton(), clip_skeleton_.heading_axis(), pose_)) {
      status_ = "Pose walk failed to start.";
      return;
    }
    UpdateWalk(0.f);
  }

  void StopWalk() {
    walk_.Stop();
    pose_valid_ = false;
    RefreshPose();  // Still frame of the selected pose.
  }

  void UpdateWalk(float _dt) {
    pose_valid_ = walk_.Update(_dt, make_span(models_));
    if (!walk_.active()) {
      status_ = "Pose walk failed to load a clip.";
      return;
    }
    // Selects the pose the character is in (shown in the transition graph):
    // the source pose until the clip reaches the next one.
    const hyperpose::PoseWalk::Segment& segment = walk_.segment();
    const int pose =
        walk_.clip_time() < segment.time || segment.to < 0 ? segment.from : segment.to;
    if (pose != pose_) {
      pose_ = pose;
      frame_ = 0;
      transition_example_.clear();
    }
  }

  void WalkGui() {
    if (!ImGui::TreeNodeEx("Pose walk", ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }
    if (walk_.active()) {
      if (ImGui::Button("Stop walk", ImVec2(ImGui::GetFontSize() * 6.f, 0.f))) {
        StopWalk();
      }
    } else if (ImGui::Button("Walk", ImVec2(ImGui::GetFontSize() * 6.f, 0.f))) {
      StartWalk();
    }
    ImGui::SetItemTooltip(
        "The main character walks the transition graph: from its pose, picks\n"
        "the next one weighted by how often the clips go there, plays an\n"
        "example of that transition, crossfades into the next, and so on.\n"
        "Starts from the selected pose.");
    if (walk_.active()) {
      const hyperpose::PoseWalk::Segment& segment = walk_.segment();
      ImGui::SameLine();
      ImGui::Text("%s#%d -> %s  %s @ %.1fs", segment.jump ? "jump " : "",
                  segment.from + 1, PoseName(segment.to).c_str(),
                  result_.clips[segment.clip].stem().string().c_str(), walk_.clip_time());
      if (segment.jump) {
        ImGui::SetItemTooltip("The previous clip ended: jumped to #%d, a pose the clips "
                              "start in.", segment.from + 1);
      }
    }
    const float width = ImGui::GetContentRegionAvail().x / 2.f - ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetNextItemWidth(width);
    ImGui::SliderFloat("##max_stay", &walk_.max_stay, .1f, 20.f, "max stay %.1fs",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip(
        "Each pose plays as long as the example clip stayed in it, from when\n"
        "the clip entered it to when it reaches the next pose, up to this.");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::SliderFloat("##blend", &walk_.blend, 0.f, 1.f, "blend %.2fs");
    ImGui::SetItemTooltip("Crossfade from one transition example to the next.");
    RootMotionGui();
    ImGui::TreePop();
  }

  // Root motion of the main character (preview and pose walk) and the
  // character's place, shared by both.
  void RootMotionGui() {
    if (ImGui::Checkbox("Root motion", &walk_.root_motion)) {
      // Continues from where the character stands.
      if (previewing_ && walk_.root_motion) {
        AnchorPreview(clip_.time, false);
      }
      RefreshPose();
    }
    ImGui::SetItemTooltip(
        "The main character moves and turns as its clip does: a preview from\n"
        "the origin, continuing from where it is when looping; the pose walk\n"
        "continuing from where each transition left it. Off: stays in place,\n"
        "facing the default heading.");
    if (walk_.root_motion) {
      ImGui::SameLine();
      if (ImGui::SmallButton("Back to origin")) {
        walk_.ResetPosition();
        if (previewing_) {
          AnchorPreview(clip_.time, true);
          RefreshPose();
        }
      }
    }
  }

  // Preview root motion: places the clip so that its root at _time stands at
  // the origin facing the default heading (_origin), or where the character
  // root is (last pose).
  void AnchorPreview(float _time, bool _origin) {
    ozz::vector<math::Float4x4> models(skeleton().num_joints());
    if (!clip_.Sample(_time, make_span(models))) {
      return;
    }
    const int axis = clip_skeleton_.heading_axis();
    math::Float3 target = math::Float3::zero();
    float target_heading = 0.f;
    if (!_origin && pose_valid_) {
      math::Store3PtrU(models_[0].cols[3], &target.x);
      target_heading = hyperpose::RootHeading(models_[0], axis);
    }
    math::Float3 root;
    math::Store3PtrU(models[0].cols[3], &root.x);
    const float heading = hyperpose::RootHeading(models[0], axis);
    auto yaw = [](float _angle) {
      return math::Float4x4::FromAxisAngle(math::simd_float4::y_axis(),
                                           math::simd_float4::Load1(_angle));
    };
    preview_anchor_ =
        math::Float4x4::Translation(math::simd_float4::Load(target.x, 0.f, target.z, 1.f)) *
        yaw(target_heading - heading) *
        math::Float4x4::Translation(math::simd_float4::Load(-root.x, 0.f, -root.z, 1.f));
  }

  bool PreviewRootMotion() const { return previewing_ && walk_.root_motion; }

  void UpdatePreview(float _dt) {
    const bool playing = clip_.playing;
    const float previous = clip_.time;
    clip_.Advance(_dt);
    // Plays the trim range only: wraps (or stops) at its ends.
    const PoseSetEntry* entry = FindEntry(clip_.path());
    if (playing && play_trim_ && entry && Trimmed(*entry)) {
      const float begin = entry->begin, end = TrimEnd(*entry);
      if (clip_.time > end || clip_.time < begin) {
        if (clip_.loop) {
          clip_.time = clip_.speed >= 0.f ? begin : end;
        } else {
          clip_.time = std::clamp(clip_.time, begin, end);
          clip_.playing = false;
        }
      }
    }
    // Wrapped around (loop or trim range): root motion continues from where
    // the character is.
    const bool wrapped = clip_.speed >= 0.f ? clip_.time < previous : clip_.time > previous;
    if (playing && wrapped && PreviewRootMotion()) {
      AnchorPreview(clip_.time, false);
    }
    SampleClip(clip_.path(), clip_.time);
  }

  void PreviewGui() {
    if (!previewing_ ||
        !ImGui::CollapsingHeader("Preview", ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }
    const std::string name = clip_.path().stem().string();
    ImGui::Text("%s  %s", name.c_str(), library_.Description(name).c_str());
    const float duration = clip_.duration();
    PoseSetEntry* entry = FindEntry(clip_.path());
    const bool trim_range = play_trim_ && entry && Trimmed(*entry);
    if (ImGui::Button(clip_.playing ? "Pause" : "Play",
                      ImVec2(ImGui::GetFontSize() * 4.f, 0.f))) {
      const float end = trim_range ? TrimEnd(*entry) : duration;
      if (!clip_.playing && !clip_.loop && clip_.time >= end) {
        clip_.time = trim_range ? entry->begin : 0.f;  // Replays from the start.
      }
      clip_.playing = !clip_.playing;
    }
    ImGui::SameLine();
    if (ImGui::Button("Close")) {
      previewing_ = false;
      pose_valid_ = false;
      RefreshPose();  // Back to the selected frame, if any.
      return;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Loop", &clip_.loop);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::SliderFloat("##speed", &clip_.speed, -2.f, 2.f, "speed %.2fx");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::SliderFloat("##time", &clip_.time, 0.f, duration, "%.2fs")) {
      clip_.playing = false;
    }
    if (entry && Trimmed(*entry) && duration > 0.f) {
      // Trimmed out parts shaded on the time slider.
      const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
      auto x = [&](float _t) { return min.x + (max.x - min.x) * _t / duration; };
      ImDrawList* draw_list = ImGui::GetWindowDrawList();
      const ImU32 shade = IM_COL32(0, 0, 0, 150);
      draw_list->AddRectFilled(min, ImVec2(x(entry->begin), max.y), shade);
      draw_list->AddRectFilled(ImVec2(x(TrimEnd(*entry)), min.y), max, shade);
      const ImU32 mark = IM_COL32(255, 170, 60, 255);
      draw_list->AddLine(ImVec2(x(entry->begin), min.y), ImVec2(x(entry->begin), max.y), mark, 2.f);
      draw_list->AddLine(ImVec2(x(TrimEnd(*entry)), min.y), ImVec2(x(TrimEnd(*entry)), max.y),
                         mark, 2.f);
    }
    TrimGui(entry);
    RootMotionGui();
    ImGui::Separator();
  }

  // Trim range of the previewed clip, for this project: the part of the clip
  // used for the common poses. The clip file isn't changed.
  void TrimGui(PoseSetEntry* _entry) {
    const bool running = job_.valid();
    if (!_entry) {
      ImGui::BeginDisabled(running);
      if (ImGui::SmallButton("Add to pose set")) {
        Add(clip_.path());
      }
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::TextDisabled("to trim it.");
      return;
    }
    ImGui::BeginDisabled(running);
    float begin = _entry->begin, end = TrimEnd(*_entry);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Trim");
    ImGui::SetItemTooltip(
        "Part of the clip used for the common poses, saved with the project.\n"
        "The clip file isn't changed.");
    ImGui::SameLine();
    const float buttons = ImGui::CalcTextSize("[ now] nowReset").x +
                          ImGui::GetStyle().FramePadding.x * 6.f +
                          ImGui::GetStyle().ItemSpacing.x * 3.f;
    ImGui::SetNextItemWidth(std::max(80.f, ImGui::GetContentRegionAvail().x - buttons));
    if (ImGui::DragFloatRange2("##trim", &begin, &end, .01f, 0.f, _entry->duration,
                               "begin %.2fs", "end %.2fs", ImGuiSliderFlags_AlwaysClamp)) {
      SetTrim(_entry, begin, end);
    }
    ImGui::SameLine();
    if (ImGui::Button("[ now")) {
      SetTrim(_entry, clip_.time, std::max(end, clip_.time));
    }
    ImGui::SetItemTooltip("Begin at the current time.");
    ImGui::SameLine();
    if (ImGui::Button("] now")) {
      SetTrim(_entry, std::min(begin, clip_.time), clip_.time);
    }
    ImGui::SetItemTooltip("End at the current time.");
    ImGui::SameLine();
    ImGui::BeginDisabled(!Trimmed(*_entry));
    if (ImGui::Button("Reset")) {
      SetTrim(_entry, 0.f, _entry->duration);
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    ImGui::Checkbox("Play trim range only", &play_trim_);
  }

  // Samples _clip at _time into models_: placed by the preview anchor with
  // root motion, else aligned to the origin if enabled.
  bool SampleClip(const fs::path& _clip, float _time) {
    pose_valid_ = false;
    if (_clip != clip_.path() && !clip_.Load(_clip, skeleton())) {
      status_ = "Failed to load " + _clip.filename().string();
      return false;
    }
    clip_.time = _time;
    if (!clip_.Sample(make_span(models_))) {
      return false;
    }
    if (PreviewRootMotion()) {
      for (math::Float4x4& model : models_) {
        model = preview_anchor_ * model;
      }
    } else if (align_to_origin_) {
      hyperpose::AlignToOrigin(make_span(models_), clip_skeleton_.heading_axis());
    }
    pose_valid_ = true;
    return true;
  }

  hyperpose::ClipLibrary library_;
  hyperpose::CharacterView character_;
  std::string status_;
  bool draw_mesh_ = true;
  bool draw_skeleton_ = false;

  // Pose set and shared skeleton.
  std::vector<PoseSetEntry> set_;

  // Project file of the set and settings, as last opened / saved.
  fs::path projects_dir_;
  fs::path project_path_;  // Empty: untitled.
  hyperpose::PoseProject saved_project_;
  std::vector<fs::path> project_list_;  // Open popup.
  char save_name_[128] = {};
  bool pending_new_ = false;
  fs::path pending_open_;
  // Preview plays only the trim range of a set clip.
  bool play_trim_ = true;
  // Set and settings the common poses were found for, to flag stale ones.
  hyperpose::PoseProject clustered_project_;
  hyperpose::ClipSkeleton clip_skeleton_;

  // Clustering.
  int k_ = 12;
  int rate_ = 1;  // kRates index.
  int seed_ = 1;
  std::future<hyperpose::CommonPoseResult> job_;
  hyperpose::CommonPoseResult result_;
  int pose_ = -1;   // Selected common pose.
  int frame_ = 0;   // Shown frame of the selected pose.

  // Every common pose, behind the displayed pose.
  hyperpose::PoseGallery gallery_;
  hyperpose::PoseWalk walk_;  // Main character pose walk, when active.
  bool walk_on_result_ = false;  // --walk: starts on the first result.
  bool show_gallery_ = true;

  // Displayed pose: a clip preview (playing), or a still frame of the
  // selected common pose.
  hyperpose::ClipPlayer clip_;
  bool previewing_ = false;
  bool align_to_origin_ = true;
  bool pose_valid_ = false;
  ozz::vector<math::Float4x4> models_;
  // Preview root motion: places the clip (see AnchorPreview).
  math::Float4x4 preview_anchor_ = math::Float4x4::identity();

  // Debug labels of this frame (screen position, text).
  struct Label {
    ImVec2 pos;
    std::string text;
  };
  std::vector<Label> labels_;
  bool debug_text_ = true;
  // Transition graph window: arrows rarer than the share are hidden; all
  // poses' transitions, or only the selected pose's.
  bool show_transitions_ = true;
  bool all_transitions_ = true;
  float min_transition_share_ = .05f;
  std::vector<int> transition_example_;  // Per edge: next transition to play.
};

int main(int _argc, const char** _argv) {
  return CommonPosesApplication().Run(_argc, _argv, "0.1", "Common Poses");
}
