// Principal Poses Walk: finds the most common poses in a set of clips, and
// walks the transitions between them. Every frame of the selected clips is
// turned into a pose feature (root relative, heading aligned joint positions),
// then grouped with k-means. Clusters are listed by size; each shows its most
// typical frame, and its members can be browsed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <map>
#include <string>
#include <vector>

#include <imgui.h>

#include "common/core/anim_utils.h"
#include "common/gui/character_view.h"
#include "common/gui/clip_library.h"
#include "common/core/clip_player.h"
#include "common/gui/clip_preview.h"
#include "common/core/common_pose.h"
#include "common/gui/pose_gallery.h"
#include "common/core/pose_project.h"
#include "common/core/pose_walk.h"
#include "common/gui/project_bar.h"
#include "common/gui/transition_graph.h"
#include "common/core/pose_match.h"
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
OZZ_OPTIONS_DECLARE_BOOL(best_match, "Best match blend for the pose walk.", false, false)
OZZ_OPTIONS_DECLARE_BOOL(best_k, "Finds the best cluster count once common poses are found.",
                         false, false)

namespace {

using hyperpose::CommonPose;
using hyperpose::PoseSetEntry;


}  // namespace

class PrincipalPosesWalkApplication : public ozz::sample::Application {
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
    best_k_on_result_ = OPTIONS_best_k;
    walk_.root_motion = OPTIONS_root_motion;
    walk_.best_match = OPTIONS_best_match;
    if (OPTIONS_clip.value()[0]) {
      StartPreview(OPTIONS_clip.value(), 0.f);
    }
    return true;
  }

  bool OnUpdate(float _dt, float) override {
    if (k_scan_.valid() &&
        k_scan_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      PickBestK();
    }
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
      if (preview_.active) {
        // Selects the first pose without interrupting the preview.
        pose_ = result_.poses.empty() ? -1 : 0;
        frame_ = 0;
        transition_example_.clear();
      } else {
        SelectPose(result_.poses.empty() ? -1 : 0);
      }
      if (best_k_on_result_) {
        best_k_on_result_ = false;  // First result only.
        StartKScan();
      } else if (walk_on_result_ && !result_.poses.empty()) {
        StartWalk();  // First result only.
        walk_on_result_ = false;
      }
    }
    if (preview_.active) {
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
                 SegmentPrefix(segment) + ("#" + std::to_string(segment.from + 1)) +
                     " -> " + hyperpose::PoseName(segment.to) + "\n",
                 result_.clips[segment.clip], walk_.clip_time(), walk_.clip_duration());
      } else {
        AddLabel(math::Float4x4::identity(), make_span(models_), "", preview_.clip.path(),
                 preview_.clip.time, preview_.clip.duration());
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


  // Frame rate of _clip (see AnimationFrameRate), from the common pose
  // features or the previewed clip; 30 if neither has it.
  float FrameRate(const fs::path& _clip) const {
    for (size_t c = 0; c < result_.clips.size() && c < result_.features.size(); ++c) {
      if (result_.clips[c] == _clip) {
        return result_.features[c].frame_rate;
      }
    }
    if (preview_.clip.loaded() && preview_.clip.path() == _clip) {
      return hyperpose::AnimationFrameRate(preview_.clip.animation());
    }
    return 30.f;
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
    const float rate = FrameRate(_clip);
    const int frame = static_cast<int>(_time * rate + .5f);
    if (_duration >= 0.f) {
      std::snprintf(text, sizeof(text), "%s%s\n%.1f/%.1fs\nf %d/%d",
                    _prefix.c_str(), _clip.stem().string().c_str(), _time,
                    _duration, frame,
                    static_cast<int>(_duration * rate + .5f));
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
    const bool running = Busy();
    ImGui::BeginChild("browser", ImVec2(0.f, ImGui::GetContentRegionAvail().y * .4f),
                      ImGuiChildFlags_ResizeY);
    std::vector<fs::path> in_set;
    for (const PoseSetEntry& entry : project_.clips) {
      in_set.push_back(entry.path);
    }
    std::vector<fs::path> highlighted;
    if (preview_.active) {
      highlighted.push_back(preview_.clip.path());
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
    const int graph_pose = graph_.Draw(result_, pose_, library_);
    if (graph_pose >= 0) {
      SelectPose(graph_pose);
    }
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
    const auto it = std::find_if(project_.clips.begin(), project_.clips.end(),
                                 [&](const PoseSetEntry& _e) { return _e.path == _clip; });
    return it == project_.clips.end() ? nullptr : &*it;
  }

  // Starts clustering the pose set on a worker thread.
  void StartClustering() {
    if (project_.clips.empty() || Busy()) {
      return;
    }
    status_.clear();
    clustered_project_ = project_;
    job_ = std::async(std::launch::async, hyperpose::FindCommonPoses, project_.clips, &skeleton(),
                      project_.settings);
  }

  // A worker is running (clustering or the k scan): the set, settings and
  // skeleton must not change.
  bool Busy() const { return job_.valid() || k_scan_.valid(); }

  // Starts scanning cluster counts on a worker thread (see PickBestK).
  void StartKScan() {
    if (project_.clips.empty() || Busy()) {
      return;
    }
    status_.clear();
    k_scan_ = std::async(std::launch::async, [clips = project_.clips, skeleton = &skeleton(),
                                              settings = project_.settings]() {
      KScan scan;
      scan.measures = hyperpose::MeasureClusterCounts(
          clips, skeleton, settings, hyperpose::CommonPoseSettings::kMinClusters,
          hyperpose::CommonPoseSettings::kMaxClusters, false, &scan.error);
      return scan;
    });
  }

  // Takes the k scan result: clusters set to the elbow, common poses found.
  void PickBestK() {
    KScan scan = k_scan_.get();
    k_measures_ = std::move(scan.measures);
    best_k_ = hyperpose::ElbowClusterCount(k_measures_);
    if (best_k_ <= 0) {
      status_ = scan.error.empty() ? "Not enough cluster counts to choose from." : scan.error;
      return;
    }
    project_.settings.clusters = best_k_;
    ozz::log::Out() << "Best k: " << best_k_ << " (elbow of the RMS over k "
                    << k_measures_.front().clusters << ".." << k_measures_.back().clusters
                    << ")" << std::endl;
    StartClustering();
  }

  // Curve of the last k scan: RMS pose distance over k, the chosen k marked.
  void KScanGui() {
    if (k_measures_.empty()) {
      return;
    }
    std::vector<float> rms;
    for (const hyperpose::ClusterCountMeasure& measure : k_measures_) {
      rms.push_back(measure.rms * 100.f);
    }
    char overlay[64];
    std::snprintf(overlay, sizeof(overlay), "RMS over k %d..%d, best k %d",
                  k_measures_.front().clusters, k_measures_.back().clusters, best_k_);
    ImGui::PlotLines("##k_scan", rms.data(), static_cast<int>(rms.size()), 0, overlay, 0.f,
                     FLT_MAX, ImVec2(-FLT_MIN, ImGui::GetFrameHeight() * 2.5f));
    // Marker at the chosen k.
    const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
    const int first = k_measures_.front().clusters, last = k_measures_.back().clusters;
    if (last > first) {
      const float x = min.x + (max.x - min.x) * (best_k_ - first) / (last - first);
      ImGui::GetWindowDrawList()->AddLine(ImVec2(x, min.y), ImVec2(x, max.y),
                                          IM_COL32(255, 170, 60, 255), 1.5f);
    }
    if (ImGui::IsItemHovered()) {
      const float t = std::clamp((ImGui::GetIO().MousePos.x - min.x) / (max.x - min.x), 0.f, 1.f);
      const size_t i = std::min(k_measures_.size() - 1,
                                static_cast<size_t>(t * (k_measures_.size() - 1) + .5f));
      const hyperpose::ClusterCountMeasure& m = k_measures_[i];
      ImGui::SetTooltip(
          "k %d: RMS %.1f cm, transition entropy %.2f bits\n"
          "The elbow (orange) is where more clusters stop making the poses much\n"
          "tighter. Lower entropy: a more predictable transition graph.",
          m.clusters, m.rms * 100.f, m.entropy);
    }
  }

  // How a walk segment was reached, other than by a transition.
  static const char* SegmentPrefix(const hyperpose::PoseWalk::Segment& _segment) {
    return _segment.jump ? "jump " : _segment.after_end ? "after end " : "";
  }

  void LogResult() const {
    if (!result_.error.empty()) {
      ozz::log::Err() << result_.error << std::endl;
      return;
    }
    // Clip frame rates, as detected, and leading bind poses left out.
    std::map<float, int> rates;
    int bind_poses = 0;
    for (const hyperpose::PoseFeatures& features : result_.features) {
      ++rates[features.frame_rate];
      bind_poses += features.first_frame > 0;
    }
    std::string rates_text;
    for (const auto& [rate, clips] : rates) {
      rates_text += (rates_text.empty() ? "" : ", ") + std::to_string(clips) + " at " +
                    std::to_string(static_cast<int>(rate)) + " Hz";
    }
    if (bind_poses > 0) {
      rates_text += ", " + std::to_string(bind_poses) + " starting with a bind pose";
    }
    ozz::log::Out() << result_.poses.size() << " common poses of "
                    << result_.num_frames << " frames from "
                    << result_.clips.size() << " clips (" << rates_text << ") in "
                    << result_.seconds << "s:" << std::endl;
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
      ozz::log::Out() << "      ->" << hyperpose::PoseExitsText(exits, 4) << "  (" << exits.count
                      << " transitions)" << std::endl;
    }
  }

  bool ProjectDirty() const { return project_ != saved_project_; }

  std::string ProjectName() const {
    return project_path_.empty() ? "Untitled" : project_path_.stem().string();
  }

  // Empty, untitled project with default settings.
  void NewProject() {
    project_ = hyperpose::PoseProject();
    project_path_.clear();
    saved_project_ = project_;
  }

  // Loads a project's set and settings, then finds its common poses. Clips
  // that are missing or don't share the skeleton are left out (the project
  // then shows as changed).
  bool OpenProject(const fs::path& _path) {
    if (Busy()) {
      return false;
    }
    hyperpose::PoseProject project;
    std::string error;
    if (!hyperpose::LoadPoseProject(_path, library_.root(), &project, &error)) {
      status_ = error;
      return false;
    }
    // Clips added one by one: checks they exist and share the skeleton, and
    // reads their durations.
    project_ = hyperpose::PoseProject();
    project_.settings = project.settings;
    int missing = 0;
    for (const PoseSetEntry& clip : project.clips) {
      const size_t size = project_.clips.size();
      std::error_code ec;
      if (!fs::exists(clip.path, ec) || !Add(clip.path) || project_.clips.size() == size) {
        ++missing;
        continue;
      }
      project_.clips.back().begin = clip.begin;
      project_.clips.back().end = clip.end;
    }
    project_path_ = _path;
    saved_project_ = missing ? project : project_;
    hyperpose::SetRecentPoseProject(_path);
    status_ = "Opened " + ProjectName() + ": " + std::to_string(project_.clips.size()) + " clips";
    if (missing) {
      status_ += ", " + std::to_string(missing) + " missing or not loaded";
    }
    walk_.Stop();
    StartClustering();
    return true;
  }

  bool SaveProject(const fs::path& _path) {
    std::string error;
    if (!hyperpose::SavePoseProject(_path, project_, library_.root(), &error)) {
      status_ = error;
      return false;
    }
    project_path_ = _path;
    saved_project_ = project_;
    hyperpose::SetRecentPoseProject(_path);
    status_ = "Saved " + _path.string();
    return true;
  }

  // Project bar: carries out what it asks for.
  void ProjectGui() {
    const hyperpose::ProjectBar::Action action =
        project_bar_.Draw(projects_dir_, project_path_, ProjectDirty(), Busy());
    switch (action.type) {
      case hyperpose::ProjectBar::Action::kNew:
        NewProject();
        status_.clear();
        break;
      case hyperpose::ProjectBar::Action::kOpen:
        OpenProject(action.path);
        break;
      case hyperpose::ProjectBar::Action::kSave:
        SaveProject(action.path);
        break;
      case hyperpose::ProjectBar::Action::kNone:
        break;
    }
  }

  // Adds a clip to the pose set, or removes it if already in. All clips must
  // share a skeleton.
  void Toggle(const fs::path& _clip) {
    const auto it = std::find_if(project_.clips.begin(), project_.clips.end(), [&](const PoseSetEntry& _e) {
      return _e.path == _clip;
    });
    if (it != project_.clips.end()) {
      project_.clips.erase(it);
      return;
    }
    Add(_clip);
  }

  bool Add(const fs::path& _clip) {
    if (std::any_of(project_.clips.begin(), project_.clips.end(),
                    [&](const PoseSetEntry& _e) { return _e.path == _clip; })) {
      return true;
    }
    if (!UseSkeleton(_clip)) {
      return false;
    }
    float duration = 0.f;
    hyperpose::ReadClipDuration(_clip.string().c_str(), &duration);
    project_.clips.push_back({_clip, duration});
    status_.clear();
    return true;
  }

  // Loads the skeleton of _clip if not current. Switching is only allowed
  // while the pose set is empty, as all set clips must share a skeleton.
  bool UseSkeleton(const fs::path& _clip) {
    const auto use = clip_skeleton_.Use(_clip, library_.root(),
                                        project_.clips.empty() && !Busy());
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
      preview_.active = false;
      preview_.clip.Unload();
      pose_valid_ = false;
    }
    return true;
  }

  const ozz::animation::Skeleton& skeleton() const {
    return clip_skeleton_.skeleton();
  }

  void PoseSetGui() {
    float total = 0.f;  // Of the trimmed clips.
    for (const PoseSetEntry& entry : project_.clips) {
      total += entry.trim_end() - entry.begin;
    }
    char header[96];
    std::snprintf(header, sizeof(header), "Pose set: %d clips, %.0fs###set",
                  static_cast<int>(project_.clips.size()), total);
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
      project_.clips.clear();
    }

    // Set list, with trim ranges.
    if (!project_.clips.empty()) {
      const float line = ImGui::GetTextLineHeightWithSpacing();
      ImGui::BeginChild("set", ImVec2(0.f, std::min<float>(project_.clips.size(), 6.f) * line + line * .5f),
                        ImGuiChildFlags_Borders);
      for (size_t i = 0; i < project_.clips.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::SmallButton("x")) {
          project_.clips.erase(project_.clips.begin() + i);
          ImGui::PopID();
          break;
        }
        ImGui::SameLine();
        const std::string name = project_.clips[i].path.stem().string();
        char label[128];
        if (project_.clips[i].trimmed()) {
          std::snprintf(label, sizeof(label), "%s  %.1f-%.1fs of %.1fs", name.c_str(),
                        project_.clips[i].begin, project_.clips[i].trim_end(), project_.clips[i].duration);
        } else {
          std::snprintf(label, sizeof(label), "%s  %.1fs", name.c_str(), project_.clips[i].duration);
        }
        if (ImGui::Selectable(label, PreviewedClip() == project_.clips[i].path, 0,
                              ImVec2(ImGui::CalcTextSize(label).x, 0.f))) {
          PreviewClip(project_.clips[i].path);
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
    const bool running = Busy();
    ImGui::BeginDisabled(running);
    const float half = ImGui::GetContentRegionAvail().x * .35f;
    ImGui::SetNextItemWidth(half);
    ImGui::SliderInt("Clusters", &project_.settings.clusters,
                     hyperpose::CommonPoseSettings::kMinClusters,
                     hyperpose::CommonPoseSettings::kMaxClusters);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(half * .5f);
    ImGui::SliderInt("Step", &project_.settings.stride, 1,
                     hyperpose::CommonPoseSettings::kMaxStride,
                     project_.settings.stride == 1 ? "every frame" : "every %d frames");
    ImGui::SetItemTooltip(
        "Clip frames clustered: every frame, or every n-th to cluster large\n"
        "sets faster. Each clip's frame rate is detected from its keys.");
    ImGui::SetNextItemWidth(half);
    ImGui::InputInt("Seed", &project_.settings.seed);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(half * .5f);
    ImGui::SliderFloat("Velocity", &project_.settings.velocity_weight, 0.f, .5f, "%.2fs");
    ImGui::SetItemTooltip(
        "Weight of joint velocities against positions (seconds): a velocity\n"
        "difference of 1 m/s counts as this many meters of position\n"
        "difference, so the same pose moving in opposite directions can split\n"
        "into different common poses. 0: positions only.");
    ImGui::BeginDisabled(project_.clips.empty());
    if (ImGui::Button("Find common poses")) {
      StartClustering();
    }
    ImGui::SameLine();
    if (ImGui::Button("Find best k")) {
      StartKScan();
    }
    ImGui::SetItemTooltip(
        "Finds common poses for every cluster count (%d to %d) and sets Clusters\n"
        "to the elbow of the RMS pose distance curve: where more clusters stop\n"
        "making the poses much tighter. Then finds the common poses.",
        hyperpose::CommonPoseSettings::kMinClusters, hyperpose::CommonPoseSettings::kMaxClusters);
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    if (running) {
      ImGui::SameLine();
      ImGui::TextUnformatted(k_scan_.valid() ? "Scanning k..." : "Working...");
    }
    KScanGui();

    if (result_.poses.empty()) {
      return;
    }
    if (!running && project_ != clustered_project_) {
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
    ImGui::Checkbox("Transition graph", &graph_.open);
    ImGui::SetItemTooltip("Window with the common poses and the transitions between them.");
    ImGui::Separator();
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
          if (ImGui::Selectable(hyperpose::PoseName(edge.to).c_str())) {
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
    preview_.active = false;
    walk_.Stop();
    const CommonPose::Frame& frame = result_.poses[pose_].frames[_frame];
    SampleClip(result_.clips[frame.clip], frame.time);
  }

  // Previews a clip picked by the user.
  void PreviewClip(const fs::path& _clip) { StartPreview(_clip, 0.f); }

  // Clip being previewed (empty when showing a still frame).
  fs::path PreviewedClip() const { return preview_.active ? preview_.clip.path() : fs::path(); }

  // Re-samples the displayed pose (preview or selected frame). The walk
  // samples every update.
  void RefreshPose() {
    if (walk_.active()) {
      return;
    }
    if (preview_.active) {
      SampleClip(preview_.clip.path(), preview_.clip.time);
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
    preview_.active = false;
    if (!SampleClip(_clip, _time)) {  // Loads the clip.
      return;
    }
    preview_.active = true;
    if (walk_.root_motion) {
      // Root motion from the origin.
      AnchorPreview(_time, true);
      SampleClip(_clip, _time);
    }
    preview_.clip.playing = true;
  }

  // Pose walk: the main character walks the transition graph from the
  // selected pose (random if none), instead of the preview / still frame.
  void StartWalk() {
    preview_.active = false;
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
      ImGui::Text("%s#%d -> %s  %s @ %.1fs", SegmentPrefix(segment), segment.from + 1,
                  hyperpose::PoseName(segment.to).c_str(),
                  result_.clips[segment.clip].stem().string().c_str(), walk_.clip_time());
      if (segment.jump) {
        ImGui::SetItemTooltip("Nowhere to go from the last pose: jumped to #%d, a pose "
                              "the clips start in.", segment.from + 1);
      } else if (segment.after_end) {
        ImGui::SetItemTooltip("The previous clip ended in #%d: goes on from it with "
                              "another clip.", segment.from + 1);
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
    ImGui::Checkbox("Best match blend", &walk_.best_match);
    ImGui::SetItemTooltip(
        "Narrow phase: once the next transition is picked, blends at the closest\n"
        "pair of frames (joint positions and velocities): this clip from the\n"
        "horizon before it reaches the next pose to the horizon after, the\n"
        "next clip while it's in that pose. Carries on without a blend when\n"
        "the transition continues the same clip.\n"
        "Off: blends when the clip reaches the next pose, into where the next\n"
        "clip entered it.");
    if (walk_.best_match) {
      ImGui::SetNextItemWidth(width);
      ImGui::SliderFloat("##horizon", &walk_.match_horizon, .05f, 2.f, "horizon %.2fs");
      ImGui::SetItemTooltip("How far before and after the transition the blend can move.");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::SliderFloat("##velocity_weight", &walk_.velocity_weight, 0.f, 1.f,
                         "velocity weight %.2fs");
      ImGui::SetItemTooltip(
          "Weight of joint velocity differences against positions (seconds),\n"
          "so poses moving the opposite way don't match.");
    }
    {
      // Always a line, so the UI below doesn't move when the walk starts.
      const hyperpose::PoseWalk::Segment& segment = walk_.segment();
      if (!walk_.active() || (!segment.continued && segment.difference < 0.f)) {
        ImGui::TextDisabled("Last blend: -");
      } else if (segment.continued) {
        ImGui::TextDisabled("Last blend: none, the clip carried on.");
      } else {
        ImGui::TextDisabled("Last blend: %.1f cm pose difference.", segment.difference * 100.f);
      }
      const hyperpose::PoseWalk::BlendStats& stats = walk_.stats();
      ImGui::SetItemTooltip(
          "Pose difference where the walk blended into this segment: per joint\n"
          "RMS of position and weighted velocity differences, as the spread.\n"
          "Since the walk started: %d blends, mean %.1f cm; %d carried on.",
          stats.blends, stats.mean() * 100.f, stats.continued);
    }
    RootMotionGui();
    ImGui::TreePop();
  }

  // Root motion of the main character (preview and pose walk) and the
  // character's place, shared by both.
  void RootMotionGui() {
    if (ImGui::Checkbox("Root motion", &walk_.root_motion)) {
      // Continues from where the character stands.
      if (preview_.active && walk_.root_motion) {
        AnchorPreview(preview_.clip.time, false);
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
        if (preview_.active) {
          AnchorPreview(preview_.clip.time, true);
          RefreshPose();
        }
      }
    }
  }

  // Preview root motion: places the clip so that its root at _time stands at
  // the origin facing the default heading (_origin), or where the character
  // root is (last pose).
  void AnchorPreview(float _time, bool _origin) {
    preview_.Anchor(_time, !_origin && pose_valid_ ? &models_[0] : nullptr,
                    clip_skeleton_.heading_axis());
  }

  bool PreviewRootMotion() const { return preview_.active && walk_.root_motion; }

  void UpdatePreview(float _dt) {
    // Wrapped around (loop or trim range): root motion continues from where
    // the character is.
    if (preview_.Advance(_dt, FindEntry(preview_.clip.path())) && PreviewRootMotion()) {
      AnchorPreview(preview_.clip.time, false);
    }
    SampleClip(preview_.clip.path(), preview_.clip.time);
  }

  void PreviewGui() {
    if (!preview_.active ||
        !ImGui::CollapsingHeader("Preview", ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }
    const std::string name = preview_.clip.path().stem().string();
    ImGui::Text("%s  %s", name.c_str(), library_.Description(name).c_str());
    const hyperpose::ClipPreview::GuiResult result =
        preview_.Gui(FindEntry(preview_.clip.path()), Busy());
    if (result.close) {
      preview_.active = false;
      pose_valid_ = false;
      RefreshPose();  // Back to the selected frame, if any.
      return;
    }
    if (result.add_to_set) {
      Add(preview_.clip.path());
    }
    RootMotionGui();
    ImGui::Separator();
  }

  // Samples _clip at _time into models_: placed by the preview anchor with
  // root motion, else aligned to the origin if enabled.
  bool SampleClip(const fs::path& _clip, float _time) {
    pose_valid_ = false;
    if (_clip != preview_.clip.path() && !preview_.clip.Load(_clip, skeleton())) {
      status_ = "Failed to load " + _clip.filename().string();
      return false;
    }
    preview_.clip.time = _time;
    if (!preview_.clip.Sample(make_span(models_))) {
      return false;
    }
    if (PreviewRootMotion()) {
      preview_.ApplyAnchor(make_span(models_));
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

  // The pose set and its common pose settings: the open project, as edited.
  hyperpose::PoseProject project_;

  // Project file of the set and settings, as last opened / saved.
  fs::path projects_dir_;
  fs::path project_path_;  // Empty: untitled.
  hyperpose::PoseProject saved_project_;
  hyperpose::ProjectBar project_bar_;
  // Set and settings the common poses were found for, to flag stale ones.
  hyperpose::PoseProject clustered_project_;
  hyperpose::ClipSkeleton clip_skeleton_;

  // Clustering.
  std::future<hyperpose::CommonPoseResult> job_;
  // Cluster count scan (Find best k): its worker, last measures and choice.
  struct KScan {
    std::vector<hyperpose::ClusterCountMeasure> measures;
    std::string error;
  };
  std::future<KScan> k_scan_;
  std::vector<hyperpose::ClusterCountMeasure> k_measures_;
  int best_k_ = 0;
  hyperpose::CommonPoseResult result_;
  int pose_ = -1;   // Selected common pose.
  int frame_ = 0;   // Shown frame of the selected pose.

  // Every common pose, behind the displayed pose.
  hyperpose::PoseGallery gallery_;
  hyperpose::PoseWalk walk_;  // Main character pose walk, when active.
  bool walk_on_result_ = false;  // --walk: starts on the first result.
  bool best_k_on_result_ = false;  // --best_k: scans k after the first result.
  bool show_gallery_ = true;

  // Displayed pose: a clip preview (playing), or a still frame of the
  // selected common pose.
  hyperpose::ClipPreview preview_;  // Its clip also shows still frames.
  bool align_to_origin_ = true;
  bool pose_valid_ = false;
  ozz::vector<math::Float4x4> models_;

  // Debug labels of this frame (screen position, text).
  struct Label {
    ImVec2 pos;
    std::string text;
  };
  std::vector<Label> labels_;
  bool debug_text_ = true;
  hyperpose::TransitionGraph graph_;  // Transition graph window.
  std::vector<int> transition_example_;  // Per edge: next transition to play.
};

int main(int _argc, const char** _argv) {
  return PrincipalPosesWalkApplication().Run(_argc, _argv, "0.1", "Principal Poses Walk");
}
