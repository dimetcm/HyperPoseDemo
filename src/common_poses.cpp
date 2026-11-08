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
    if (OPTIONS_auto_set.value()[0]) {
      library_.SetFilter(OPTIONS_auto_set);
      for (const fs::path& clip : library_.FilteredClips()) {
        if (!Add(clip)) break;
      }
      StartClustering();
    }
    if (OPTIONS_clip.value()[0]) {
      StartPreview(OPTIONS_clip.value(), 0.f);
    }
    return true;
  }

  bool OnUpdate(float _dt, float) override {
    // Picks up the clustering result when the worker is done.
    if (job_.valid() &&
        job_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      result_ = job_.get();
      status_ = result_.error;
      LogResult();
      if (!gallery_.Build(result_, &skeleton(), clip_skeleton_.heading_axis())) {
        status_ = gallery_.error();
      }
      SelectPose(result_.poses.empty() ? -1 : 0);
    }
    if (previewing_) {
      UpdatePreview(_dt);
    }
    // Before posing the character for the displayed pose, as baking poses it.
    if (show_gallery_) {
      gallery_.Update(_dt, &character_);
    }
    if (pose_valid_) {
      character_.Update(make_span(models_));
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
      AddLabel(math::Float4x4::identity(), make_span(models_), "", clip_.path(),
               clip_.time, clip_.duration());
    }
    // Every common pose, in rows behind the displayed pose. Drawing the
    // animated gallery re-poses the character, so it's restored after.
    if (show_gallery_ && !gallery_.empty()) {
      success &= gallery_.Draw(_renderer, &character_, draw_mesh_, draw_skeleton_);
      if (pose_valid_ && gallery_.animate) {
        character_.Update(make_span(models_));
      }
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

    // WorldToScreen maps the projection to the whole window, bottom up; the
    // 3D view is the part right of the panel.
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    const math::Float2 screen = WorldToScreen(anchor);
    const float ndc_x = 2.f * screen.x / display.x - 1.f;
    const ImVec2 pos(gui_left_inset_ + (display.x - gui_left_inset_) * (ndc_x + 1.f) * .5f,
                     display.y - screen.y);

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
    labels_.push_back({pos, text});
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
      success &= character_.Draw(_renderer, _transform);
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
    DrawLabels();
    return true;
  }

  ozz::math::Box GetSceneBounds() const override {
    ozz::math::Box bounds;
    if (pose_valid_) {
      const auto& models =
          character_.posed() ? character_.joint_model_transforms() : models_;
      bounds = ozz::sample::ComputePostureBounds(make_span(models),
                                                 math::Float4x4::identity());
    }
    if (show_gallery_ && !gallery_.empty()) {
      // Clip skeleton poses: close enough to the character for framing.
      const ozz::math::Box box = gallery_.Bounds();
      bounds = bounds.is_valid() ? Merge(bounds, box) : box;
    }
    return bounds;
  }

 private:
  // Starts clustering the pose set on a worker thread.
  void StartClustering() {
    if (set_.empty() || job_.valid()) {
      return;
    }
    status_.clear();
    job_ = std::async(std::launch::async, hyperpose::FindCommonPoses, set_, &skeleton(),
                      kRates[rate_], k_, static_cast<uint32_t>(seed_));
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
    }
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
    float total = 0.f;
    for (const PoseSetEntry& entry : set_) {
      total += entry.duration;
    }
    char header[96];
    std::snprintf(header, sizeof(header), "Pose set: %d clips, %.0fs###set",
                  static_cast<int>(set_.size()), total);
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }
    ImGui::TextDisabled("Tick clips below to add them, click any clip to preview it.");
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

    // Set list (time ranges are kept for trimming).
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
        std::snprintf(label, sizeof(label), "%s  %.1fs", name.c_str(),
                      set_[i].duration);
        if (ImGui::Selectable(label, PreviewedClip() == set_[i].path, 0,
                              ImVec2(ImGui::CalcTextSize(label).x, 0.f))) {
          PreviewClip(set_[i].path);
        }
        ImGui::SetItemTooltip("Click to preview.");
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
    ImGui::Separator();
  }

  // Selects a common pose (-1 for none), showing its most typical frame.
  void SelectPose(int _pose) {
    pose_ = _pose;
    frame_ = 0;
    if (pose_ >= 0) {
      ShowFrame(frame_);
    }
  }

  // Shows a frame of the selected pose, still.
  void ShowFrame(int _frame) {
    previewing_ = false;
    const CommonPose::Frame& frame = result_.poses[pose_].frames[_frame];
    SampleClip(result_.clips[frame.clip], frame.time);
  }

  // Previews a clip picked by the user.
  void PreviewClip(const fs::path& _clip) { StartPreview(_clip, 0.f); }

  // Clip being previewed (empty when showing a still frame).
  fs::path PreviewedClip() const { return previewing_ ? clip_.path() : fs::path(); }

  // Re-samples the displayed pose (preview or selected frame).
  void RefreshPose() {
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
    previewing_ = SampleClip(_clip, _time);
    clip_.playing = true;
  }

  void UpdatePreview(float _dt) {
    clip_.Advance(_dt);
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
    if (ImGui::Button(clip_.playing ? "Pause" : "Play",
                      ImVec2(ImGui::GetFontSize() * 4.f, 0.f))) {
      if (!clip_.playing && !clip_.loop && clip_.time >= duration) {
        clip_.time = 0.f;  // Replays from the start.
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
    ImGui::Separator();
  }

  // Samples _clip at _time into models_, aligned to the origin if enabled.
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
    if (align_to_origin_) {
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
  bool show_gallery_ = true;

  // Displayed pose: a clip preview (playing), or a still frame of the
  // selected common pose.
  hyperpose::ClipPlayer clip_;
  bool previewing_ = false;
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
};

int main(int _argc, const char** _argv) {
  return CommonPosesApplication().Run(_argc, _argv, "0.1", "Common Poses");
}
