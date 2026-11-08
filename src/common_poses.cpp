// Common Poses: finds the most common poses in a set of clips. Every frame of
// the selected clips is turned into a pose feature (root relative, heading
// aligned joint positions), then grouped with k-means. Clusters are listed by
// size; each shows its most typical frame, and its members can be browsed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <imgui.h>

#include "common/anim_utils.h"
#include "common/character_view.h"
#include "common/clip_library.h"
#include "common/clip_player.h"
#include "common/pose_cluster.h"
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

// A clip of the pose set, with the time range to consider (trimming).
struct PoseSetEntry {
  fs::path path;
  float duration = 0.f;
  float begin = 0.f;
  float end = -1.f;  // Negative: clip end.
};

// A common pose: the pose set frames that look alike, as clip + time, ordered
// by distance to the pose (cluster center), closest (most typical) first.
struct CommonPose {
  struct Frame {
    int clip;        // Index in ClusterResult::clips.
    float time;      // Seconds.
    float distance;  // RMS joint distance to the cluster center (m).
  };
  std::vector<Frame> frames;
  float spread = 0.f;  // RMS of frame distances (m).
};

// Clustering output.
struct ClusterResult {
  std::vector<fs::path> clips;
  std::vector<CommonPose> poses;  // Most frames first.
  int num_frames = 0;
  double seconds = 0.;
  std::string error;
};

// Frame rate the clips were converted at (Tools/convert_fbx.py), for frame
// numbers in debug text.
const float kClipFrameRate = 30.f;

// Cluster gallery layout: poses per row, and the side (z sign) rows go to.
// Aligned poses face +z, so -z is behind.
const int kGalleryColumns = 8;
const float kGallerySide = -1.f;

const float kRates[] = {5.f, 10.f, 15.f};
const char* kRateLabels[] = {"5 Hz", "10 Hz", "15 Hz"};

// Extracts pose features of every entry frame, then clusters them. Runs on a
// worker thread: only reads its arguments.
ClusterResult FindCommonPoses(std::vector<PoseSetEntry> _entries,
                              const ozz::animation::Skeleton* _skeleton,
                              float _rate, int _k, uint32_t _seed) {
  const auto start = std::chrono::steady_clock::now();
  ClusterResult result;
  struct Sample {
    int clip;
    float time;
  };
  std::vector<Sample> samples;
  std::vector<float> data;
  ozz::animation::Animation animation;
  hyperpose::PoseFeatures features;
  for (size_t e = 0; e < _entries.size(); ++e) {
    const PoseSetEntry& entry = _entries[e];
    result.clips.push_back(entry.path);
    if (!ozz::sample::LoadAnimation(entry.path.string().c_str(), &animation) ||
        animation.num_tracks() != _skeleton->num_joints() ||
        !hyperpose::ExtractPoseFeatures(animation, *_skeleton, _rate, &features)) {
      result.error = "Failed to load " + entry.path.filename().string();
      return result;
    }
    const float end = entry.end < 0.f ? animation.duration() : entry.end;
    for (int frame = 0; frame < features.num_frames; ++frame) {
      const float time = frame / _rate;
      if (time < entry.begin || time > end) {
        continue;
      }
      samples.push_back({static_cast<int>(e), time});
      const math::Float3* p = &features.positions[frame * features.num_joints];
      for (int j = 0; j < features.num_joints; ++j) {
        data.insert(data.end(), {p[j].x, p[j].y, p[j].z});
      }
    }
  }
  result.num_frames = static_cast<int>(samples.size());
  for (const hyperpose::PoseCluster& cluster :
       hyperpose::ClusterPoses(data, _skeleton->num_joints(), _k, _seed)) {
    CommonPose pose;
    pose.spread = cluster.spread;
    pose.frames.reserve(cluster.members.size());
    for (size_t m = 0; m < cluster.members.size(); ++m) {
      const Sample& sample = samples[cluster.members[m]];
      pose.frames.push_back({sample.clip, sample.time, cluster.distances[m]});
    }
    result.poses.push_back(std::move(pose));
  }
  result.seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  return result;
}

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
      BuildClusterPoses();
      BuildGallery();
      SelectPose(result_.poses.empty() ? -1 : 0);
    }
    if (previewing_) {
      UpdatePreview(_dt);
    }
    if (show_gallery_ && animate_gallery_) {
      UpdateGallery(_dt);
    }
    // Before posing the character for the displayed pose, as baking poses it.
    if (show_gallery_ && !animate_gallery_ && !cluster_poses_.empty() &&
        character_.posed() && !GalleryBaked()) {
      BakeClusterPoses();
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
    // Every common pose, in rows behind the displayed pose. Animated: random
    // frames of the pose, skinned live (the character is re-posed for each,
    // then restored). Static: most typical frame, skinned once.
    if (show_gallery_ && animate_gallery_ && !gallery_.empty()) {
      const bool show_character = draw_mesh_ && character_.posed();
      for (size_t c = 0; c < gallery_.size(); ++c) {
        const math::Float4x4 transform = GalleryTransform(static_cast<int>(c));
        const auto models = make_span(gallery_[c]->models);
        if (show_character) {
          character_.Update(models);
          success &= character_.Draw(_renderer, transform);
        }
        if (draw_skeleton_ || !show_character) {
          success &= _renderer->DrawPosture(skeleton(), models, transform);
        }
        const GallerySlot& slot = *gallery_[c];
        AddLabel(transform, models, "#" + std::to_string(c + 1) + " ",
                 slot.clip.path(), slot.clip.time, slot.clip.duration());
      }
      if (pose_valid_) {
        character_.Update(make_span(models_));
      }
    } else if (show_gallery_ && !cluster_poses_.empty()) {
      const bool show_character = draw_mesh_ && GalleryBaked();
      for (size_t c = 0; c < cluster_poses_.size(); ++c) {
        const math::Float4x4 transform = GalleryTransform(static_cast<int>(c));
        if (show_character) {
          success &= hyperpose::CharacterView::DrawBaked(
              _renderer, cluster_meshes_[c], transform);
        }
        if (draw_skeleton_ || !show_character) {
          success &= _renderer->DrawPosture(
              skeleton(), make_span(cluster_poses_[c]), transform);
        }
        const CommonPose::Frame& frame = result_.poses[c].frames.front();
        AddLabel(transform, make_span(cluster_poses_[c]),
                 "#" + std::to_string(c + 1) + " ", result_.clips[frame.clip],
                 frame.time, -1.f);
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

  // Gallery slot of cluster _c: rows of up to kGalleryColumns, largest
  // cluster first, left to right, starting one row behind the origin.
  math::Float4x4 GalleryTransform(int _c) const {
    const int count = static_cast<int>(result_.poses.size());
    const int columns = std::max(1, std::min(count, kGalleryColumns));
    const int row = _c / columns;
    const int column = _c % columns;
    const int in_row = std::min(columns, count - row * columns);
    const float x = (column - (in_row - 1) * .5f) * gallery_spacing_;
    const float z = kGallerySide * (row + 1) * gallery_spacing_ * 1.5f;
    return math::Float4x4::Translation(math::simd_float4::Load(x, 0.f, z, 1.f));
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
    if (show_gallery_ && !cluster_poses_.empty()) {
      // Clip skeleton poses: close enough to the character for framing.
      for (size_t c = 0; c < cluster_poses_.size(); ++c) {
        const ozz::math::Box box = ozz::sample::ComputePostureBounds(
            make_span(cluster_poses_[c]), GalleryTransform(static_cast<int>(c)));
        bounds = bounds.is_valid() ? Merge(bounds, box) : box;
      }
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
    job_ = std::async(std::launch::async, FindCommonPoses, set_, &skeleton(),
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
      result_ = ClusterResult();
      cluster_poses_.clear();
      gallery_.clear();
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
    ImGui::Checkbox("Animate", &animate_gallery_);
    ImGui::SetItemTooltip(
        "Each pose plays its clips from a random frame of the pose, switching\n"
        "after the play time. Frames closer to the pose are picked more often.\n"
        "Off: most typical frame, still (cheaper to draw).");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::SliderFloat("##spacing", &gallery_spacing_, .5f, 3.f, "spacing %.1f m");
    ImGui::BeginDisabled(!animate_gallery_);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * .5f);
    ImGui::SliderFloat("Play time", &play_duration_, .1f, 10.f, "%.1f s",
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

  // One gallery character per common pose, playing random frames of it.
  struct GallerySlot {
    std::discrete_distribution<int> picker;  // Frame index, weighted.
    int frame = -1;
    float timer = 0.f;  // Until the next pick (s).
    hyperpose::ClipPlayer clip;
    ozz::vector<math::Float4x4> models;
  };

  void BuildGallery() {
    gallery_.clear();
    std::uniform_real_distribution<float> offset(0.f, play_duration_);
    for (const CommonPose& pose : result_.poses) {
      auto slot = std::make_unique<GallerySlot>();
      // Weights: gaussian of the frame distance to the pose, with the pose
      // spread as deviation, so typical frames come up more often.
      const float sigma = std::max(pose.spread, 1e-4f);
      std::vector<double> weights;
      weights.reserve(pose.frames.size());
      for (const CommonPose::Frame& frame : pose.frames) {
        const double x = frame.distance / sigma;
        weights.push_back(std::exp(-.5 * x * x));
      }
      slot->picker = std::discrete_distribution<int>(weights.begin(), weights.end());
      slot->models.resize(skeleton().num_joints());
      // Staggered, so the poses don't all switch at once.
      slot->timer = offset(rng_);
      PickGalleryFrame(pose, slot.get());
      gallery_.push_back(std::move(slot));
    }
  }

  // Starts playing a weighted random frame of _pose in _slot.
  bool PickGalleryFrame(const CommonPose& _pose, GallerySlot* _slot) {
    _slot->frame = _slot->picker(rng_);
    const CommonPose::Frame& frame = _pose.frames[_slot->frame];
    const fs::path& clip = result_.clips[frame.clip];
    if (clip != _slot->clip.path() && !_slot->clip.Load(clip, skeleton())) {
      return false;
    }
    _slot->clip.time = frame.time;
    return true;
  }

  void UpdateGallery(float _dt) {
    for (size_t c = 0; c < gallery_.size(); ++c) {
      GallerySlot& slot = *gallery_[c];
      slot.timer -= _dt;
      slot.clip.Advance(_dt);  // Loops.
      if (slot.timer <= 0.f) {
        // Re-armed from the current play time, which the user may change.
        slot.timer = play_duration_;
        PickGalleryFrame(result_.poses[c], &slot);
      }
      // A failed load keeps the last pose.
      if (slot.clip.Sample(make_span(slot.models))) {
        AlignToOrigin(make_span(slot.models));
      }
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

  // Samples the typical pose of every cluster, aligned to the origin.
  void BuildClusterPoses() {
    cluster_poses_.clear();
    cluster_meshes_.clear();  // Re-baked on next update.
    hyperpose::ClipPlayer player;
    for (const CommonPose& pose : result_.poses) {
      const CommonPose::Frame& sample = pose.frames.front();  // Most typical.
      const fs::path& clip = result_.clips[sample.clip];
      ozz::vector<math::Float4x4> models(skeleton().num_joints());
      if ((clip != player.path() && !player.Load(clip, skeleton())) ||
          !player.Sample(sample.time, make_span(models))) {
        status_ = "Failed to load " + clip.filename().string();
        cluster_poses_.clear();
        return;
      }
      AlignToOrigin(make_span(models));
      cluster_poses_.push_back(std::move(models));
    }
  }

  // Skins the character once in each cluster pose, so the gallery is drawn
  // without per frame skinning. Redone when the character changes.
  void BakeClusterPoses() {
    cluster_meshes_.assign(cluster_poses_.size(), {});
    for (size_t c = 0; c < cluster_poses_.size(); ++c) {
      character_.Update(make_span(cluster_poses_[c]));
      if (!character_.Bake(&cluster_meshes_[c])) {
        cluster_meshes_.clear();
        break;
      }
    }
    baked_generation_ = character_.generation();
  }

  bool GalleryBaked() const {
    return baked_generation_ == character_.generation() &&
           cluster_meshes_.size() == cluster_poses_.size();
  }

  // Moves the root above the origin, facing the default heading, so poses
  // compare from the same viewpoint.
  void AlignToOrigin(ozz::span<math::Float4x4> _models) const {
    math::Float3 root;
    math::Store3PtrU(_models[0].cols[3], &root.x);
    const float heading =
        hyperpose::RootHeading(_models[0], clip_skeleton_.heading_axis());
    const math::Float4x4 align =
        math::Float4x4::FromAxisAngle(math::simd_float4::y_axis(),
                                      math::simd_float4::Load1(-heading)) *
        math::Float4x4::Translation(math::simd_float4::Load(-root.x, 0.f, -root.z, 1.f));
    for (auto& model : _models) {
      model = align * model;
    }
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
      AlignToOrigin(make_span(models_));
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
  std::future<ClusterResult> job_;
  ClusterResult result_;
  int pose_ = -1;   // Selected common pose.
  int frame_ = 0;   // Shown frame of the selected pose.


  // Typical pose of each cluster (clip skeleton, aligned to origin).
  std::vector<ozz::vector<math::Float4x4>> cluster_poses_;
  std::vector<ozz::vector<ozz::sample::Mesh>> cluster_meshes_;  // Baked.
  int baked_generation_ = -1;  // Character generation of cluster_meshes_.
  bool show_gallery_ = true;
  bool animate_gallery_ = true;
  float gallery_spacing_ = 1.f;
  std::vector<std::unique_ptr<GallerySlot>> gallery_;  // Per common pose.
  std::mt19937 rng_{std::random_device{}()};

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
  float play_duration_ = 1.f;  // Gallery: play time of each picked frame (s).
};

int main(int _argc, const char** _argv) {
  return CommonPosesApplication().Run(_argc, _argv, "0.1", "Common Poses");
}
