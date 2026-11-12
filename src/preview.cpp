// AnimPreview: browses ozz animation archives in a folder tree and plays the
// selected clip. Each clip uses the nearest skeleton.ozz in its folder or above.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <imgui.h>

#include "common/core/anim_utils.h"
#include "common/gui/character_view.h"
#include "common/gui/clip_library.h"
#include "common/core/clip_player.h"
#include "common/core/point_field.h"
#include "framework/application.h"
#include "framework/imgui.h"
#include "framework/renderer.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/maths/box.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/options/options.h"

namespace fs = std::filesystem;

OZZ_OPTIONS_DECLARE_STRING(root, "Folder to browse for .ozz animations.",
                           HYPERPOSE_ASSETS_DIR "/ozz", false)
OZZ_OPTIONS_DECLARE_STRING(clip, "Clip (.ozz) to load at startup.", "", false)
OZZ_OPTIONS_DECLARE_STRING(
    metadata, "CSV of clip descriptions (file_name,description).",
    HYPERPOSE_ASSETS_DIR "/HuggingFace/gbionics__cmu-fbx/metadata.csv", false)

namespace {

// Point field samplers selectable in the UI.
struct SamplerChoice {
  const char* label;
  const char* name;  // PointFieldGenerator::FindSampler name.
};
const SamplerChoice kSamplers[] = {{"Random", "random"},
                                   {"Uniform (blue noise)", "uniform"}};
const char* kMethods[] = {"Skinned (sample rest pose)", "Per frame (resample)"};

// Distinct, stable color per point index (golden ratio hue walk).
ozz::sample::Color PointColor(int _index) {
  const float h = std::fmod(_index * 0.618034f, 1.f) * 6.f;
  const float x = 1.f - std::fabs(std::fmod(h, 2.f) - 1.f);
  const int sector = static_cast<int>(h);
  const float rgb[6][3] = {{1, x, 0}, {x, 1, 0}, {0, 1, x},
                           {0, x, 1}, {x, 0, 1}, {1, 0, x}};
  const float* c = rgb[std::min(sector, 5)];
  return {.25f + .75f * c[0], .25f + .75f * c[1], .25f + .75f * c[2], 1.f};
}

double MillisecondsSince(std::chrono::steady_clock::time_point _start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - _start)
      .count();
}

}  // namespace

class PreviewApplication : public ozz::sample::Application {
 protected:
  bool OnInitialize() override {
    if (!library_.Scan(OPTIONS_root.value(), OPTIONS_metadata.value())) {
      status_ = "No .ozz clips found in " + std::string(OPTIONS_root.value());
    }
    character_.Scan(OPTIONS_root.value());
    if (OPTIONS_clip.value()[0]) {
      Load(fs::path(OPTIONS_clip.value()));
    }
    return true;
  }

  bool OnUpdate(float _dt, float) override {
    if (!clip_.loaded()) {
      return true;
    }
    controller_.Update(clip_.animation(), _dt);
    if (!clip_.Sample(controller_.time_ratio() * clip_.duration(),
                      make_span(models_))) {
      return false;
    }
    UpdatePointField();
    return true;
  }

  bool OnDisplay(ozz::sample::Renderer* _renderer) override {
    if (!clip_.loaded()) {
      return true;
    }
    const auto transform = ozz::math::Float4x4::identity();
    bool success = true;
    const bool show_character = ShowCharacter();
    if (show_character) {
      success &= character_.Draw(_renderer, make_span(models_), transform);
    }
    if (draw_skeleton_ || !show_character) {
      success &= _renderer->DrawPosture(clip_skeleton_.skeleton(),
                                        make_span(models_), transform);
    }
    const PointFieldPreview& pf = point_field_;
    if (pf.enabled && !pf.points.empty()) {
      const ozz::sample::Color single = {1.f, .85f, .2f, 1.f};
      success &= _renderer->DrawPoints(
          ozz::span<const float>(&pf.points[0].x, pf.points.size() * 3),
          ozz::span<const float>(&pf.point_size, 1),
          pf.color_by_id ? make_span(pf.colors)
                         : ozz::span<const ozz::sample::Color>(&single, 1),
          transform, true);
    }
    return success;
  }

  bool OnGui(ozz::sample::ImGui* _im_gui) override {
    if (!clip_.loaded()) {
      return true;
    }
    static bool open = true;
    ozz::sample::ImGui::OpenClose oc(_im_gui, "Animation control", &open);
    if (open) {
      controller_.OnGui(clip_.animation(), _im_gui);
    }
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
    const float min_width = 150.f * dpi;
    ImGui::SetNextWindowPos(ImVec2(0.f, 0.f));
    ImGui::SetNextWindowSize(ImVec2(560.f * dpi, io.DisplaySize.y),
                             ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(min_width, io.DisplaySize.y),
        ImVec2(io.DisplaySize.x * .6f, io.DisplaySize.y));
    ImGui::Begin("Clips", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    gui_left_inset_ = ImGui::GetWindowWidth();

    library_.DrawFilter();
    if (!status_.empty()) {
      ImGui::TextWrapped("%s", status_.c_str());
    }
    ImGui::Separator();

    // Character selection; clips from other skeletons are retargeted onto it.
    character_.DrawGui();
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Point field")) {
      PointFieldGui();
      ImGui::Separator();
    }

    fs::path clicked;
    if (library_.DrawTable({selected_}, &clicked) && clicked != selected_) {
      Load(clicked);
    }

    ImGui::End();
    return true;
  }

  ozz::math::Box GetSceneBounds() const override {
    if (!clip_.loaded()) {
      return ozz::math::Box();
    }
    return character_.Bounds(make_span(models_), ozz::math::Float4x4::identity());
  }

 private:
  bool ShowCharacter() const { return draw_mesh_ && character_.posed(); }

  // Skins the character mesh and places the point field on it. Samples are
  // regenerated when settings change, and every new frame in per frame mode.
  void UpdatePointField() {
    PointFieldPreview& pf = point_field_;
    if (pf.character_generation != character_.generation()) {
      pf.character_generation = character_.generation();
      hyperpose::BuildRestMesh(character_.meshes(), &pf.rest_mesh);
      pf.posed_mesh = pf.rest_mesh;
      pf.dirty = true;
    }
    if (!pf.enabled || !character_.Pose(make_span(models_), &pf.character_models)) {
      pf.points.clear();
      return;
    }
    auto start = std::chrono::steady_clock::now();
    if (!hyperpose::PoseMesh(character_.meshes(), make_span(pf.character_models),
                             &pf.posed_mesh)) {
      pf.points.clear();
      return;
    }
    pf.skinning_ms = MillisecondsSince(start);

    const bool per_frame = pf.method == 1;
    const float ratio = controller_.time_ratio();
    if (pf.dirty || (per_frame && ratio != pf.sampled_ratio)) {
      const auto sampler =
          hyperpose::PointFieldGenerator::FindSampler(kSamplers[pf.sampler].name);
      // Per frame mode reseeds with the frame index, like pointfield_gen.
      const int frame = static_cast<int>(ratio * clip_.duration() * 30.f);
      start = std::chrono::steady_clock::now();
      pf.samples = per_frame ? sampler(pf.posed_mesh, pf.count, pf.seed + frame)
                             : sampler(pf.rest_mesh, pf.count, pf.seed);
      pf.sampling_ms = MillisecondsSince(start);
      pf.sampled_ratio = ratio;
      pf.dirty = false;
      pf.spacing_valid = false;
    }
    hyperpose::PointFieldGenerator::Evaluate(pf.posed_mesh,
                                             make_span(pf.samples), &pf.points);
    if (pf.colors.size() != pf.points.size()) {
      pf.colors.resize(pf.points.size());
      for (size_t i = 0; i < pf.colors.size(); ++i) {
        pf.colors[i] = PointColor(static_cast<int>(i));
      }
    }
  }

  void PointFieldGui() {
    PointFieldPreview& pf = point_field_;
    if (!character_.posed()) {
      ImGui::TextDisabled("Select a character and a clip.");
    }
    if (ImGui::Checkbox("Enabled", &pf.enabled)) {
      pf.dirty = true;
    }
    ImGui::PushItemWidth(-ImGui::CalcTextSize("Point size").x -
                         ImGui::GetStyle().ItemInnerSpacing.x);
    pf.dirty |= ImGui::Combo("Method", &pf.method, kMethods,
                             IM_ARRAYSIZE(kMethods));
    if (ImGui::BeginCombo("Sampler", kSamplers[pf.sampler].label)) {
      for (int i = 0; i < IM_ARRAYSIZE(kSamplers); ++i) {
        if (ImGui::Selectable(kSamplers[i].label, pf.sampler == i)) {
          pf.dirty |= pf.sampler != i;
          pf.sampler = i;
        }
      }
      ImGui::EndCombo();
    }
    // Applied on release: uniform sampling of many points takes a while.
    ImGui::SliderInt("Count", &pf.pending_count, 1, 500, "%d",
                     ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemDeactivatedAfterEdit() && pf.pending_count != pf.count) {
      pf.count = pf.pending_count;
      pf.dirty = true;
    }
    pf.dirty |= ImGui::InputInt("Seed", &pf.seed);
    ImGui::SliderFloat("Point size", &pf.point_size, 1.f, 10.f, "%.1f px");
    ImGui::PopItemWidth();
    ImGui::Checkbox("Color by point", &pf.color_by_id);
    ImGui::SetItemTooltip(
        "Each point keeps its color: stable colors mean points keep their "
        "identity across frames.");
    ImGui::Checkbox("Draw character mesh", &draw_mesh_);

    if (pf.enabled && !pf.points.empty()) {
      ImGui::Text("Sampling %.1f ms, skinning %.2f ms", pf.sampling_ms,
                  pf.skinning_ms);
      if (ImGui::Button("Measure spacing")) {
        pf.spacing = hyperpose::PointFieldGenerator::MeasureSpacing(
            make_span(pf.points));
        pf.spacing_valid = true;
      }
      ImGui::SetItemTooltip(
          "Nearest neighbor distances on the current frame. cv (std/mean) "
          "measures evenness: ~0.5 random, lower is more even.");
      if (pf.spacing_valid) {
        ImGui::Text("nn %.1f mm, cv %.3f, min %.2f mm", pf.spacing.mean * 1000.f,
                    pf.spacing.cv, pf.spacing.min * 1000.f);
      }
    }
  }

  void Load(const fs::path& _clip) {
    selected_ = _clip;
    clip_.Unload();
    const auto use = clip_skeleton_.Use(_clip, library_.root(), true);
    if (use == hyperpose::ClipSkeleton::Result::kFailed) {
      status_ = clip_skeleton_.error();
      return;
    }
    if (use == hyperpose::ClipSkeleton::Result::kChanged) {
      models_.resize(clip_skeleton_.num_joints());
      character_.SetClipSkeleton(&clip_skeleton_.skeleton(), clip_skeleton_.path());
    }
    if (!clip_.Load(_clip, clip_skeleton_.skeleton())) {
      status_ = "Failed to load " + _clip.filename().string() +
                " (or it doesn't match " + clip_skeleton_.path().filename().string() +
                ")";
      return;
    }
    controller_.set_time_ratio(0.f);

    char info[128];
    std::snprintf(info, sizeof(info), "%s: %.1fs, %d joints",
                  _clip.stem().string().c_str(), clip_.duration(),
                  clip_skeleton_.num_joints());
    status_ = info;
    const std::string& description = library_.Description(_clip.stem().string());
    if (!description.empty()) {
      status_ += "\n" + description;
    }
  }

  hyperpose::ClipLibrary library_;
  fs::path selected_;
  std::string status_;

  // Playback: time from the ozz playback controller (its gui), sampled by the
  // clip player.
  hyperpose::ClipSkeleton clip_skeleton_;
  hyperpose::ClipPlayer clip_;
  ozz::sample::PlaybackController controller_;
  ozz::vector<ozz::math::Float4x4> models_;

  // Character: skeleton + skinned mesh, driven directly or retargeted.
  hyperpose::CharacterView character_;
  bool draw_mesh_ = true;
  bool draw_skeleton_ = false;

  // Point field preview, generated live on the character.
  struct PointFieldPreview {
    bool enabled = false;
    int method = 0;   // kMethods index.
    int sampler = 1;  // kSamplers index.
    int count = 500;
    int pending_count = 500;  // Slider value, applied on release.
    int seed = 1;
    float point_size = 3.f;
    bool color_by_id = true;

    int character_generation = -1;  // Rest mesh built for this character.
    hyperpose::TriMesh rest_mesh;
    hyperpose::TriMesh posed_mesh;
    ozz::vector<ozz::math::Float4x4> character_models;  // Character pose.
    ozz::vector<hyperpose::SurfacePoint> samples;
    ozz::vector<ozz::math::Float3> points;
    ozz::vector<ozz::sample::Color> colors;
    bool dirty = true;
    float sampled_ratio = -1.f;
    double sampling_ms = 0.;
    double skinning_ms = 0.;
    hyperpose::PointFieldGenerator::Spacing spacing = {};
    bool spacing_valid = false;
  } point_field_;
};

int main(int _argc, const char** _argv) {
  return PreviewApplication().Run(_argc, _argv, "0.1", "HyperPose Preview");
}
