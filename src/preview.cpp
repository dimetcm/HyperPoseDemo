// AnimPreview: browses ozz animation archives in a folder tree and plays the
// selected clip. Each clip uses the nearest skeleton.ozz in its folder or above.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <imgui.h>

#include "common/anim_utils.h"
#include "common/point_field.h"
#include "common/retarget.h"
#include "framework/application.h"
#include "framework/imgui.h"
#include "framework/mesh.h"
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

namespace {

constexpr const char* kSkeletonName = "skeleton.ozz";
constexpr const char* kMeshName = "mesh.ozz";

struct Clip {
  fs::path path;
  std::string name;  // File name without extension.
  float duration;
};

struct Folder {
  std::string name;
  std::vector<Folder> folders;
  std::vector<Clip> clips;
  int num_clips = 0;  // Including sub-folders.
};

// Recursively collects animation clips, skipping other archives (skeletons,
// meshes), hidden entries and folders without clips.
bool ScanFolder(const fs::path& _path, Folder* _folder) {
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
      if (hyperpose::ReadClipDuration(entry.path().string().c_str(), &duration)) {
        _folder->clips.push_back(
            {entry.path(), entry.path().stem().string(), duration});
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

// A character is a folder holding both a skeleton and a skinned mesh.
struct Character {
  std::string name;
  fs::path skeleton;
  fs::path mesh;
};

void ScanCharacters(const fs::path& _path, std::vector<Character>* _characters) {
  if (fs::exists(_path / kSkeletonName) && fs::exists(_path / kMeshName)) {
    _characters->push_back(
        {_path.filename().string(), _path / kSkeletonName, _path / kMeshName});
  }
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(_path, ec)) {
    if (entry.is_directory() && entry.path().filename().string()[0] != '.') {
      ScanCharacters(entry.path(), _characters);
    }
  }
}

}  // namespace

class PreviewApplication : public ozz::sample::Application {
 protected:
  bool OnInitialize() override {
    root_path_ = fs::path(OPTIONS_root.value());
    if (!ScanFolder(root_path_, &root_)) {
      status_ = "No .ozz clips found in " + root_path_.string();
    }
    ScanCharacters(root_path_, &characters_);
    if (!characters_.empty()) {
      LoadCharacter(0);
    }
    return true;
  }

  bool OnUpdate(float _dt, float) override {
    if (!loaded_) {
      return true;
    }
    controller_.Update(animation_, _dt);
    if (!hyperpose::SamplePose(animation_, skeleton_, controller_.time_ratio(),
                               &context_, make_span(locals_),
                               make_span(models_))) {
      return false;
    }

    if (CharacterPosed() && !SameSkeleton()) {
      retargeter_.Run(make_span(models_), make_span(character_models_));
    }
    UpdatePointField();
    return true;
  }

  bool OnDisplay(ozz::sample::Renderer* _renderer) override {
    if (!loaded_) {
      return true;
    }
    const auto transform = ozz::math::Float4x4::identity();
    bool success = true;
    const bool show_character = ShowCharacter();
    if (show_character) {
      const auto& models = CharacterModels();
      for (const ozz::sample::Mesh& mesh : meshes_) {
        hyperpose::ComputeSkinningMatrices(mesh, make_span(models),
                                           make_span(skinning_matrices_));
        success &= _renderer->DrawSkinnedMesh(
            mesh, make_span(skinning_matrices_), transform);
      }
    }
    if (draw_skeleton_ || !CharacterPosed()) {
      success &= _renderer->DrawPosture(skeleton_, make_span(models_), transform);
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
    if (!loaded_) {
      return true;
    }
    static bool open = true;
    ozz::sample::ImGui::OpenClose oc(_im_gui, "Animation control", &open);
    if (open) {
      controller_.OnGui(animation_, _im_gui);
    }
    if (!meshes_.empty()) {
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
    ImGui::SetNextWindowSize(ImVec2(220.f * dpi, io.DisplaySize.y),
                             ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(min_width, io.DisplaySize.y),
        ImVec2(io.DisplaySize.x * .6f, io.DisplaySize.y));
    ImGui::Begin("Clips", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    gui_left_inset_ = ImGui::GetWindowWidth();

    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputTextWithHint("##filter", "Filter clips", filter_.InputBuf,
                                 IM_ARRAYSIZE(filter_.InputBuf))) {
      filter_.Build();
    }
    if (!status_.empty()) {
      ImGui::TextWrapped("%s", status_.c_str());
    }
    ImGui::Separator();

    // Character selection; clips from other skeletons are retargeted onto it.
    const char* none = "None (skeleton only)";
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##character", character_ >= 0
                                             ? characters_[character_].name.c_str()
                                             : none)) {
      if (ImGui::Selectable(none, character_ < 0)) {
        character_ = -1;
      }
      for (int i = 0; i < static_cast<int>(characters_.size()); ++i) {
        if (ImGui::Selectable(characters_[i].name.c_str(), character_ == i)) {
          LoadCharacter(i);
        }
      }
      ImGui::EndCombo();
    }
    if (character_ >= 0 && !character_status_.empty()) {
      ImGui::TextWrapped("%s", character_status_.c_str());
    }
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Point field")) {
      PointFieldGui();
      ImGui::Separator();
    }

    const ImGuiTableFlags table_flags =
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;
    if (ImGui::BeginTable("clips", 2, table_flags)) {
      ImGui::TableSetupScrollFreeze(0, 1);
      ImGui::TableSetupColumn("Clip", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Duration", ImGuiTableColumnFlags_WidthFixed,
                              ImGui::CalcTextSize("Duration").x);
      ImGui::TableHeadersRow();
      DrawFolder(root_, root_path_, true);
      ImGui::EndTable();
    }

    ImGui::End();
    return true;
  }

  ozz::math::Box GetSceneBounds() const override {
    if (!loaded_) {
      return ozz::math::Box();
    }
    const auto& models = CharacterPosed() ? CharacterModels() : models_;
    return ozz::sample::ComputePostureBounds(make_span(models),
                                             ozz::math::Float4x4::identity());
  }

 private:
  bool SameSkeleton() const { return skeleton_path_ == character_skeleton_path_; }

  // The character can be posed when it shares the clip skeleton, or when the
  // clip could be retargeted onto it.
  bool CharacterPosed() const {
    return loaded_ && character_ >= 0 && !meshes_.empty() &&
           (SameSkeleton() || retarget_ok_);
  }

  bool ShowCharacter() const { return draw_mesh_ && CharacterPosed(); }

  // Skins the character mesh and places the point field on it. Samples are
  // regenerated when settings change, and every new frame in per frame mode.
  void UpdatePointField() {
    PointFieldPreview& pf = point_field_;
    if (!pf.enabled || !CharacterPosed()) {
      pf.points.clear();
      return;
    }
    auto start = std::chrono::steady_clock::now();
    if (!hyperpose::PoseMesh(meshes_, make_span(CharacterModels()),
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
      const int frame = static_cast<int>(ratio * animation_.duration() * 30.f);
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
    if (!CharacterPosed()) {
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

  const ozz::vector<ozz::math::Float4x4>& CharacterModels() const {
    return SameSkeleton() ? models_ : character_models_;
  }

  void LoadCharacter(int _index) {
    character_ = _index;
    const Character& character = characters_[_index];
    character_skeleton_path_.clear();
    meshes_.clear();
    if (!ozz::sample::LoadSkeleton(character.skeleton.string().c_str(),
                                   &character_skeleton_)) {
      character_status_ = "Failed to load " + character.skeleton.string();
      return;
    }
    character_skeleton_path_ = character.skeleton;
    character_models_.resize(character_skeleton_.num_joints());
    LoadMeshes(character.mesh);
    hyperpose::BuildRestMesh(meshes_, &point_field_.rest_mesh);
    point_field_.posed_mesh = point_field_.rest_mesh;
    point_field_.dirty = true;
    SetupRetarget();
  }

  // Rebuilds the retargeting when the clip skeleton or the character changes.
  void SetupRetarget() {
    retarget_ok_ = false;
    character_status_.clear();
    if (character_ < 0) {
      return;
    }
    if (meshes_.empty()) {
      character_status_ = "Character mesh failed to load.";
      return;
    }
    if (!loaded_ || SameSkeleton()) {
      return;
    }
    retarget_ok_ = retargeter_.Setup(skeleton_, character_skeleton_);
    char info[128];
    std::snprintf(info, sizeof(info),
                  retarget_ok_ ? "Retargeted, %d bones mapped."
                               : "Can't retarget: only %d bones mapped.",
                  retargeter_.num_mapped());
    character_status_ = info;
  }

  bool HasMatch(const Folder& _folder) const {
    for (const auto& clip : _folder.clips) {
      if (filter_.PassFilter(clip.name.c_str())) {
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

  void DrawFolder(const Folder& _folder, const fs::path& _path, bool _is_root) {
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
    ImGui::TableNextColumn();
    const bool open = ImGui::TreeNodeEx(_path.string().c_str(), folder_flags,
                                        "%s", _folder.name.c_str());
    ImGui::TableNextColumn();
    ImGui::TextDisabled("%d", _folder.num_clips);
    if (!open) {
      return;
    }
    for (const auto& folder : _folder.folders) {
      DrawFolder(folder, _path / folder.name, false);
    }

    // Clips, clipped to the visible range as folders can hold thousands.
    visible_.clear();
    for (const auto& clip : _folder.clips) {
      if (!filtering || filter_.PassFilter(clip.name.c_str())) {
        visible_.push_back(&clip);
      }
    }
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(visible_.size()));
    while (clipper.Step()) {
      for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
        const Clip& clip = *visible_[i];
        const bool selected = clip.path == selected_;
        ImGuiTreeNodeFlags flags =
            ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
            ImGuiTreeNodeFlags_SpanAllColumns |
            (selected ? ImGuiTreeNodeFlags_Selected : 0);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx(clip.path.string().c_str(), flags, "%s",
                          clip.name.c_str());
        if (ImGui::IsItemClicked() && !selected) {
          Load(clip.path);
        }
        ImGui::TableNextColumn();
        ImGui::Text("%.1fs", clip.duration);
      }
    }
    ImGui::TreePop();
  }

  // Finds the nearest skeleton.ozz, from the clip folder up to the root.
  fs::path FindSkeleton(const fs::path& _clip) const {
    for (fs::path dir = _clip.parent_path(); !dir.empty();
         dir = dir.parent_path()) {
      if (fs::exists(dir / kSkeletonName)) {
        return dir / kSkeletonName;
      }
      if (dir == root_path_ || dir == dir.parent_path()) {
        break;
      }
    }
    return {};
  }

  // Loads the optional skinned mesh that sits next to the skeleton.
  void LoadMeshes(const fs::path& _path) {
    meshes_.clear();
    if (!fs::exists(_path) ||
        !ozz::sample::LoadMeshes(_path.string().c_str(), &meshes_)) {
      meshes_.clear();
      return;
    }
    size_t num_skinning_matrices = 0;
    for (const ozz::sample::Mesh& mesh : meshes_) {
      if (mesh.highest_joint_index() >= character_skeleton_.num_joints()) {
        meshes_.clear();  // Not skinned to this skeleton.
        return;
      }
      num_skinning_matrices =
          std::max(num_skinning_matrices, mesh.joint_remaps.size());
    }
    skinning_matrices_.resize(num_skinning_matrices);
  }

  void Load(const fs::path& _clip) {
    selected_ = _clip;
    const fs::path skeleton_path = FindSkeleton(_clip);
    if (skeleton_path.empty()) {
      status_ = "No skeleton.ozz found for " + _clip.filename().string();
      loaded_ = false;
      return;
    }
    bool skeleton_changed = false;
    if (skeleton_path != skeleton_path_) {
      skeleton_path_.clear();
      if (!ozz::sample::LoadSkeleton(skeleton_path.string().c_str(), &skeleton_)) {
        status_ = "Failed to load " + skeleton_path.string();
        loaded_ = false;
        return;
      }
      skeleton_path_ = skeleton_path;
      skeleton_changed = true;
    }
    if (!ozz::sample::LoadAnimation(_clip.string().c_str(), &animation_) ||
        animation_.num_tracks() != skeleton_.num_joints()) {
      status_ = "Failed to load " + _clip.filename().string() +
                " (or it doesn't match " + skeleton_path.filename().string() + ")";
      loaded_ = false;
      return;
    }

    locals_.resize(skeleton_.num_soa_joints());
    models_.resize(skeleton_.num_joints());
    context_.Resize(skeleton_.num_joints());
    controller_.set_time_ratio(0.f);
    const bool was_loaded = loaded_;
    loaded_ = true;
    if (skeleton_changed || !was_loaded) {
      SetupRetarget();
    }

    char info[128];
    std::snprintf(info, sizeof(info), "%s: %.1fs, %d joints",
                  _clip.stem().string().c_str(), animation_.duration(),
                  skeleton_.num_joints());
    status_ = info;
  }

  // Browser state.
  fs::path root_path_;
  Folder root_;
  ImGuiTextFilter filter_;
  std::vector<const Clip*> visible_;
  fs::path selected_;
  std::string status_;

  // Playback state.
  bool loaded_ = false;
  fs::path skeleton_path_;
  ozz::sample::PlaybackController controller_;
  ozz::animation::Skeleton skeleton_;
  ozz::animation::Animation animation_;
  ozz::animation::SamplingJob::Context context_;
  ozz::vector<ozz::math::SoaTransform> locals_;
  ozz::vector<ozz::math::Float4x4> models_;

  // Character: skeleton + skinned mesh, driven directly or retargeted.
  std::vector<Character> characters_;
  int character_ = -1;
  std::string character_status_;
  fs::path character_skeleton_path_;
  ozz::animation::Skeleton character_skeleton_;
  ozz::vector<ozz::math::Float4x4> character_models_;
  Retargeter retargeter_;
  bool retarget_ok_ = false;
  ozz::vector<ozz::sample::Mesh> meshes_;
  ozz::vector<ozz::math::Float4x4> skinning_matrices_;
  bool draw_mesh_ = true;
  bool draw_skeleton_ = false;

  // Point field preview, generated live on the character.
  struct PointFieldPreview {
    bool enabled = false;
    int method = 0;   // kMethods index.
    int sampler = 1;  // kSamplers index.
    int count = 150;
    int pending_count = 150;  // Slider value, applied on release.
    int seed = 1;
    float point_size = 3.f;
    bool color_by_id = true;

    hyperpose::TriMesh rest_mesh;
    hyperpose::TriMesh posed_mesh;
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
