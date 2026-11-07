// AnimPreview: browses ozz animation archives in a folder tree and plays the
// selected clip. Each clip uses the nearest skeleton.ozz in its folder or above.

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <imgui.h>

#include "framework/application.h"
#include "framework/imgui.h"
#include "framework/mesh.h"
#include "framework/renderer.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/io/archive.h"
#include "ozz/base/io/stream.h"
#include "ozz/base/maths/box.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/options/options.h"
#include "retarget.h"

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

// Reads an animation archive header up to its duration, without loading the
// keyframes. Returns false if the file isn't an ozz animation.
bool ReadClipDuration(const fs::path& _path, float* _duration) {
  ozz::io::File file(_path.string().c_str(), "rb");
  if (!file.opened()) {
    return false;
  }
  ozz::io::IArchive archive(&file);
  if (!archive.TestTag<ozz::animation::Animation>()) {
    return false;
  }
  // Layout: tag, version, then duration first (see Animation::Load).
  file.Seek(ozz::io::internal::Tag<const ozz::animation::Animation>::kTagLength,
            ozz::io::Stream::kCurrent);
  uint32_t version = 0;
  archive >> version;
  archive >> *_duration;
  return version == ozz::io::internal::Version<const ozz::animation::Animation>::kValue;
}

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
      if (ReadClipDuration(entry.path(), &duration)) {
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

    ozz::animation::SamplingJob sampling_job;
    sampling_job.animation = &animation_;
    sampling_job.context = &context_;
    sampling_job.ratio = controller_.time_ratio();
    sampling_job.output = make_span(locals_);
    if (!sampling_job.Run()) {
      return false;
    }

    ozz::animation::LocalToModelJob ltm_job;
    ltm_job.skeleton = &skeleton_;
    ltm_job.input = make_span(locals_);
    ltm_job.output = make_span(models_);
    if (!ltm_job.Run()) {
      return false;
    }

    if (ShowCharacter() && !SameSkeleton()) {
      retargeter_.Run(make_span(models_), make_span(character_models_));
    }
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
        for (size_t i = 0; i < mesh.joint_remaps.size(); ++i) {
          skinning_matrices_[i] =
              models[mesh.joint_remaps[i]] * mesh.inverse_bind_poses[i];
        }
        success &= _renderer->DrawSkinnedMesh(
            mesh, make_span(skinning_matrices_), transform);
      }
    }
    if (draw_skeleton_ || !show_character) {
      success &= _renderer->DrawPosture(skeleton_, make_span(models_), transform);
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
    const auto& models = ShowCharacter() ? CharacterModels() : models_;
    return ozz::sample::ComputePostureBounds(make_span(models),
                                             ozz::math::Float4x4::identity());
  }

 private:
  bool SameSkeleton() const { return skeleton_path_ == character_skeleton_path_; }

  // The character is drawn when it shares the clip skeleton, or when the clip
  // could be retargeted onto it.
  bool ShowCharacter() const {
    return loaded_ && draw_mesh_ && character_ >= 0 && !meshes_.empty() &&
           (SameSkeleton() || retarget_ok_);
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
};

int main(int _argc, const char** _argv) {
  return PreviewApplication().Run(_argc, _argv, "0.1", "HyperPose Preview");
}
