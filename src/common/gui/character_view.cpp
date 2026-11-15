#include "common/gui/character_view.h"

#include <algorithm>
#include <cstdio>

#include <imgui.h>

#include "common/core/anim_utils.h"
#include "common/gui/clip_library.h"
#include "common/core/clip_player.h"
#include "framework/renderer.h"
#include "framework/utils.h"
#include "ozz/geometry/runtime/skinning_job.h"

namespace fs = std::filesystem;

namespace hyperpose {

void CharacterView::Scan(const fs::path& _root) {
  characters_.clear();
  ScanFolder(_root);
  Select(characters_.empty() ? -1 : 0);
}

void CharacterView::ScanFolder(const fs::path& _path) {
  if (fs::exists(_path / kSkeletonFileName) && fs::exists(_path / kMeshFileName)) {
    characters_.push_back({_path.filename().string(), _path / kSkeletonFileName,
                           _path / kMeshFileName});
  }
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(_path, ec)) {
    if (entry.is_directory() && entry.path().filename().string()[0] != '.') {
      ScanFolder(entry.path());
    }
  }
}

bool CharacterView::Select(int _index) {
  index_ = _index;
  ++generation_;
  skeleton_path_.clear();
  meshes_.clear();
  status_.clear();
  if (_index < 0) {
    return true;
  }
  const Character& character = characters_[_index];
  if (!ozz::sample::LoadSkeleton(character.skeleton.string().c_str(), &skeleton_)) {
    status_ = "Failed to load " + character.skeleton.string();
    return false;
  }
  skeleton_path_ = character.skeleton;

  // Meshes must be skinned to this skeleton.
  if (!ozz::sample::LoadMeshes(character.mesh.string().c_str(), &meshes_)) {
    meshes_.clear();
  }
  size_t num_skinning_matrices = 0;
  for (const ozz::sample::Mesh& mesh : meshes_) {
    if (mesh.highest_joint_index() >= skeleton_.num_joints()) {
      meshes_.clear();
      break;
    }
    num_skinning_matrices = std::max(num_skinning_matrices, mesh.joint_remaps.size());
  }
  skinning_matrices_.resize(num_skinning_matrices);
  SetupRetarget();
  return !meshes_.empty();
}

void CharacterView::SetClipSkeleton(const ozz::animation::Skeleton* _skeleton,
                                    const fs::path& _path) {
  clip_skeleton_ = _skeleton;
  clip_skeleton_path_ = _skeleton ? _path : fs::path();
  SetupRetarget();
}

void CharacterView::SetupRetarget() {
  retarget_ok_ = false;
  if (index_ < 0) {
    return;
  }
  if (meshes_.empty()) {
    status_ = "Character mesh failed to load.";
    return;
  }
  status_.clear();
  if (!clip_skeleton_ || SameSkeleton()) {
    return;
  }
  retarget_ok_ = retargeter_.Setup(*clip_skeleton_, skeleton_);
  char info[128];
  std::snprintf(info, sizeof(info),
                retarget_ok_ ? "Retargeted, %d bones mapped."
                             : "Can't retarget: only %d bones mapped.",
                retargeter_.num_mapped());
  status_ = info;
}

bool CharacterView::posed() const {
  return index_ >= 0 && !meshes_.empty() && clip_skeleton_ &&
         (SameSkeleton() || retarget_ok_);
}

bool CharacterView::Pose(ozz::span<const ozz::math::Float4x4> _clip_models,
                         ozz::vector<ozz::math::Float4x4>* _character_models) const {
  if (!posed() ||
      static_cast<int>(_clip_models.size()) != clip_skeleton_->num_joints()) {
    return false;
  }
  _character_models->resize(skeleton_.num_joints());
  if (SameSkeleton()) {
    std::copy(_clip_models.begin(), _clip_models.end(), _character_models->begin());
  } else {
    retargeter_.Run(_clip_models, make_span(*_character_models));
  }
  return true;
}

ozz::math::Box CharacterView::Bounds(
    ozz::span<const ozz::math::Float4x4> _clip_models,
    const ozz::math::Float4x4& _transform) const {
  ozz::vector<ozz::math::Float4x4> models;
  if (Pose(_clip_models, &models)) {
    return ozz::sample::ComputePostureBounds(make_span(models), _transform);
  }
  return ozz::sample::ComputePostureBounds(_clip_models, _transform);
}

bool CharacterView::Draw(ozz::sample::Renderer* _renderer,
                         ozz::span<const ozz::math::Float4x4> _clip_models,
                         const ozz::math::Float4x4& _transform) {
  if (!Pose(_clip_models, &models_)) {
    return true;  // Nothing to draw.
  }
  UpdateGpuMeshes(_renderer);
  bool success = true;
  for (size_t m = 0; m < meshes_.size(); ++m) {
    ComputeSkinningMatrices(meshes_[m], make_span(models_), make_span(skinning_matrices_));
    success &= DrawMesh(_renderer, m, make_span(skinning_matrices_), _transform);
  }
  return success;
}

bool CharacterView::Bake(ozz::span<const ozz::math::Float4x4> _clip_models, Baked* _baked) {
  _baked->clear();
  if (!Pose(_clip_models, &models_)) {
    return false;
  }
  for (const ozz::sample::Mesh& mesh : meshes_) {
    ComputeSkinningMatrices(mesh, make_span(models_), make_span(skinning_matrices_));
    _baked->emplace_back(skinning_matrices_.begin(),
                         skinning_matrices_.begin() + mesh.joint_remaps.size());
  }
  return true;
}

bool CharacterView::DrawBaked(ozz::sample::Renderer* _renderer, const Baked& _baked,
                              const ozz::math::Float4x4& _transform) {
  if (_baked.size() != meshes_.size()) {
    return false;
  }
  UpdateGpuMeshes(_renderer);
  bool success = true;
  for (size_t m = 0; m < meshes_.size(); ++m) {
    // The framework's CPU path takes mutable matrices.
    ozz::vector<ozz::math::Float4x4>& matrices =
        const_cast<ozz::vector<ozz::math::Float4x4>&>(_baked[m]);
    success &= DrawMesh(_renderer, m, make_span(matrices), _transform);
  }
  return success;
}

void CharacterView::UpdateGpuMeshes(ozz::sample::Renderer* _renderer) {
  if (gpu_generation_ == generation_ && gpu_meshes_.size() == meshes_.size()) {
    return;
  }
  gpu_meshes_.clear();
  for (const ozz::sample::Mesh& mesh : meshes_) {
    gpu_meshes_.push_back(_renderer->CreateGpuMesh(mesh));  // nullptr: CPU.
  }
  gpu_generation_ = generation_;
}

bool CharacterView::DrawMesh(ozz::sample::Renderer* _renderer, size_t _mesh,
                             ozz::span<ozz::math::Float4x4> _skinning_matrices,
                             const ozz::math::Float4x4& _transform) {
  if (_mesh < gpu_meshes_.size() && gpu_meshes_[_mesh]) {
    return _renderer->DrawGpuMesh(
        *gpu_meshes_[_mesh],
        ozz::span<const ozz::math::Float4x4>(_skinning_matrices.data(),
                                             _skinning_matrices.size()),
        _transform);
  }
  return _renderer->DrawSkinnedMesh(meshes_[_mesh], _skinning_matrices, _transform);
}

bool CharacterView::DrawGui() {
  bool changed = false;
  const char* none = "None (skeleton only)";
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::BeginCombo("##character",
                        index_ >= 0 ? characters_[index_].name.c_str() : none)) {
    if (ImGui::Selectable(none, index_ < 0) && index_ >= 0) {
      Select(-1);
      changed = true;
    }
    for (int i = 0; i < static_cast<int>(characters_.size()); ++i) {
      if (ImGui::Selectable(characters_[i].name.c_str(), index_ == i) &&
          index_ != i) {
        Select(i);
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
  if (index_ >= 0 && !status_.empty()) {
    ImGui::TextWrapped("%s", status_.c_str());
  }
  return changed;
}

}  // namespace hyperpose
