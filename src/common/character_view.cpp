#include "common/character_view.h"

#include <algorithm>
#include <cstdio>

#include <imgui.h>

#include "common/anim_utils.h"
#include "common/clip_library.h"
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
  models_.resize(skeleton_.num_joints());

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

void CharacterView::Update(ozz::span<const ozz::math::Float4x4> _clip_joint_model_transforms) {
  if (!posed()) {
    return;
  }
  if (SameSkeleton()) {
    std::copy(_clip_joint_model_transforms.begin(),
              _clip_joint_model_transforms.end(), models_.begin());
  } else {
    retargeter_.Run(_clip_joint_model_transforms, make_span(models_));
  }
}

bool CharacterView::Draw(ozz::sample::Renderer* _renderer,
                         const ozz::math::Float4x4& _transform) {
  if (!posed()) {
    return true;
  }
  bool success = true;
  for (const ozz::sample::Mesh& mesh : meshes_) {
    ComputeSkinningMatrices(mesh, make_span(models_), make_span(skinning_matrices_));
    success &= _renderer->DrawSkinnedMesh(mesh, make_span(skinning_matrices_),
                                          _transform);
  }
  return success;
}

bool CharacterView::Bake(ozz::vector<ozz::sample::Mesh>* _baked) {
  _baked->clear();
  if (!posed()) {
    return false;
  }
  for (const ozz::sample::Mesh& mesh : meshes_) {
    ComputeSkinningMatrices(mesh, make_span(models_), make_span(skinning_matrices_));
    ozz::sample::Mesh baked;
    baked.triangle_indices = mesh.triangle_indices;
    for (const ozz::sample::Mesh::Part& part : mesh.parts) {
      ozz::sample::Mesh::Part out;
      out.uvs = part.uvs;
      out.colors = part.colors;
      out.positions.resize(part.positions.size());
      out.normals.resize(part.normals.size());  // Tangents: unused by DrawMesh.
      const int vertex_count = part.vertex_count();
      if (vertex_count > 0) {
        const int influences = part.influences_count();
        ozz::geometry::SkinningJob job;
        job.vertex_count = vertex_count;
        job.influences_count = influences;
        job.joint_matrices = make_span(skinning_matrices_);
        job.joint_indices = make_span(part.joint_indices);
        job.joint_indices_stride = sizeof(uint16_t) * influences;
        if (influences > 1) {
          job.joint_weights = make_span(part.joint_weights);
          job.joint_weights_stride = sizeof(float) * (influences - 1);
        }
        job.in_positions = make_span(part.positions);
        job.in_positions_stride = sizeof(float) * 3;
        job.out_positions = make_span(out.positions);
        job.out_positions_stride = sizeof(float) * 3;
        if (part.normals.size() == part.positions.size()) {
          job.in_normals = make_span(part.normals);
          job.in_normals_stride = sizeof(float) * 3;
          job.out_normals = make_span(out.normals);
          job.out_normals_stride = sizeof(float) * 3;
        }
        if (!job.Run()) {
          _baked->clear();
          return false;
        }
      }
      baked.parts.push_back(std::move(out));
    }
    _baked->push_back(std::move(baked));
  }
  return true;
}

bool CharacterView::DrawBaked(ozz::sample::Renderer* _renderer,
                              const ozz::vector<ozz::sample::Mesh>& _baked,
                              const ozz::math::Float4x4& _transform) {
  bool success = true;
  for (const ozz::sample::Mesh& mesh : _baked) {
    success &= _renderer->DrawMesh(mesh, _transform);
  }
  return success;
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
