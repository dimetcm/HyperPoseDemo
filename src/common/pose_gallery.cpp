#include "common/pose_gallery.h"

#include <algorithm>
#include <cmath>

#include "common/character_view.h"
#include "common/pose_match.h"
#include "framework/renderer.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/skeleton.h"

namespace fs = std::filesystem;
namespace math = ozz::math;

namespace hyperpose {
namespace {

// Poses per row, and the side (z sign) rows go to. Aligned poses face +z, so
// -z is behind.
const int kColumns = 8;
const float kSide = -1.f;

}  // namespace

PoseGallery::PoseGallery() : rng_(std::random_device{}()) {}
PoseGallery::~PoseGallery() = default;

bool PoseGallery::Build(const CommonPoseResult& _result,
                        const ozz::animation::Skeleton* _skeleton,
                        int _heading_axis) {
  Clear();
  result_ = _result;
  skeleton_ = _skeleton;
  heading_axis_ = _heading_axis;

  // Most typical frame of each pose.
  ClipPlayer player;
  for (const CommonPose& pose : result_.poses) {
    const CommonPose::Frame& frame = pose.frames.front();
    const fs::path& clip = result_.clips[frame.clip];
    ozz::vector<math::Float4x4> models(skeleton_->num_joints());
    if ((clip != player.path() && !player.Load(clip, *skeleton_)) ||
        !player.Sample(frame.time, make_span(models))) {
      Clear();
      error_ = "Failed to load " + clip.filename().string();
      return false;
    }
    AlignToOrigin(make_span(models), heading_axis_);
    still_.push_back(std::move(models));
  }

  // Animated slots, starting from the still pose until first updated.
  std::uniform_real_distribution<float> offset(0.f, play_time);
  for (size_t c = 0; c < result_.poses.size(); ++c) {
    const CommonPose& pose = result_.poses[c];
    auto slot = std::make_unique<Slot>();
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
    slot->models = still_[c];
    // Staggered, so the poses don't all switch at once.
    slot->timer = offset(rng_);
    slots_.push_back(std::move(slot));
    PickFrame(static_cast<int>(c));
  }
  return true;
}

void PoseGallery::Clear() {
  result_ = CommonPoseResult();
  error_.clear();
  still_.clear();
  baked_.clear();
  baked_generation_ = -1;
  slots_.clear();
}

bool PoseGallery::PickFrame(int _c) {
  Slot& slot = *slots_[_c];
  const CommonPose& pose = result_.poses[_c];
  slot.frame = slot.picker(rng_);
  const CommonPose::Frame& frame = pose.frames[slot.frame];
  const fs::path& clip = result_.clips[frame.clip];
  if (clip != slot.clip.path() && !slot.clip.Load(clip, *skeleton_)) {
    return false;
  }
  slot.clip.time = frame.time;
  return true;
}

void PoseGallery::Update(float _dt, CharacterView* _character) {
  if (!animate) {
    if (!still_.empty() && _character->posed() && !Baked(*_character)) {
      Bake(_character);
    }
    return;
  }
  for (size_t c = 0; c < slots_.size(); ++c) {
    Slot& slot = *slots_[c];
    slot.timer -= _dt;
    slot.clip.Advance(_dt);  // Loops.
    if (slot.timer <= 0.f) {
      // Re-armed from the current play time, which the user may change.
      slot.timer = play_time;
      PickFrame(static_cast<int>(c));
    }
    // A failed load keeps the last pose.
    if (slot.clip.Sample(make_span(slot.models))) {
      AlignToOrigin(make_span(slot.models), heading_axis_);
    }
  }
}

bool PoseGallery::Baked(const CharacterView& _character) const {
  return baked_generation_ == _character.generation() &&
         baked_.size() == still_.size();
}

void PoseGallery::Bake(CharacterView* _character) {
  baked_.assign(still_.size(), {});
  for (size_t c = 0; c < still_.size(); ++c) {
    _character->Update(make_span(still_[c]));
    if (!_character->Bake(&baked_[c])) {
      baked_.clear();
      break;
    }
  }
  baked_generation_ = _character->generation();
}

bool PoseGallery::Draw(ozz::sample::Renderer* _renderer,
                       CharacterView* _character, bool _draw_mesh,
                       bool _draw_skeleton) {
  bool success = true;
  const bool show_character =
      _draw_mesh && (animate ? _character->posed() : Baked(*_character));
  for (int c = 0; c < size(); ++c) {
    const math::Float4x4 transform = Transform(c);
    const Shown pose = shown(c);
    if (show_character) {
      if (animate) {
        _character->Update(pose.models);
        success &= _character->Draw(_renderer, transform);
      } else {
        success &= CharacterView::DrawBaked(_renderer, baked_[c], transform);
      }
    }
    if (_draw_skeleton || !show_character) {
      success &= _renderer->DrawPosture(*skeleton_, pose.models, transform);
    }
  }
  return success;
}

PoseGallery::Shown PoseGallery::shown(int _c) const {
  if (animate) {
    const Slot& slot = *slots_[_c];
    return {make_span(slot.models), &slot.clip.path(), slot.clip.time,
            slot.clip.duration()};
  }
  const CommonPose::Frame& frame = result_.poses[_c].frames.front();
  return {make_span(still_[_c]), &result_.clips[frame.clip], frame.time, -1.f};
}

math::Float4x4 PoseGallery::Transform(int _c) const {
  // Rows of up to kColumns, left to right, starting one row behind the
  // origin.
  const int count = size();
  const int columns = std::max(1, std::min(count, kColumns));
  const int row = _c / columns;
  const int column = _c % columns;
  const int in_row = std::min(columns, count - row * columns);
  const float x = (column - (in_row - 1) * .5f) * spacing;
  const float z = kSide * (row + 1) * spacing * 1.5f;
  return math::Float4x4::Translation(math::simd_float4::Load(x, 0.f, z, 1.f));
}

math::Box PoseGallery::Bounds() const {
  math::Box bounds;
  for (int c = 0; c < size(); ++c) {
    const math::Box box =
        ozz::sample::ComputePostureBounds(make_span(still_[c]), Transform(c));
    bounds = bounds.is_valid() ? Merge(bounds, box) : box;
  }
  return bounds;
}

}  // namespace hyperpose
