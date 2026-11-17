#include "common/gui/pose_gallery.h"

#include <algorithm>
#include <cmath>

#include "common/gui/character_view.h"
#include "common/core/pose_match.h"
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
  stays_.clear();
  players_.clear();
  stays_built_ = false;
  stays_at_anchor_ = false;
  stays_baked_generation_ = -1;
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

float PoseGallery::MotionStart(int _clip) const {
  if (_clip < 0 || _clip >= static_cast<int>(result_.features.size())) {
    return 0.f;
  }
  const PoseFeatures& features = result_.features[_clip];
  return features.frame_rate > 0.f ? features.first_frame / features.frame_rate : 0.f;
}

bool PoseGallery::BuildStays() {
  stays_built_ = true;
  stays_.assign(result_.poses.size(), {});
  players_.clear();
  players_.resize(result_.clips.size());
  // Every transition into a pose starts a stay, until the clip's next
  // transition (each clip's transitions end with its clip end).
  const std::vector<PoseTransition>& transitions = result_.transitions;
  for (size_t i = 0; i + 1 < transitions.size(); ++i) {
    const PoseTransition& in = transitions[i];
    const PoseTransition& out = transitions[i + 1];
    if (in.to < 0 || out.clip != in.clip) {
      continue;
    }
    std::unique_ptr<ClipPlayer>& player = players_[in.clip];
    if (!player) {
      player = std::make_unique<ClipPlayer>();
      if (!player->Load(result_.clips[in.clip], *skeleton_)) {
        error_ = "Failed to load " + result_.clips[in.clip].filename().string();
        stays_.clear();
        players_.clear();
        return false;
      }
    }
    Stay stay;
    stay.clip = in.clip;
    // The anchor: the stay's frame closest to the pose (its middle if none
    // was clustered, with a stride).
    const float end = std::max(out.time, in.time);
    stay.anchor = (in.time + end) * .5f;
    float closest = -1.f;
    for (const CommonPose::Frame& frame : result_.poses[in.to].frames) {
      if (frame.clip == in.clip && frame.time >= in.time - 1e-4f && frame.time < end &&
          (closest < 0.f || frame.distance < closest)) {
        closest = frame.distance;
        stay.anchor = frame.time;
      }
    }
    stay.time = stay.anchor;
    stay.models = still_[in.to];  // Until first sampled.
    stays_[in.to].push_back(std::move(stay));
  }
  return true;
}

void PoseGallery::UpdateStays(float _dt, bool _still) {
  if (!stays_built_ && !BuildStays()) {
    return;
  }
  if (_still && stays_at_anchor_) {
    return;
  }
  // One phase for all, from -sync_window to sync_window, in a loop; still,
  // 0 (the anchors).
  float phase = 0.f;
  if (!_still) {
    const float window = std::max(sync_window, 0.f);
    phase_ += _dt * speed;
    if (phase_ > window || phase_ < -window) {
      phase_ = -window;
    }
    phase = phase_;
  }
  stays_at_anchor_ = _still;
  stays_baked_generation_ = -1;  // Posed anew.
  for (std::vector<Stay>& stays : stays_) {
    for (Stay& stay : stays) {
      ClipPlayer& player = *players_[stay.clip];
      // Held at the clip ends (the motion's start: not a leading bind pose).
      const float start = std::min(MotionStart(stay.clip), player.duration());
      stay.time = std::clamp(stay.anchor + phase, start, player.duration());
      if (player.Sample(stay.time, make_span(stay.models))) {
        AlignToOrigin(make_span(stay.models), heading_axis_);
      }
    }
  }
}

void PoseGallery::Update(float _dt, CharacterView* _character) {
  if (mode == Mode::kStill) {
    if (!still_.empty() && _character->posed() && !Baked(*_character)) {
      Bake(_character);
    }
    return;
  }
  if (mode == Mode::kClips) {
    UpdateStays(_dt, false);
    return;
  }
  if (mode == Mode::kClipsStill) {
    UpdateStays(_dt, true);
    if (_character->posed() && stays_baked_generation_ != _character->generation()) {
      for (std::vector<Stay>& stays : stays_) {
        for (Stay& stay : stays) {
          _character->Bake(make_span(stay.models), &stay.baked);
        }
      }
      stays_baked_generation_ = _character->generation();
    }
    return;
  }
  for (size_t c = 0; c < slots_.size(); ++c) {
    Slot& slot = *slots_[c];
    slot.timer -= _dt;
    slot.clip.Advance(_dt * speed);  // Loops.
    // Looped past the motion's start (onto a leading bind pose): back to it.
    const int clip = result_.poses[c].frames[slot.frame].clip;
    if (slot.clip.time < MotionStart(clip)) {
      slot.clip.time = std::min(MotionStart(clip), slot.clip.duration());
    }
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
    if (!_character->Bake(make_span(still_[c]), &baked_[c])) {
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
  if (mode == Mode::kClips || mode == Mode::kClipsStill) {
    // Every stay overlaid; still, skinned once.
    const bool still = mode == Mode::kClipsStill;
    const bool show_character =
        _draw_mesh && _character->posed() &&
        (!still || stays_baked_generation_ == _character->generation());
    for (int c = 0; c < size() && c < static_cast<int>(stays_.size()); ++c) {
      const math::Float4x4 transform = Transform(c);
      for (const Stay& stay : stays_[c]) {
        if (show_character) {
          success &= still ? _character->DrawBaked(_renderer, stay.baked, transform)
                           : _character->Draw(_renderer, make_span(stay.models), transform);
        }
        if (_draw_skeleton || !show_character) {
          success &= _renderer->DrawPosture(*skeleton_, make_span(stay.models), transform);
        }
      }
    }
    return success;
  }
  const bool show_character =
      _draw_mesh && (animated() ? _character->posed() : Baked(*_character));
  for (int c = 0; c < size(); ++c) {
    const math::Float4x4 transform = Transform(c);
    const Shown pose = shown(c);
    if (show_character) {
      if (animated()) {
        success &= _character->Draw(_renderer, pose.models, transform);
      } else {
        success &= _character->DrawBaked(_renderer, baked_[c], transform);
      }
    }
    if (_draw_skeleton || !show_character) {
      success &= _renderer->DrawPosture(*skeleton_, pose.models, transform);
    }
  }
  return success;
}

PoseGallery::Shown PoseGallery::shown(int _c) const {
  if ((mode == Mode::kClips || mode == Mode::kClipsStill) &&
      _c < static_cast<int>(stays_.size()) && !stays_[_c].empty()) {
    const Stay& stay = stays_[_c].front();
    Shown shown = {make_span(stay.models), &result_.clips[stay.clip], stay.time,
                   mode == Mode::kClips ? players_[stay.clip]->duration() : -1.f};
    shown.count = static_cast<int>(stays_[_c].size());
    return shown;
  }
  if (mode == Mode::kFrames) {
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
