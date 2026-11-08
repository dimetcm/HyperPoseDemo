// Minimal Pose Difference Blend: pick two clips (A and B). The app compares
// every frame of A with every frame of B, finds the most similar poses, and
// plays A, crossfading into B at that point. B is aligned to A's root position
// and heading at the transition, so motion continues smoothly.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <imgui.h>

#include "common/character_view.h"
#include "common/clip_library.h"
#include "common/clip_player.h"
#include "common/pose_match.h"
#include "framework/application.h"
#include "framework/imgui.h"
#include "framework/renderer.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/blending_job.h"
#include "ozz/animation/runtime/local_to_model_job.h"
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
OZZ_OPTIONS_DECLARE_STRING(a, "Clip A (.ozz) to load at startup.", "", false)
OZZ_OPTIONS_DECLARE_STRING(b, "Clip B (.ozz) to load at startup.", "", false)

namespace {

// Pose matching rate. Lower is faster, the matrix grows with the square.
constexpr float kMatchRate = 15.f;
// Weight of joint velocity differences (seconds), relative to positions.
constexpr float kVelocityWeight = .1f;
// Seconds played around the transition when looping around it.
constexpr float kLoopMargin = 2.f;

double MillisecondsSince(std::chrono::steady_clock::time_point _start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - _start)
      .count();
}

float SmoothStep(float _x) {
  _x = std::clamp(_x, 0.f, 1.f);
  return _x * _x * (3.f - 2.f * _x);
}

// Low difference to high: dark blue, teal, yellow.
ImU32 HeatColor(float _t) {
  _t = std::clamp(_t, 0.f, 1.f);
  const float stops[3][3] = {{.05f, .05f, .25f}, {.1f, .6f, .6f}, {1.f, .9f, .2f}};
  const float s = _t * 2.f;
  const int i = std::min(static_cast<int>(s), 1);
  const float u = s - i;
  return ImGui::ColorConvertFloat4ToU32(
      ImVec4(stops[i][0] + (stops[i + 1][0] - stops[i][0]) * u,
             stops[i][1] + (stops[i + 1][1] - stops[i][1]) * u,
             stops[i][2] + (stops[i + 1][2] - stops[i][2]) * u, 1.f));
}

}  // namespace

class PoseBlendApplication : public ozz::sample::Application {
 protected:
  bool OnInitialize() override {
    if (!library_.Scan(OPTIONS_root.value(), OPTIONS_metadata.value())) {
      status_ = "No .ozz clips found in " + std::string(OPTIONS_root.value());
    }
    character_.Scan(OPTIONS_root.value());
    if (OPTIONS_a.value()[0]) {
      Assign(0, OPTIONS_a.value());
    }
    if (OPTIONS_b.value()[0]) {
      Assign(1, OPTIONS_b.value());
    }
    return true;
  }

  bool OnUpdate(float _dt, float) override {
    if (!transition_.valid()) {
      return true;
    }
    if (playing_) {
      time_ += _dt * speed_;
      const float begin = loop_transition_ ? LoopBegin() : 0.f;
      const float end = loop_transition_ ? LoopEnd() : TotalDuration();
      if (time_ > end || time_ < begin) {
        time_ = begin;
      }
    }
    if (!Evaluate(time_)) {
      return false;
    }
    character_.Update(make_span(models_));
    return true;
  }

  bool OnDisplay(ozz::sample::Renderer* _renderer) override {
    if (!transition_.valid()) {
      return true;
    }
    const auto transform = math::Float4x4::identity();
    bool success = true;
    const bool show_character = draw_mesh_ && character_.posed();
    if (show_character) {
      success &= character_.Draw(_renderer, transform);
    }
    if (draw_skeleton_ || !show_character) {
      success &= _renderer->DrawPosture(skeleton(), make_span(models_), transform);
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
    character_.DrawGui();
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Transition", ImGuiTreeNodeFlags_DefaultOpen)) {
      TransitionGui();
      ImGui::Separator();
    }

    fs::path clicked;
    if (library_.DrawTable({slots_[0].clip.path(), slots_[1].clip.path()}, &clicked)) {
      Assign(pick_, clicked);
    }
    ImGui::End();
    return true;
  }

  ozz::math::Box GetSceneBounds() const override {
    if (!transition_.valid()) {
      return ozz::math::Box();
    }
    const auto& models =
        character_.posed() ? character_.joint_model_transforms() : models_;
    return ozz::sample::ComputePostureBounds(make_span(models),
                                             math::Float4x4::identity());
  }

 private:
  struct Slot {
    hyperpose::ClipPlayer clip;
    hyperpose::PoseFeatures features;
  };

  const ozz::animation::Skeleton& skeleton() const {
    return clip_skeleton_.skeleton();
  }

  // Transition timing, in seconds.
  float TransitionTimeA() const { return transition_.a_frame / kMatchRate; }
  float TransitionTimeB() const { return transition_.b_frame / kMatchRate; }
  float BlendDuration() const { return blend_frames_ / kMatchRate; }
  float TotalDuration() const {
    return TransitionTimeA() + slots_[1].clip.duration() - TransitionTimeB();
  }
  float LoopBegin() const { return std::max(0.f, TransitionTimeA() - kLoopMargin); }
  float LoopEnd() const {
    return std::min(TotalDuration(),
                    TransitionTimeA() + BlendDuration() + kLoopMargin);
  }

  // Loads a clip into slot A (0) or B (1). Both must share a skeleton.
  void Assign(int _slot, const fs::path& _clip) {
    Slot& slot = slots_[_slot];
    const Slot& other = slots_[1 - _slot];
    const auto use =
        clip_skeleton_.Use(_clip, library_.root(), !other.clip.loaded());
    if (use == hyperpose::ClipSkeleton::Result::kFailed) {
      status_ = clip_skeleton_.error();
      return;
    }
    if (use == hyperpose::ClipSkeleton::Result::kChanged) {
      blended_.resize(skeleton().num_soa_joints());
      models_.resize(skeleton().num_joints());
      character_.SetClipSkeleton(&skeleton(), clip_skeleton_.path());
    }
    if (!slot.clip.Load(_clip, skeleton()) ||
        !hyperpose::ExtractPoseFeatures(slot.clip.animation(), skeleton(),
                                        kMatchRate, &slot.features)) {
      slot.clip.Unload();
      status_ = "Failed to load " + _clip.filename().string();
      ComputeDifference();
      return;
    }
    status_.clear();
    pick_ = 1 - _slot;  // Next click fills the other slot.
    ComputeDifference();
  }

  // Pose difference matrix between A and B, then best transition.
  void ComputeDifference() {
    transition_ = hyperpose::Transition();
    difference_ = hyperpose::PoseDifference();
    map_colors_.clear();
    if (!slots_[0].clip.loaded() || !slots_[1].clip.loaded()) {
      return;
    }
    const auto start = std::chrono::steady_clock::now();
    hyperpose::ComputePoseDifference(slots_[0].features, slots_[1].features,
                                     kVelocityWeight, &difference_);
    difference_ms_ = MillisecondsSince(start);

    // Heat map colors, normalized between the lowest difference and the 90th
    // percentile, so the low (interesting) range keeps contrast.
    std::vector<float> sorted(difference_.values.begin(), difference_.values.end());
    if (sorted.empty()) {
      return;
    }
    const size_t p90 = sorted.size() * 9 / 10;
    std::nth_element(sorted.begin(), sorted.begin() + p90, sorted.end());
    map_max_ = sorted[p90];
    map_min_ = *std::min_element(difference_.values.begin(), difference_.values.end());
    BuildMapColors();
    FindTransition();
  }

  int MinGap() const {
    // Same clip: skip trivial matches with nearby frames.
    return slots_[0].clip.path() == slots_[1].clip.path()
               ? std::max(blend_frames_ * 2, static_cast<int>(kMatchRate))
               : 0;
  }

  void FindTransition() {
    best_ = hyperpose::FindBestTransition(difference_, blend_frames_, MinGap());
    SetTransition(best_);
    if (best_.valid()) {
      ozz::log::Out() << difference_.rows << " x " << difference_.cols
                      << " poses compared in " << difference_ms_
                      << " ms. Best transition: A " << TransitionTimeA()
                      << "s -> B " << TransitionTimeB() << "s, difference "
                      << best_.difference * 100.f << " cm." << std::endl;
    }
  }

  void SetTransition(const hyperpose::Transition& _transition) {
    transition_ = _transition;
    if (!transition_.valid()) {
      status_ = "Clips are too short for this blend duration.";
      return;
    }
    // Aligns B root onto A root at the transition (horizontal position and
    // heading).
    const auto& fa = slots_[0].features;
    const auto& fb = slots_[1].features;
    const math::Float3 a_root = fa.root_positions[transition_.a_frame];
    const math::Float3 b_root = fb.root_positions[transition_.b_frame];
    align_angle_ = fa.root_headings[transition_.a_frame] -
                   fb.root_headings[transition_.b_frame];
    align_from_ = math::Float3(b_root.x, 0.f, b_root.z);
    align_to_ = math::Float3(a_root.x, 0.f, a_root.z);
    time_ = LoopBegin();
    playing_ = true;
  }

  // Applies the B to A alignment to the root joint (joint 0, which has no
  // parent so its local transform is model-space).
  void AlignRoot(ozz::span<math::SoaTransform> _locals) const {
    math::SoaTransform& soa = _locals[0];
    math::Float3 t(math::GetX(soa.translation.x), math::GetX(soa.translation.y),
                   math::GetX(soa.translation.z));
    const math::Quaternion q(math::GetX(soa.rotation.x), math::GetX(soa.rotation.y),
                             math::GetX(soa.rotation.z), math::GetX(soa.rotation.w));
    t = hyperpose::RotateY(t - align_from_, align_angle_) + align_to_;
    const math::Quaternion r =
        math::Quaternion::FromAxisAngle(math::Float3::y_axis(), align_angle_) * q;
    auto set = [](math::SimdFloat4& _v, float _f) {
      _v = math::SetX(_v, math::simd_float4::Load1(_f));
    };
    set(soa.translation.x, t.x);
    set(soa.translation.y, t.y);
    set(soa.translation.z, t.z);
    set(soa.rotation.x, r.x);
    set(soa.rotation.y, r.y);
    set(soa.rotation.z, r.z);
    set(soa.rotation.w, r.w);
  }

  // Pose at sequence time _time: A, then crossfade, then B (aligned).
  bool Evaluate(float _time) {
    const float a_time = TransitionTimeA();
    const float b_time = TransitionTimeB() + (_time - a_time);
    const float blend = (_time - a_time) / BlendDuration();
    weight_ = SmoothStep(blend);

    hyperpose::ClipPlayer& a = slots_[0].clip;
    hyperpose::ClipPlayer& b = slots_[1].clip;
    ozz::span<const math::SoaTransform> locals;
    if (blend <= 0.f) {
      if (!a.SampleLocals(_time)) return false;
      locals = a.locals();
    } else if (blend >= 1.f) {
      if (!b.SampleLocals(b_time)) return false;
      AlignRoot(b.locals());
      locals = b.locals();
    } else {
      if (!a.SampleLocals(_time) || !b.SampleLocals(b_time)) return false;
      AlignRoot(b.locals());
      ozz::animation::BlendingJob::Layer layers[2];
      layers[0].weight = 1.f - weight_;
      layers[0].transform = a.locals();
      layers[1].weight = weight_;
      layers[1].transform = b.locals();
      ozz::animation::BlendingJob job;
      job.layers = layers;
      job.rest_pose = skeleton().joint_rest_poses();
      job.output = make_span(blended_);
      if (!job.Run()) return false;
      locals = make_span(blended_);
    }

    ozz::animation::LocalToModelJob ltm;
    ltm.skeleton = &skeleton();
    ltm.input = locals;
    ltm.output = make_span(models_);
    return ltm.Run();
  }

  void SlotGui(int _slot, const char* _label) {
    const Slot& slot = slots_[_slot];
    if (!slot.clip.loaded()) {
      ImGui::TextDisabled("%s: click a clip below", _label);
      return;
    }
    const std::string name = slot.clip.path().stem().string();
    ImGui::Text("%s: %s (%.1fs)", _label, name.c_str(), slot.clip.duration());
    const std::string& description = library_.Description(name);
    if (!description.empty()) {
      ImGui::SameLine();
      ImGui::TextDisabled("%s", description.c_str());
    }
  }

  void TransitionGui() {
    ImGui::TextUnformatted("Next click sets:");
    ImGui::SameLine();
    ImGui::RadioButton("A", &pick_, 0);
    ImGui::SameLine();
    ImGui::RadioButton("B", &pick_, 1);
    SlotGui(0, "A");
    SlotGui(1, "B");

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * .5f);
    ImGui::SliderFloat("Blend duration", &pending_blend_, .1f, 1.5f, "%.2f s");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
      blend_frames_ = std::max(1, static_cast<int>(pending_blend_ * kMatchRate + .5f));
      if (!difference_.values.empty()) {
        FindTransition();
      }
    }

    if (!difference_.values.empty()) {
      ImGui::Text("%d x %d poses compared in %.0f ms", difference_.rows,
                  difference_.cols, difference_ms_);
    }
    if (!transition_.valid()) {
      return;
    }
    ImGui::Text("Transition: A %.2fs -> B %.2fs, difference %.1f cm%s",
                TransitionTimeA(), TransitionTimeB(),
                transition_.difference * 100.f,
                transition_.a_frame == best_.a_frame &&
                        transition_.b_frame == best_.b_frame
                    ? " (best)"
                    : "");
    if (transition_.a_frame != best_.a_frame ||
        transition_.b_frame != best_.b_frame) {
      ImGui::SameLine();
      if (ImGui::SmallButton("Use best")) {
        SetTransition(best_);
      }
    }

    DifferenceMapGui();

    // Playback.
    if (ImGui::Button(playing_ ? "Pause" : "Play")) {
      playing_ = !playing_;
    }
    ImGui::SameLine();
    if (ImGui::Button("Replay transition")) {
      time_ = LoopBegin();
      playing_ = true;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Loop around transition", &loop_transition_);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::SliderFloat("##time", &time_, 0.f, TotalDuration(), "%.2f s");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * .5f);
    ImGui::SliderFloat("Speed", &speed_, .1f, 2.f, "%.2fx");
    const float a_time = TransitionTimeA();
    if (time_ < a_time) {
      ImGui::Text("Playing A");
    } else if (time_ < a_time + BlendDuration()) {
      ImGui::Text("Blending A -> B: %.0f%%", weight_ * 100.f);
    } else {
      ImGui::Text("Playing B");
    }
  }

  // Heat map cells: minimum difference over each block of frames.
  static constexpr int kMaxMapCells = 160;

  void BuildMapColors() {
    map_cols_ = std::min(difference_.rows, kMaxMapCells);  // A, horizontal.
    map_rows_ = std::min(difference_.cols, kMaxMapCells);  // B, vertical.
    map_colors_.assign(map_cols_ * map_rows_, 0);
    const float range = std::max(map_max_ - map_min_, 1e-6f);
    for (int x = 0; x < map_cols_; ++x) {
      const int a0 = x * difference_.rows / map_cols_;
      const int a1 = std::max(a0 + 1, (x + 1) * difference_.rows / map_cols_);
      for (int y = 0; y < map_rows_; ++y) {
        const int b0 = y * difference_.cols / map_rows_;
        const int b1 = std::max(b0 + 1, (y + 1) * difference_.cols / map_rows_);
        float value = difference_.at(a0, b0);
        for (int a = a0; a < a1; ++a) {
          for (int b = b0; b < b1; ++b) {
            value = std::min(value, difference_.at(a, b));
          }
        }
        map_colors_[y * map_cols_ + x] = HeatColor((value - map_min_) / range);
      }
    }
  }

  void DifferenceMapGui() {
    if (map_colors_.empty()) {
      return;
    }
    ImGui::TextDisabled("Pose difference: A time ->, B time v. Click to pick.");
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = std::min(width * map_rows_ / map_cols_,
                                  160.f * ImGui::GetStyle().FontScaleDpi);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("difference_map", ImVec2(width, height));
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const float cw = width / map_cols_, ch = height / map_rows_;
    for (int y = 0; y < map_rows_; ++y) {
      for (int x = 0; x < map_cols_; ++x) {
        const ImVec2 p0(origin.x + x * cw, origin.y + y * ch);
        draw->AddRectFilled(p0, ImVec2(p0.x + cw + 1.f, p0.y + ch + 1.f),
                            map_colors_[y * map_cols_ + x]);
      }
    }
    auto to_screen = [&](int _a, int _b) {
      return ImVec2(origin.x + (_a + .5f) * width / difference_.rows,
                    origin.y + (_b + .5f) * height / difference_.cols);
    };
    const float r = 4.f * ImGui::GetStyle().FontScaleDpi;
    draw->AddCircle(to_screen(best_.a_frame, best_.b_frame), r * 1.5f,
                    IM_COL32(255, 255, 255, 255), 0, 2.f);
    const ImVec2 t = to_screen(transition_.a_frame, transition_.b_frame);
    draw->AddLine(ImVec2(t.x - r, t.y), ImVec2(t.x + r, t.y), IM_COL32(255, 60, 60, 255), 2.f);
    draw->AddLine(ImVec2(t.x, t.y - r), ImVec2(t.x, t.y + r), IM_COL32(255, 60, 60, 255), 2.f);

    if (ImGui::IsItemHovered()) {
      const ImVec2 mouse = ImGui::GetIO().MousePos;
      const int a = std::clamp(static_cast<int>((mouse.x - origin.x) / width * difference_.rows),
                               0, difference_.rows - 1);
      const int b = std::clamp(static_cast<int>((mouse.y - origin.y) / height * difference_.cols),
                               0, difference_.cols - 1);
      const bool valid = hyperpose::IsValidTransition(difference_, a, b,
                                                      blend_frames_, MinGap());
      ImGui::SetTooltip("A %.2fs -> B %.2fs\ndifference %.1f cm%s", a / kMatchRate,
                        b / kMatchRate, difference_.at(a, b) * 100.f,
                        valid ? "" : "\n(not enough frames left to blend)");
      if (ImGui::IsItemClicked() && valid) {
        SetTransition({a, b, difference_.at(a, b)});
      }
    }
  }

  hyperpose::ClipLibrary library_;
  hyperpose::CharacterView character_;
  std::string status_;
  bool draw_mesh_ = true;
  bool draw_skeleton_ = false;

  // Shared clip skeleton and output pose.
  hyperpose::ClipSkeleton clip_skeleton_;
  ozz::vector<math::SoaTransform> blended_;
  ozz::vector<math::Float4x4> models_;

  Slot slots_[2];
  int pick_ = 0;  // Slot set by the next clip click.

  // Matching.
  hyperpose::PoseDifference difference_;
  double difference_ms_ = 0.;
  hyperpose::Transition best_;
  hyperpose::Transition transition_;
  int blend_frames_ = static_cast<int>(.4f * kMatchRate);
  float pending_blend_ = .4f;
  float align_angle_ = 0.f;
  math::Float3 align_from_ = math::Float3::zero();
  math::Float3 align_to_ = math::Float3::zero();

  // Heat map.
  std::vector<ImU32> map_colors_;
  int map_cols_ = 0, map_rows_ = 0;
  float map_min_ = 0.f, map_max_ = 1.f;

  // Playback.
  float time_ = 0.f;
  float speed_ = 1.f;
  float weight_ = 0.f;
  bool playing_ = true;
  bool loop_transition_ = true;
};

int main(int _argc, const char** _argv) {
  return PoseBlendApplication().Run(_argc, _argv, "0.1",
                                    "Minimal Pose Difference Blend");
}
