#include "common/gui/clip_preview.h"

#include <algorithm>
#include <cfloat>

#include <imgui.h>

#include "common/core/pose_match.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/base/containers/vector.h"

namespace math = ozz::math;

namespace hyperpose {

void ClipPreview::Anchor(float _time, const math::Float4x4* _character_root,
                         int _heading_axis) {
  ozz::vector<math::Float4x4> models(clip.animation().num_tracks());
  if (!clip.loaded() || !clip.Sample(_time, make_span(models))) {
    return;
  }
  placement_ = _character_root
                   ? RootPlacement::Between(models[0], *_character_root, _heading_axis)
                   : RootPlacement::ToOrigin(models[0], _heading_axis);
}

void ClipPreview::ApplyAnchor(ozz::span<math::Float4x4> _models) const {
  placement_.ApplyToModels(_models);
}

bool ClipPreview::Advance(float _dt, const PoseSetEntry* _entry) {
  const bool playing = clip.playing;
  const float previous = clip.time;
  clip.Advance(_dt);
  // Plays the trim range only: wraps (or stops) at its ends.
  if (playing && play_trim && _entry && _entry->trimmed()) {
    const float begin = _entry->begin, end = _entry->trim_end();
    if (clip.time > end || clip.time < begin) {
      if (clip.loop) {
        clip.time = clip.speed >= 0.f ? begin : end;
      } else {
        clip.time = std::clamp(clip.time, begin, end);
        clip.playing = false;
      }
    }
  }
  const bool wrapped = clip.speed >= 0.f ? clip.time < previous : clip.time > previous;
  return playing && wrapped;
}

ClipPreview::GuiResult ClipPreview::Gui(PoseSetEntry* _entry, bool _busy) {
  GuiResult result;
  const float duration = clip.duration();
  const bool trim_range = play_trim && _entry && _entry->trimmed();
  if (ImGui::Button(clip.playing ? "Pause" : "Play",
                    ImVec2(ImGui::GetFontSize() * 4.f, 0.f))) {
    const float end = trim_range ? _entry->trim_end() : duration;
    if (!clip.playing && !clip.loop && clip.time >= end) {
      clip.time = trim_range ? _entry->begin : 0.f;  // Replays from the start.
    }
    clip.playing = !clip.playing;
  }
  ImGui::SameLine();
  if (ImGui::Button("Close")) {
    result.close = true;
  }
  ImGui::SameLine();
  ImGui::Checkbox("Loop", &clip.loop);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::SliderFloat("##speed", &clip.speed, -2.f, 2.f, "speed %.2fx");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::SliderFloat("##time", &clip.time, 0.f, duration, "%.2fs")) {
    clip.playing = false;
  }
  if (_entry && _entry->trimmed() && duration > 0.f) {
    // Trimmed out parts shaded on the time slider.
    const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
    auto x = [&](float _t) { return min.x + (max.x - min.x) * _t / duration; };
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImU32 shade = IM_COL32(0, 0, 0, 150);
    draw_list->AddRectFilled(min, ImVec2(x(_entry->begin), max.y), shade);
    draw_list->AddRectFilled(ImVec2(x(_entry->trim_end()), min.y), max, shade);
    const ImU32 mark = IM_COL32(255, 170, 60, 255);
    draw_list->AddLine(ImVec2(x(_entry->begin), min.y), ImVec2(x(_entry->begin), max.y),
                       mark, 2.f);
    draw_list->AddLine(ImVec2(x(_entry->trim_end()), min.y),
                       ImVec2(x(_entry->trim_end()), max.y), mark, 2.f);
  }

  // Trim range, for the project: the part of the clip used for the common
  // poses. The clip file isn't changed.
  if (!_entry) {
    ImGui::BeginDisabled(_busy);
    if (ImGui::SmallButton("Add to pose set")) {
      result.add_to_set = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("to trim it.");
    return result;
  }
  ImGui::BeginDisabled(_busy);
  float begin = _entry->begin, end = _entry->trim_end();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Trim");
  ImGui::SetItemTooltip(
      "Part of the clip used for the common poses, saved with the project.\n"
      "The clip file isn't changed.");
  ImGui::SameLine();
  const float buttons = ImGui::CalcTextSize("[ now] nowReset").x +
                        ImGui::GetStyle().FramePadding.x * 6.f +
                        ImGui::GetStyle().ItemSpacing.x * 3.f;
  ImGui::SetNextItemWidth(std::max(80.f, ImGui::GetContentRegionAvail().x - buttons));
  if (ImGui::DragFloatRange2("##trim", &begin, &end, .01f, 0.f, _entry->duration,
                             "begin %.2fs", "end %.2fs", ImGuiSliderFlags_AlwaysClamp)) {
    _entry->SetTrim(begin, end);
  }
  ImGui::SameLine();
  if (ImGui::Button("[ now")) {
    _entry->SetTrim(clip.time, std::max(end, clip.time));
  }
  ImGui::SetItemTooltip("Begin at the current time.");
  ImGui::SameLine();
  if (ImGui::Button("] now")) {
    _entry->SetTrim(std::min(begin, clip.time), clip.time);
  }
  ImGui::SetItemTooltip("End at the current time.");
  ImGui::SameLine();
  ImGui::BeginDisabled(!_entry->trimmed());
  if (ImGui::Button("Reset")) {
    _entry->SetTrim(0.f, _entry->duration);
  }
  ImGui::EndDisabled();
  ImGui::EndDisabled();
  ImGui::Checkbox("Play trim range only", &play_trim);
  return result;
}

}  // namespace hyperpose
