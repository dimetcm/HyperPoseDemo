#include "common/transition_graph.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <imgui.h>

#include "common/clip_library.h"

namespace hyperpose {

int TransitionGraph::Draw(const CommonPoseResult& _result, int _selected,
                          const ClipLibrary& _library) {
  if (!open || _result.poses.empty()) {
    return -1;
  }
  const ImGuiIO& io = ImGui::GetIO();
  const float dpi = ImGui::GetStyle().FontScaleDpi;
  ImGui::SetNextWindowSize(ImVec2(360.f * dpi, 400.f * dpi), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowPos(
      ImVec2(io.DisplaySize.x - 370.f * dpi, io.DisplaySize.y - 410.f * dpi),
      ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Pose transitions", &open)) {
    ImGui::End();
    return -1;
  }
  ImGui::Checkbox("All poses", &all_poses);
  ImGui::SetItemTooltip("Also the other poses' transitions, in grey.");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::SliderFloat("##min_share", &min_share, 0.f, .5f,
                     "hide below %.2f share");
  if (_selected >= 0) {
    ImGui::TextColored(ImVec4(1.f, .67f, .24f, 1.f), "out of #%d", _selected + 1);
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(.35f, .67f, 1.f, 1.f), "into #%d", _selected + 1);
    ImGui::SameLine();
    ImGui::TextDisabled("(%% of source's)");
    ImGui::SetItemTooltip("Arrow shares are of the source pose's transitions.");
  }

  // Canvas.
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const ImVec2 size(std::max(avail.x, 60.f), std::max(avail.y, 60.f));
  ImGui::InvisibleButton("graph", size);
  const bool hovered_canvas = ImGui::IsItemHovered();
  const bool clicked = ImGui::IsItemClicked();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  draw_list->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), true);

  // Nodes.
  const int count = static_cast<int>(_result.poses.size());
  const float max_radius = 24.f * dpi;
  const float min_radius = 9.f * dpi;
  const ImVec2 center(origin.x + size.x * .5f, origin.y + size.y * .5f);
  // Room outside the circle for dead end arrows and their labels.
  const float kEndLength = 14.f * dpi;
  const float ring = std::max(
      10.f, std::min(size.x, size.y) * .5f - max_radius - kEndLength - 24.f * dpi);
  const float largest = static_cast<float>(_result.poses.front().frames.size());
  std::vector<ImVec2> positions(count);
  std::vector<float> radii(count);
  int hovered = -1;
  int clicked_pose = -1;
  for (int p = 0; p < count; ++p) {
    const float angle = -1.5707963f + 6.2831853f * p / count;
    positions[p] = ImVec2(center.x + ring * std::cos(angle),
                          center.y + ring * std::sin(angle));
    radii[p] = min_radius + (max_radius - min_radius) *
                                std::sqrt(_result.poses[p].frames.size() / largest);
    const float dx = io.MousePos.x - positions[p].x;
    const float dy = io.MousePos.y - positions[p].y;
    if (hovered_canvas && dx * dx + dy * dy <= radii[p] * radii[p]) {
      hovered = p;
    }
  }

  // Edges: curved to the right of the direction, so a->b and b->a don't
  // overlap, with an arrow head on the target and the share at the middle.
  // A dead end (_to -1, the clips end) is a short straight arrow out of the
  // circle, labeled "end".
  auto edge = [&](int _from, int _to, float _share, ImU32 _color, float _thickness,
                  bool _label) {
    auto toward = [](ImVec2 _p, ImVec2 _q, float _distance) {
      const float x = _q.x - _p.x, y = _q.y - _p.y;
      const float l = std::max(1e-3f, std::sqrt(x * x + y * y));
      return ImVec2(_p.x + x / l * _distance, _p.y + y / l * _distance);
    };
    const ImVec2 a = positions[_from];
    const ImVec2 b = _to >= 0 ? positions[_to]
                              : toward(center, a, ring + radii[_from] + kEndLength);
    const float dx = b.x - a.x, dy = b.y - a.y;
    const float length = std::max(1.f, std::sqrt(dx * dx + dy * dy));
    const float bend = _to >= 0 ? .15f : 0.f;
    const ImVec2 control((a.x + b.x) * .5f - dy / length * length * bend,
                         (a.y + b.y) * .5f + dx / length * length * bend);
    const ImVec2 start = toward(a, control, radii[_from]);
    const ImVec2 end = _to >= 0 ? toward(b, control, radii[_to] + 2.f) : b;
    // Arrow head along the curve tangent at the end.
    const float tx = end.x - control.x, ty = end.y - control.y;
    const float tl = std::max(1e-3f, std::sqrt(tx * tx + ty * ty));
    const ImVec2 dir(tx / tl, ty / tl);
    const float head = (6.f + _thickness * 1.5f) * dpi;
    const ImVec2 base(end.x - dir.x * head, end.y - dir.y * head);
    draw_list->AddBezierQuadratic(start, control, base, _color, _thickness);
    draw_list->AddTriangleFilled(
        end, ImVec2(base.x - dir.y * head * .5f, base.y + dir.x * head * .5f),
        ImVec2(base.x + dir.y * head * .5f, base.y - dir.x * head * .5f), _color);
    if (_label) {
      char text[16];
      std::snprintf(text, sizeof(text), _to >= 0 ? "%.0f%%" : "end %.0f%%",
                    _share * 100.f);
      const ImVec2 text_size = ImGui::CalcTextSize(text);
      // Past the arrow tip for a dead end, else mid curve.
      const ImVec2 mid =
          _to >= 0 ? ImVec2(.25f * start.x + .5f * control.x + .25f * base.x,
                            .25f * start.y + .5f * control.y + .25f * base.y)
                   : toward(end, ImVec2(2.f * end.x - start.x, 2.f * end.y - start.y),
                            .5f * std::max(text_size.x, text_size.y) + 3.f * dpi);
      const ImVec2 pos(mid.x - text_size.x * .5f, mid.y - text_size.y * .5f);
      draw_list->AddRectFilled(ImVec2(pos.x - 2.f, pos.y),
                               ImVec2(pos.x + text_size.x + 2.f, pos.y + text_size.y),
                               IM_COL32(20, 20, 20, 200), 3.f);
      draw_list->AddText(pos, _color, text);
    }
  };
  // Grey: other poses' transitions (with "All poses"). Blue: into the
  // selected pose, orange: out of it, drawn last to stay on top. Shares are
  // of the source pose's transitions.
  const ImU32 kOut = IM_COL32(255, 170, 60, 255);
  const ImU32 kIn = IM_COL32(90, 170, 255, 255);
  std::vector<bool> connected(count, false);  // Has a drawn arrow.
  std::vector<bool> targets(count, false), sources(count, false);
  for (int pass = 0; pass < 3; ++pass) {  // Others, in, out.
    for (int p = 0; p < count; ++p) {
      const PoseExits& exits = _result.exits[p];
      for (const PoseEdge& e : exits.edges) {
        const float share = exits.share(e);
        const bool out = p == _selected;
        const bool in = e.to >= 0 && e.to == _selected;
        if (share < min_share ||
            (pass == 0 && (out || in || !all_poses)) ||
            (pass == 1 && !in) || (pass == 2 && !out)) {
          continue;
        }
        connected[p] = true;
        if (e.to >= 0) {
          connected[e.to] = true;
        }
        if (pass == 0) {
          edge(p, e.to, share, IM_COL32(170, 170, 170, 170),
               (1.f + 4.f * share) * dpi, false);
        } else {
          if (e.to >= 0) {
            (out ? targets[e.to] : sources[p]) = true;
          }
          edge(p, e.to, share, out ? kOut : kIn, (1.f + 7.f * share) * dpi, true);
        }
      }
    }
  }

  // Nodes over the edges: selected orange, its targets / sources tinted,
  // nodes without a drawn arrow dimmed.
  for (int p = 0; p < count; ++p) {
    const bool dim = !connected[p] && p != _selected;
    const ImU32 fill = p == _selected     ? IM_COL32(230, 140, 40, 255)
                       : targets[p] ? IM_COL32(120, 90, 50, 255)
                       : sources[p] ? IM_COL32(50, 85, 125, 255)
                       : dim        ? IM_COL32(40, 40, 40, 255)
                                    : IM_COL32(75, 75, 75, 255);
    draw_list->AddCircleFilled(positions[p], radii[p], fill);
    draw_list->AddCircle(positions[p], radii[p],
                         p == hovered ? IM_COL32(255, 255, 255, 255)
                         : dim        ? IM_COL32(90, 90, 90, 255)
                                      : IM_COL32(160, 160, 160, 255),
                         0, p == hovered ? 2.f : 1.f);
    char text[8];
    std::snprintf(text, sizeof(text), "#%d", p + 1);
    const ImVec2 text_size = ImGui::CalcTextSize(text);
    draw_list->AddText(ImVec2(positions[p].x - text_size.x * .5f,
                              positions[p].y - text_size.y * .5f),
                       dim ? IM_COL32(120, 120, 120, 255) : IM_COL32(255, 255, 255, 255),
                       text);
  }
  draw_list->PopClipRect();

  if (hovered >= 0) {
    const CommonPose& pose = _result.poses[hovered];
    const CommonPose::Frame& frame = pose.frames.front();
    const std::string name = _result.clips[frame.clip].stem().string();
    const std::string next = PoseExitsText(_result.exits[hovered], 4);
    ImGui::SetTooltip("#%d: %.1f%% of frames, spread %.1f cm\n%s @ %.1fs  %s\nnext:%s",
                      hovered + 1, 100.f * pose.frames.size() / _result.num_frames,
                      pose.spread * 100.f, name.c_str(), frame.time,
                      _library.Description(name).c_str(),
                      next.empty() ? " none" : next.c_str());
    if (clicked) {
      clicked_pose = hovered;
    }
  }
  ImGui::End();
  return clicked_pose;
}

}  // namespace hyperpose
