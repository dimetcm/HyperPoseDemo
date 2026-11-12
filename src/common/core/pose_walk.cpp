#include "common/core/pose_walk.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "common/core/pose_match.h"

namespace math = ozz::math;

namespace hyperpose {
PoseWalk::PoseWalk() : rng_(std::random_device{}()) {}
PoseWalk::~PoseWalk() = default;

bool PoseWalk::Start(const CommonPoseResult* _result,
                     const ozz::animation::Skeleton* _skeleton,
                     int _heading_axis, int _pose) {
  Stop();
  if (_result->poses.empty()) {
    return false;
  }
  result_ = _result;
  crossfade_.root_motion = root_motion;
  crossfade_.blend = blend;
  crossfade_.Reset(_skeleton, _heading_axis);
  stats_ = BlendStats();
  if (_pose < 0) {
    std::vector<double> weights;
    for (const CommonPose& pose : result_->poses) {
      weights.push_back(static_cast<double>(pose.frames.size()));
    }
    _pose = std::discrete_distribution<int>(weights.begin(), weights.end())(rng_);
  }
  if (!Pick(_pose, false, &next_) || !Begin(next_)) {
    Stop();
    return false;
  }
  return true;
}

void PoseWalk::Stop() {
  result_ = nullptr;
  segment_ = Segment();
  pending_ = false;
  // Loaded clips may not match the next skeleton.
  crossfade_.Clear();
}

bool PoseWalk::Pick(int _from, bool _after_end, Segment* _segment) {
  Segment& segment = *_segment;
  segment = Segment();
  segment.after_end = _after_end;
  // Right after a clip end, only transitions to a pose, so the walk goes on
  // instead of ending again.
  auto usable = [&](const PoseEdge& _edge) { return !_after_end || _edge.to >= 0; };
  auto leads_on = [&](int _pose) {
    const auto& edges = result_->exits[_pose].edges;
    return std::any_of(edges.begin(), edges.end(), usable);
  };
  if (!leads_on(_from)) {
    // Nowhere to go from this pose: jumps to a pose the clips start in,
    // weighted by how many do.
    std::vector<double> weights(result_->poses.size(), 0.);
    for (const PoseTransition& transition : result_->transitions) {
      if (transition.from < 0 && leads_on(transition.to)) {
        weights[transition.to] += 1.;
      }
    }
    if (std::all_of(weights.begin(), weights.end(), [](double _w) { return _w == 0.; })) {
      return false;
    }
    _from = std::discrete_distribution<int>(weights.begin(), weights.end())(rng_);
    segment.jump = true;
  }
  // Next pose (or the clip end), weighted by how often the clips go there,
  // then one of the transitions there.
  const PoseExits& exits = result_->exits[_from];
  std::vector<double> weights;
  for (const PoseEdge& edge : exits.edges) {
    weights.push_back(usable(edge) ? static_cast<double>(edge.transitions.size()) : 0.);
  }
  const PoseEdge& edge =
      exits.edges[std::discrete_distribution<int>(weights.begin(), weights.end())(rng_)];
  const int index = edge.transitions[std::uniform_int_distribution<size_t>(
      0, edge.transitions.size() - 1)(rng_)];
  const PoseTransition& transition = result_->transitions[index];
  segment.transition = index;
  segment.from = _from;
  segment.to = transition.to;
  segment.clip = transition.clip;
  segment.time = transition.time;
  segment.start = std::max(EnteredAt(*result_, index), transition.time - max_stay);
  return true;
}

void PoseWalk::Plan(float _now) {
  Segment& next = next_;
  blend_at_ = segment_.time;
  const PoseFeatures& a = result_->features[segment_.clip];
  const PoseFeatures& b = result_->features[next.clip];
  auto frame = [](const PoseFeatures& _f, float _time) {
    return std::clamp(static_cast<int>(std::lround(_time * _f.frame_rate)), 0,
                      _f.num_frames - 1);
  };
  if (!best_match) {
    next.difference = PoseFrameDifference(a, frame(a, segment_.time), b,
                                          frame(b, next.start), velocity_weight);
    return;
  }
  if (next.clip == segment_.clip && next.transition == segment_.transition + 1) {
    // Carries on in the same clip: no blend.
    next.continued = true;
    next.start = segment_.time;
    next.difference = 0.f;
    return;
  }
  // Closest pair: this clip from now until match_horizon past its transition,
  // the next clip while it's in the shared pose.
  const int a_first = frame(a, _now);
  const int a_last = std::max(a_first, frame(a, segment_.time + match_horizon));
  const int b_first = frame(b, next.start);
  const int b_last = std::max(b_first, frame(b, next.time));
  int best_a = a_first, best_b = b_first;
  float best = std::numeric_limits<float>::max();
  for (int i = a_first; i <= a_last; ++i) {
    for (int j = b_first; j <= b_last; ++j) {
      const float difference = PoseFrameDifference(a, i, b, j, velocity_weight);
      if (difference < best) {
        best = difference;
        best_a = i;
        best_b = j;
      }
    }
  }
  blend_at_ = best_a / a.frame_rate;
  next.start = best_b / b.frame_rate;
  next.difference = best;
}

bool PoseWalk::Begin(const Segment& _segment) {
  if (!crossfade_.Play(result_->clips[_segment.clip], _segment.start)) {
    return false;
  }
  segment_ = _segment;
  segment_.time = std::min(segment_.time, crossfade_.current().duration());
  return true;
}

void PoseWalk::ResetPosition() { crossfade_.ResetPosition(); }

bool PoseWalk::Update(float _dt, ozz::span<math::Float4x4> _models) {
  if (!active()) {
    return false;
  }
  crossfade_.root_motion = root_motion;
  crossfade_.blend = blend;
  crossfade_.Advance(_dt);
  const ClipPlayer& current = crossfade_.current();
  const float now = current.time;
  const bool done = !current.playing;  // Reached the clip end.
  if (!pending_) {
    // The next segment is picked when the clip reaches the next pose, or
    // match_horizon before for a best match blend.
    const float trigger = best_match ? segment_.time - match_horizon : segment_.time;
    if (now >= trigger || done) {
      // After a clip end, goes on from the pose it ended in, with another
      // clip that leaves that pose.
      const bool ended = segment_.to < 0;
      if (!Pick(ended ? segment_.from : segment_.to, ended, &next_)) {
        Stop();
        return false;
      }
      Plan(now);
      pending_ = true;
    }
  }
  if (pending_ && (now >= blend_at_ || done)) {
    pending_ = false;
    if (next_.continued) {
      ++stats_.continued;
      segment_ = next_;
    } else {
      ++stats_.blends;
      stats_.total_difference += next_.difference;
      if (!Begin(next_)) {
        Stop();
        return false;
      }
    }
  }
  return crossfade_.Sample(_models);
}

float PoseWalk::clip_time() const {
  return crossfade_.empty() ? 0.f : crossfade_.current().time;
}
float PoseWalk::clip_duration() const {
  return crossfade_.empty() ? 0.f : crossfade_.current().duration();
}

}  // namespace hyperpose
