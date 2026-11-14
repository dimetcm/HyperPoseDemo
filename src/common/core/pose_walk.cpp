#include "common/core/pose_walk.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>
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
  tag_path_.clear();
  // Loaded clips may not match the next skeleton.
  crossfade_.Clear();
}

bool PoseWalk::Pick(int _from, bool _after_end, Segment* _segment) {
  Segment& segment = *_segment;
  segment = Segment();
  segment.after_end = _after_end;
  const std::vector<PoseTransition>& transitions = result_->transitions;
  const std::vector<PoseBridge>& bridges = result_->bridges;
  const int count = static_cast<int>(transitions.size());
  // Go to tag state of the picked segment, from where it starts.
  auto tag_state = [&](int _pose) {
    return target_tag.empty()          ? Segment::Tag::kOff
           : TaggedPoses(*result_)[_pose] ? Segment::Tag::kInTag
                                          : Segment::Tag::kSeeking;
  };
  // The plan step after the playing segment's, if it's still this plan's.
  const bool on_plan = !_after_end && segment_.plan_id == plan_id_ &&
                       segment_.plan_step >= 0 &&
                       segment_.plan_step + 1 < static_cast<int>(plan_.size());

  // Right after a bridge: its other clip plays on from where the bridge
  // lands, to its next transition.
  if (!_after_end && segment_.bridge >= 0 && bridges[segment_.bridge].next_transition >= 0) {
    const PoseBridge& bridge = bridges[segment_.bridge];
    FillSegment(bridge.to, bridge.next_transition, -1, &segment);
    segment.start = bridge.to_time;
    segment.tag = tag_state(bridge.to);
    if (on_plan && plan_[segment_.plan_step + 1] == bridge.next_transition) {
      segment.plan_id = plan_id_;
      segment.plan_step = segment_.plan_step + 1;
    }
    return true;
  }

  // Right after a clip end, only transitions to a pose, so the walk goes on
  // instead of ending again.
  auto usable = [&](const PoseEdge& _edge) { return !_after_end || _edge.to >= 0; };
  auto leads_on = [&](int _pose) {
    const auto& edges = result_->exits[_pose].edges;
    return std::any_of(edges.begin(), edges.end(), usable) ||
           std::any_of(bridges.begin(), bridges.end(),
                       [&](const PoseBridge& _bridge) { return _bridge.from == _pose; });
  };
  if (!leads_on(_from)) {
    // Nowhere to go from this pose: jumps to a pose the clips start in,
    // weighted by how many do.
    std::vector<double> weights(result_->poses.size(), 0.);
    for (const PoseTransition& transition : transitions) {
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

  // Go to tag: the next step of the plan, or the first of a new plan
  // (target reached, or off the plan: jumps, clip ends, a tag change).
  int planned = -1;  // A plan entry (see plan_).
  if (!target_tag.empty()) {
    if (planned_tag_ != target_tag) {
      plan_.clear();
      ++plan_id_;
    }
    const bool on_clip =
        !segment.jump && !_after_end && segment_.transition >= 0 && segment_.to == _from;
    // Checked again: a tag change above drops the plan.
    const bool next_step = on_plan && segment_.plan_id == plan_id_ &&
                           segment_.plan_step + 1 < static_cast<int>(plan_.size());
    if (!segment.jump && next_step && segment_.to == _from) {
      segment.plan_step = segment_.plan_step + 1;
    } else if (PlanTagPath(_from, on_clip ? segment_.transition : -1)) {
      segment.plan_step = 0;
    }
    if (segment.plan_step >= 0) {
      segment.plan_id = plan_id_;
      planned = plan_[segment.plan_step];
    } else {
      plan_.clear();
      tag_path_.clear();
    }
  }

  // Else random: a transition weighted by how often the clips go there, or
  // a bridge, each weighing as one transition.
  int index = -1, bridge = -1;
  if (planned >= count) {
    bridge = planned - count;
  } else if (planned >= 0) {
    index = planned;
  } else {
    const PoseExits& exits = result_->exits[_from];
    std::vector<double> weights;
    for (const PoseEdge& edge : exits.edges) {
      weights.push_back(usable(edge) ? static_cast<double>(edge.transitions.size()) : 0.);
    }
    std::vector<int> from_bridges;
    for (size_t b = 0; b < bridges.size(); ++b) {
      if (bridges[b].from == _from) {
        from_bridges.push_back(static_cast<int>(b));
        weights.push_back(1.);
      }
    }
    const size_t pick =
        std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng_);
    if (pick < exits.edges.size()) {
      const PoseEdge& edge = exits.edges[pick];
      index = edge.transitions[std::uniform_int_distribution<size_t>(
          0, edge.transitions.size() - 1)(rng_)];
    } else {
      bridge = from_bridges[pick - exits.edges.size()];
    }
  }
  const int plan_step = segment.plan_step, plan_id = segment.plan_id;
  const bool after_end = segment.after_end, jump = segment.jump;
  FillSegment(_from, index, bridge, &segment);
  segment.plan_step = plan_step;
  segment.plan_id = plan_id;
  segment.after_end = after_end;
  segment.jump = jump;
  segment.tag = target_tag.empty() ? Segment::Tag::kOff
                : planned < 0      ? Segment::Tag::kUnreachable
                                   : tag_state(_from);
  // The playing clip carrying on plays on from where it is.
  const bool on_clip = !jump && !after_end && segment_.clip == segment.clip &&
                       segment_.to == _from && segment_.bridge < 0;
  if (on_clip && segment_.time <= segment.time &&
      segment_.time >= (bridge >= 0 ? bridges[bridge].from_entered
                                    : EnteredAt(*result_, index)) - 1e-4f) {
    segment.start = segment_.time;
  }
  return true;
}

void PoseWalk::FillSegment(int _from, int _index, int _bridge, Segment* _segment) const {
  Segment& segment = *_segment;
  segment = Segment();
  segment.from = _from;
  if (_bridge >= 0) {
    // Plays the bridge's clip up to it, from when it entered the pose.
    const PoseBridge& bridge = result_->bridges[_bridge];
    segment.bridge = _bridge;
    segment.to = bridge.to;
    segment.clip = bridge.from_clip;
    segment.time = bridge.from_time;
    segment.start = std::max(bridge.from_entered, bridge.from_time - max_stay);
    return;
  }
  const PoseTransition& transition = result_->transitions[_index];
  segment.transition = _index;
  segment.to = transition.to;
  segment.clip = transition.clip;
  segment.time = transition.time;
  // From where the clip entered the pose, at most max_stay before.
  segment.start = std::max(EnteredAt(*result_, _index), transition.time - max_stay);
}

int PoseWalk::FeatureFrame(const PoseFeatures& _features, float _time) {
  return std::clamp(static_cast<int>(std::lround(_time * _features.frame_rate)), 0,
                    _features.num_frames - 1);
}

std::vector<float> PoseWalk::TagShares(const CommonPoseResult& _result) const {
  std::vector<float> shares(_result.poses.size(), 0.f);
  for (size_t p = 0; p < shares.size() && !target_tag.empty(); ++p) {
    for (const CommonPose::Tag& tag : _result.poses[p].tags) {
      if (tag.name == target_tag) {
        shares[p] = tag.share;
      }
    }
  }
  return shares;
}

std::vector<bool> PoseWalk::TaggedPoses(const CommonPoseResult& _result) const {
  const std::vector<float> shares = TagShares(_result);
  std::vector<bool> tagged(shares.size(), false);
  for (size_t p = 0; p < shares.size(); ++p) {
    tagged[p] = shares[p] > 0.f;
  }
  return tagged;
}

bool PoseWalk::PlanTagPath(int _from, int _start) {
  plan_.clear();
  tag_path_.clear();
  planned_tag_ = target_tag;
  const std::vector<PoseTransition>& transitions = result_->transitions;
  const std::vector<bool> tagged = TaggedPoses(*result_);
  const std::vector<float> shares = TagShares(*result_);
  const int count = static_cast<int>(transitions.size());
  const int poses = static_cast<int>(tagged.size());
  auto clip_tagged = [&](int _clip) {
    const auto& tags = result_->clip_tags[_clip];
    return std::find(tags.begin(), tags.end(), target_tag) != tags.end();
  };
  // Costs: the shortest path, 1 per transition; among equally short ones,
  // the fewest clip switches (blends), then tagged clips. The tie-breakers
  // stay well below 1 for any realistic path length.
  const double kSwitch = 1e-3;
  auto step = [&](int _transition) {
    return clip_tagged(transitions[_transition].clip) ? 1. : 1. + 1e-6;
  };
  const std::vector<PoseBridge>& bridges = result_->bridges;
  const int bridge_count = static_cast<int>(bridges.size());
  // Nodes: transitions [0, count), each a clip arriving in its `to` pose,
  // pose hubs [count, count + poses), switching clips in a pose, then
  // bridges [count + poses, ...), a clip blended into in their `to` pose.
  // With no playing clip, the search starts from _from's hub.
  const int start = _start >= 0 ? _start : count + _from;
  const int first_bridge = count + poses;
  const int nodes = first_bridge + bridge_count;
  const double kInfinity = std::numeric_limits<double>::infinity();
  // The pose a node arrives in.
  auto arrival = [&](int _node) {
    return _node < count ? transitions[_node].to
           : _node >= first_bridge ? bridges[_node - first_bridge].to
                                   : _node - count;
  };

  std::vector<double> cost(nodes, kInfinity);
  std::vector<int> previous(nodes, -1);
  std::vector<bool> done(nodes, false);
  using Entry = std::pair<double, int>;  // Cost, node.
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
  auto push = [&](int _node, int _parent, double _cost) {
    if (_cost < cost[_node]) {
      cost[_node] = _cost;
      previous[_node] = _parent;
      open.push({_cost, _node});
    }
  };
  // A transition the path may take: to a pose (not a clip end).
  auto allowed = [&](int _transition) { return transitions[_transition].to >= 0; };
  // A goal: a transition or bridge arriving in a tagged pose other than
  // _from.
  auto goal = [&](int _node) {
    const bool arrives = _node < count || _node >= first_bridge;
    const int pose = arrival(_node);
    return arrives && _node != start && pose >= 0 && tagged[pose] && pose != _from;
  };
  cost[start] = 0.;
  open.push({0., start});
  int stopped = -1;
  while (!open.empty()) {
    const auto [at_cost, node] = open.top();
    open.pop();
    if (done[node] || at_cost > cost[node]) {
      continue;
    }
    done[node] = true;
    if (goal(node) && shares[arrival(node)] >= 1.f - 1e-6f) {
      stopped = node;  // 100% can't be beaten, and settled first: cheapest.
      break;
    }
    if (node < count) {
      // Follow: the same clip's next transition. Switch: through the hub.
      const int next = node + 1;
      if (next < count && transitions[next].clip == transitions[node].clip &&
          allowed(next)) {
        push(next, node, at_cost + step(next));
      }
      push(count + transitions[node].to, node, at_cost + kSwitch);
    } else if (node >= first_bridge) {
      // After a bridge, its clip plays on (no switch: it was just blended
      // into).
      const int next = bridges[node - first_bridge].next_transition;
      if (next >= 0 && allowed(next)) {
        push(next, node, at_cost + step(next));
      }
    } else {
      const int pose = node - count;
      for (const PoseEdge& edge : result_->exits[pose].edges) {
        for (int transition : edge.transitions) {
          if (allowed(transition)) {
            push(transition, node, at_cost + step(transition));
          }
        }
      }
      for (int b = 0; b < bridge_count; ++b) {
        if (bridges[b].from == pose) {
          push(first_bridge + b, node, at_cost + 1. + kSwitch);
        }
      }
    }
  }

  // The target: the highest tag share, then the cheapest way there.
  int best = stopped;
  if (best < 0) {
    for (int node = 0; node < nodes; ++node) {
      if (!goal(node) || cost[node] == kInfinity) {
        continue;
      }
      if (best < 0) {
        best = node;
        continue;
      }
      const float share = shares[arrival(node)];
      const float best_share = shares[arrival(best)];
      if (share > best_share + 1e-6f ||
          (share > best_share - 1e-6f && cost[node] < cost[best])) {
        best = node;
      }
    }
  }
  if (best < 0) {
    return false;
  }
  // The path's transitions and bridges back to the start, hubs aside.
  for (int node = best; node >= 0 && node != start; node = previous[node]) {
    if (node < count) {
      plan_.push_back(node);
    } else if (node >= first_bridge) {
      plan_.push_back(count + node - first_bridge);
    }
  }
  std::reverse(plan_.begin(), plan_.end());
  ++plan_id_;
  tag_path_.assign(1, _from);
  for (int entry : plan_) {
    tag_path_.push_back(entry < count ? transitions[entry].to : bridges[entry - count].to);
  }
  return !plan_.empty();
}

void PoseWalk::Plan(float _now) {
  Segment& next = next_;
  blend_at_ = segment_.time;
  const PoseFeatures& a = result_->features[segment_.clip];
  const PoseFeatures& b = result_->features[next.clip];
  auto frame = &PoseWalk::FeatureFrame;
  // Carries on in the same clip: no blend. Always with best match; else
  // when it starts where the clip already is (a stay within max_stay).
  const bool same_clip_on = next.clip == segment_.clip &&
                           std::fabs(next.start - segment_.time) < 1e-4f;
  const bool consecutive = next.clip == segment_.clip && segment_.bridge < 0 &&
                           next.transition >= 0 &&
                           next.transition == segment_.transition + 1;
  if (same_clip_on || (consecutive && best_match)) {
    next.continued = true;
    next.start = segment_.time;
    next.difference = 0.f;
    return;
  }
  if (!best_match) {
    next.difference = PoseFrameDifference(a, frame(a, segment_.time), b,
                                          frame(b, next.start), velocity_weight);
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
