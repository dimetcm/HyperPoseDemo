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
  following_ = false;
  tag_path_.clear();
  // Loaded clips may not match the next skeleton.
  crossfade_.Clear();
}

bool PoseWalk::Pick(int _from, bool _after_end, Segment* _segment) {
  Segment& segment = *_segment;
  segment = Segment();
  segment.after_end = _after_end;
  // Follow clip: once a tagged clip steps into the tag, it plays on,
  // transition after transition, to its end (then a new search). An
  // untagged clip isn't followed: the next step picks a tagged one if any.
  auto clip_tagged = [&](int _clip) {
    const auto& tags = result_->clip_tags[_clip];
    return std::find(tags.begin(), tags.end(), target_tag) != tags.end();
  };
  if (!target_tag.empty() && follow_clip && !_after_end && segment_.transition >= 0 &&
      segment_.to == _from &&
      (following_ || (TaggedPoses(*result_)[_from] && clip_tagged(segment_.clip)))) {
    const int index = segment_.transition + 1;
    if (index < static_cast<int>(result_->transitions.size()) &&
        result_->transitions[index].clip == segment_.clip) {
      const PoseTransition& transition = result_->transitions[index];
      following_ = true;
      tag_path_.clear();
      segment.transition = index;
      segment.from = _from;
      segment.to = transition.to;
      segment.clip = transition.clip;
      segment.time = transition.time;
      segment.start = segment_.time;  // Where the clip is: plays on.
      segment.tag = Segment::Tag::kFollowing;
      return true;
    }
  }
  following_ = false;
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
  if (!target_tag.empty()) {
    // The next step of the path to the current target, or the first of a
    // path to a new random target (reached, or off the path: jumps, a tag
    // or threshold change).
    if (planned_tag_ != target_tag || planned_threshold_ != tag_threshold) {
      tag_path_.clear();
    }
    int next = -1;
    const auto at = std::find(tag_path_.begin(), tag_path_.end(), _from);
    if (at != tag_path_.end() && at + 1 != tag_path_.end()) {
      next = *(at + 1);
    } else if (PlanTagPath(_from)) {
      next = tag_path_[1];
    }
    if (next >= 0) {
      for (size_t e = 0; e < exits.edges.size(); ++e) {
        weights[e] = exits.edges[e].to == next ? weights[e] : 0.;
      }
      segment.tag = TaggedPoses(*result_)[_from] ? Segment::Tag::kInTag
                                                 : Segment::Tag::kSeeking;
    } else {
      tag_path_.clear();
      segment.tag = Segment::Tag::kUnreachable;
    }
  }
  const PoseEdge& edge =
      exits.edges[std::discrete_distribution<int>(weights.begin(), weights.end())(rng_)];
  // Which clip example: the best fitting one with a target tag, else random.
  const int index = !target_tag.empty()
                        ? BestTransition(edge)
                        : edge.transitions[std::uniform_int_distribution<size_t>(
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

float PoseWalk::PoseDifference(int _a, int _b) const {
  const CommonPose::Frame& a = result_->poses[_a].frames.front();
  const CommonPose::Frame& b = result_->poses[_b].frames.front();
  const PoseFeatures& fa = result_->features[a.clip];
  const PoseFeatures& fb = result_->features[b.clip];
  return PoseFrameDifference(fa, FeatureFrame(fa, a.time), fb, FeatureFrame(fb, b.time),
                             velocity_weight);
}

int PoseWalk::FeatureFrame(const PoseFeatures& _features, float _time) {
  return std::clamp(static_cast<int>(std::lround(_time * _features.frame_rate)), 0,
                    _features.num_frames - 1);
}

int PoseWalk::BestTransition(const PoseEdge& _edge) const {
  auto tagged_clip = [&](int _transition) {
    const auto& tags = result_->clip_tags[result_->transitions[_transition].clip];
    return std::find(tags.begin(), tags.end(), target_tag) != tags.end();
  };
  const bool any_tagged = std::any_of(_edge.transitions.begin(), _edge.transitions.end(),
                                      tagged_clip);
  int best = _edge.transitions.front();
  float best_score = std::numeric_limits<float>::max();
  for (int index : _edge.transitions) {
    if (any_tagged && !tagged_clip(index)) {
      continue;
    }
    const PoseTransition& transition = result_->transitions[index];
    const float start = std::max(EnteredAt(*result_, index), transition.time - max_stay);
    float score;
    if (segment_.clip >= 0) {
      // Closest to the current clip where it reaches the next pose (blend).
      const PoseFeatures& a = result_->features[segment_.clip];
      const PoseFeatures& b = result_->features[transition.clip];
      score = PoseFrameDifference(a, FeatureFrame(a, segment_.time), b,
                                  FeatureFrame(b, start), velocity_weight);
    } else {
      score = -(transition.time - start);  // First segment: the longest stay.
    }
    if (score < best_score) {
      best_score = score;
      best = index;
    }
  }
  return best;
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
  const float most = shares.empty() ? 0.f : *std::max_element(shares.begin(), shares.end());
  // The threshold if any pose reaches it, else the highest share.
  const float threshold = std::min(tag_threshold, most);
  std::vector<bool> tagged(shares.size(), false);
  for (size_t p = 0; p < shares.size(); ++p) {
    tagged[p] = shares[p] > 0.f && shares[p] >= threshold - 1e-6f;
  }
  return tagged;
}

void PoseWalk::Paths(int _from, const std::vector<bool>* _through, std::vector<double>* _cost,
                     std::vector<int>* _previous) const {
  const int count = static_cast<int>(result_->poses.size());
  std::vector<double>& cost = *_cost;
  std::vector<int>& previous = *_previous;
  cost.assign(count, std::numeric_limits<double>::infinity());
  previous.assign(count, -1);
  std::vector<bool> done(count, false);
  using Entry = std::pair<double, int>;  // Cost, pose.
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
  auto relax = [&](int _pose, double _cost) {
    for (const PoseEdge& edge : result_->exits[_pose].edges) {
      if (edge.to < 0 || (_through && !(*_through)[edge.to])) {
        continue;
      }
      const double to_cost = _cost + PoseDifference(_pose, edge.to);
      if (to_cost < cost[edge.to]) {
        cost[edge.to] = to_cost;
        previous[edge.to] = _pose;
        open.push({to_cost, edge.to});
      }
    }
  };
  // _from isn't settled at 0: reached again only through a loop.
  relax(_from, 0.);
  while (!open.empty()) {
    const auto [at_cost, pose] = open.top();
    open.pop();
    if (done[pose] || at_cost > cost[pose]) {
      continue;
    }
    done[pose] = true;
    relax(pose, at_cost);
  }
}

bool PoseWalk::PlanTagPath(int _from) {
  tag_path_.clear();
  planned_tag_ = target_tag;
  planned_threshold_ = tag_threshold;
  const std::vector<bool> tagged = TaggedPoses(*result_);
  const int count = static_cast<int>(tagged.size());
  // Through tagged poses only if possible, else through any.
  for (const std::vector<bool>* through : {&tagged, static_cast<const std::vector<bool>*>(nullptr)}) {
    std::vector<double> cost;
    std::vector<int> previous;
    Paths(_from, through, &cost, &previous);
    std::vector<int> targets;
    for (int p = 0; p < count; ++p) {
      if (tagged[p] && p != _from && cost[p] < std::numeric_limits<double>::infinity()) {
        targets.push_back(p);
      }
    }
    if (targets.empty() && tagged[_from] && cost[_from] < std::numeric_limits<double>::infinity()) {
      targets.push_back(_from);  // The only one: back to it through a loop.
    }
    if (targets.empty()) {
      continue;
    }
    int target = targets[std::uniform_int_distribution<size_t>(0, targets.size() - 1)(rng_)];
    if (follow_clip) {
      // The best tag share, then the smoothest path.
      const std::vector<float> shares = TagShares(*result_);
      for (int candidate : targets) {
        if (shares[candidate] > shares[target] + 1e-6f ||
            (shares[candidate] > shares[target] - 1e-6f && cost[candidate] < cost[target])) {
          target = candidate;
        }
      }
    }
    // Back along the predecessors to the pose reached from _from.
    tag_path_.assign(1, target);
    for (int p = target; previous[p] != _from && tag_path_.size() <= static_cast<size_t>(count);) {
      p = previous[p];
      tag_path_.push_back(p);
    }
    tag_path_.push_back(_from);
    std::reverse(tag_path_.begin(), tag_path_.end());
    return true;
  }
  return false;
}

void PoseWalk::Plan(float _now) {
  Segment& next = next_;
  blend_at_ = segment_.time;
  const PoseFeatures& a = result_->features[segment_.clip];
  const PoseFeatures& b = result_->features[next.clip];
  auto frame = &PoseWalk::FeatureFrame;
  // Carries on in the same clip: no blend. Always with best match; else
  // when it starts where the clip already is (a stay within max_stay).
  if (next.clip == segment_.clip && next.transition == segment_.transition + 1 &&
      (best_match || next.start <= segment_.time + 1e-4f)) {
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
