#pragma once

// Random walk over the common pose transition graph: from the current pose,
// picks the next one (or the clip end) weighted by how often the clips go
// there, plays one of those transitions (a clip segment), then picks again
// from the reached pose (the same one after a clip end).
// Segments are crossfaded, with every clip's root aligned to the origin, so
// the character stays in place (see root_motion).

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/core/clip_crossfade.h"
#include "common/core/clip_player.h"
#include "common/core/common_pose.h"
#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/span.h"

namespace ozz::animation {
class Skeleton;
}

namespace hyperpose {

class PoseWalk {
 public:
  PoseWalk();
  ~PoseWalk();

  // Settings (seconds). Each pose plays as long as the example clip stayed in
  // it, up to max_stay.
  float max_stay = 4.f;
  float blend = .25f;  // Crossfade between segments.

  // Best match blend (narrow phase, after the common pose transition is
  // picked): the next segment is picked match_horizon before the clip
  // reaches the next pose, then the blend is placed at the closest pair of
  // frames (see PoseFrameDifference): this clip from now until
  // match_horizon past its transition, the next clip while it's in the
  // shared pose. A transition that carries on in the same clip keeps
  // playing it. Off: blends when the clip reaches the next pose, into where
  // the next clip entered it.
  bool best_match = false;
  float match_horizon = .5f;
  float velocity_weight = .1f;  // Seconds, see PoseFrameDifference.

  // Root motion: the character moves and turns as the clips do, each segment
  // continuing from where the previous one left the root. Off: the root
  // stays above the origin, facing the default heading.
  bool root_motion = false;
  // Moves the character back to the origin, facing the default heading.
  void ResetPosition();

  // Go to tag (empty: the random walk). A pose counts as tagged when enough
  // of its frames come from clips with the tag (CommonPose::tags, see
  // tag_threshold). The walk picks a random tagged pose as its target and
  // follows the smoothest path to it (Dijkstra, transitions costing the pose
  // difference, see PoseDifference), through tagged poses if possible;
  // reached, it picks the next target. No tagged pose reachable: the random
  // walk. The clip example of each transition isn't random (see
  // BestTransition). Takes effect from the next pick.
  std::string target_tag;
  // Share of a pose's frames from clips with target_tag for the pose to count
  // as tagged (0..1). If no pose reaches it, the poses with the highest share
  // count.
  float tag_threshold = .5f;
  // Per pose of _result, whether it counts as tagged (see tag_threshold).
  std::vector<bool> TaggedPoses(const CommonPoseResult& _result) const;
  // Per pose of _result, its share of frames from clips with target_tag.
  std::vector<float> TagShares(const CommonPoseResult& _result) const;
  // Follow clip: the target is the reachable tagged pose with the best tag
  // share (then the smoothest path), and once a step into the tag is made,
  // that clip plays on to its end, transition after transition, without
  // blends; at its end, a new search from where it ended.
  bool follow_clip = false;

  // Starts walking from _pose (-1: a random one, weighted by frames). _result
  // and _skeleton must outlive the walk (or Stop it before they change).
  bool Start(const CommonPoseResult* _result,
             const ozz::animation::Skeleton* _skeleton, int _heading_axis,
             int _pose);
  void Stop();
  bool active() const { return result_ != nullptr; }

  // Advances by _dt and outputs the pose, aligned to the origin.
  bool Update(float _dt, ozz::span<ozz::math::Float4x4> _models);

  // Played segment: a transition from -> to of a clip, from when the clip
  // entered `from` (start, at most max_stay before) until it reaches `to`
  // (time). The next segment starts where its clip entered `to`,
  // crossfading from this clip, which plays on into `to`. A transition to
  // -1 plays until the clip ends; the next segment then goes on from the
  // same pose (after_end), with a transition of another clip out of it. A
  // pose with nowhere to go jumps instead: `from` becomes a pose the clips
  // start in.
  struct Segment {
    int transition = -1;  // Index in CommonPoseResult::transitions.
    int from = -1;
    int to = -1;
    bool after_end = false;
    bool jump = false;
    int clip = -1;  // Index in CommonPoseResult::clips.
    float start = 0.f;
    float time = 0.f;
    // How it was entered: the pose difference of the blend into it (m, -1
    // for the first segment), or continued: the previous segment's clip
    // carried on into it, no blend.
    float difference = -1.f;
    bool continued = false;
    // How it was picked for target_tag.
    enum class Tag {
      kOff,          // No target tag: random walk.
      kSeeking,      // From an untagged pose, on the path to a tagged one.
      kInTag,        // From a tagged pose, on to the next target.
      kUnreachable,  // No tagged pose reachable: random walk.
      kFollowing,    // Follow clip: the clip that stepped into the tag plays on.
    };
    Tag tag = Tag::kOff;
  };
  const Segment& segment() const { return segment_; }
  // With target_tag: the planned path, poses to the current target (its
  // last pose).
  const std::vector<int>& tag_path() const { return tag_path_; }

  // Blends since the walk started, to compare blend settings.
  struct BlendStats {
    int blends = 0;
    float total_difference = 0.f;  // m.
    int continued = 0;             // Segments entered without a blend.
    float mean() const { return blends ? total_difference / blends : 0.f; }
  };
  const BlendStats& stats() const { return stats_; }
  // Playback time in the segment clip.
  float clip_time() const;
  float clip_duration() const;

 private:
  // Picks the segment out of _from (broad phase); _after_end: right after a
  // clip end, so not to a clip end again.
  bool Pick(int _from, bool _after_end, Segment* _segment);
  // Places the blend into next_ (blend_at_, next_.start), from _now in the
  // current segment's clip.
  void Plan(float _now);
  // Starts playing _segment's clip, crossfading over the playing ones.
  bool Begin(const Segment& _segment);
  // With target_tag: the clip example of _edge to play, not random: among
  // the transitions from clips with target_tag (all if none), the one whose
  // start is closest to the current clip where it reaches the next pose
  // (PoseFrameDifference, velocity_weight); the longest stay at the start.
  int BestTransition(const PoseEdge& _edge) const;
  // Frame of _features at _time (rounded, clamped).
  static int FeatureFrame(const PoseFeatures& _features, float _time);
  // Pose difference between common poses _a and _b (m): PoseFrameDifference
  // of their most typical frames (closest to their centers), with
  // velocity_weight. The cost of a transition in the path search.
  float PoseDifference(int _a, int _b) const;
  // Smoothest paths from _from to every pose (Dijkstra, PoseDifference
  // costs, clip ends aside): costs (infinite: unreachable) and predecessors.
  // _through: the poses paths may enter (nullptr: any). _from itself is only
  // reached back through a loop.
  void Paths(int _from, const std::vector<bool>* _through, std::vector<double>* _cost,
             std::vector<int>* _previous) const;
  // Plans tag_path_ from _from to a random tagged pose, through tagged poses
  // if possible. False if none can be reached.
  bool PlanTagPath(int _from);

  const CommonPoseResult* result_ = nullptr;
  Segment segment_;
  // Picked next segment, waiting for the current clip to reach blend_at_.
  Segment next_;
  bool pending_ = false;
  float blend_at_ = 0.f;
  BlendStats stats_;
  std::vector<int> tag_path_;
  bool following_ = false;  // Follow clip: playing a clip on.
  // The tag and threshold tag_path_ was planned for.
  std::string planned_tag_;
  float planned_threshold_ = -1.f;
  // Plays the segments' clips (the last one is the current segment's).
  ClipCrossfade crossfade_;
  std::mt19937 rng_;
};

}  // namespace hyperpose
