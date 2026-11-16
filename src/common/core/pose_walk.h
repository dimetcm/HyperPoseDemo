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

  // Go to tag (empty: the random walk). A pose counts as tagged when any of
  // its frames come from clips with the tag (CommonPose::tags). The walk
  // plans a path of transitions to the reachable tagged pose with the
  // highest tag share, along the shortest path (fewest transitions); among
  // equally short ones, the fewest clip switches (blends), then tagged clips
  // (see PlanTagPath). It plays exactly those transitions, the clip carrying
  // on where the plan follows it; reached, it plans to the next target. No tagged pose reachable: the
  // random walk. Takes effect from the next pick.
  std::string target_tag;
  // Per pose of _result, whether it counts as tagged (any share).
  std::vector<bool> TaggedPoses(const CommonPoseResult& _result) const;
  // Per pose of _result, its share of frames from clips with target_tag.
  std::vector<float> TagShares(const CommonPoseResult& _result) const;
  // Character control (motion matching over the transition graph): with a
  // goal, each pick greedily builds a path of transitions (and bridges) out
  // of the pose whose root motion follows the goal trajectory: at each step,
  // the transition whose path so far, extrapolated at its end velocity,
  // costs least. It plays the first step; the next pick searches again.
  // Takes precedence over target_tag.
  struct TrajectoryGoal {
    std::vector<float> times;  // Ahead of now (s), increasing.
    std::vector<ozz::math::Float3> positions;  // World, on the ground.
    std::vector<float> facings;  // World (radians about Y, x = sin, z = cos).
    std::vector<ozz::math::Float3> velocities;  // World, m/s.
    bool empty() const { return times.empty(); }
  };
  TrajectoryGoal goal;
  bool use_goal = false;
  // Search settings: path cost = sum over goal points of the squared
  // position error (m^2) + (facing_weight * facing error (rad))^2 +
  // (goal_velocity_weight * root velocity error (m/s))^2, plus
  // blend_weight * the pose difference (m) of each blend on the path.
  float facing_weight = 1.f;          // m per radian.
  float goal_velocity_weight = .3f;   // Seconds: 1 m/s counts as that many m.
  float blend_weight = 1.f;
  // Steps per path at most: the path ends once it covers the goal; a
  // shorter one is extrapolated at its end velocity.
  int search_depth = 12;
  // The last search: its best path's poses (from the pose searched from),
  // predicted root positions at the goal times (world), and cost.
  struct Match {
    std::vector<int> poses;
    std::vector<ozz::math::Float3> positions;
    std::vector<float> facings;  // World, as the goal's.
    float cost = -1.f;           // < 0: no search yet.
    int paths = 0;               // Candidate steps evaluated.
  };
  const Match& match() const { return match_; }
  // The character's facing (as the goal's) from its root heading: the
  // heading plus this offset, measured on the clips (their root heading
  // against the direction they move in, while they move).
  float facing_offset() const { return facing_offset_; }
  // The character root on the ground and its facing, as of the last Update.
  ozz::math::Float3 root_position() const { return crossfade_.root_position(); }
  float root_facing() const { return crossfade_.root_heading() + facing_offset_; }

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
    // A bridge (index in CommonPoseResult::bridges, -1: none): the clip
    // plays to the bridge (time), then the next segment blends into the
    // bridge's other clip and plays it on (transition is -1).
    int bridge = -1;
    // Go to tag: the step of the plan it plays (-1: none), of plan plan_id.
    int plan_step = -1;
    int plan_id = -1;
    // How it was entered: the pose difference of the blend into it (m, -1
    // for the first segment), or continued: the previous segment's clip
    // carried on into it, no blend.
    float difference = -1.f;
    bool continued = false;
    // Picked by the trajectory search (see goal).
    bool matched = false;
    // How it was picked for target_tag.
    enum class Tag {
      kOff,          // No target tag: random walk.
      kSeeking,      // From an untagged pose, on the path to a tagged one.
      kInTag,        // From a tagged pose, on to the next target.
      kUnreachable,  // No tagged pose reachable: random walk.
    };
    Tag tag = Tag::kOff;
  };
  const Segment& segment() const { return segment_; }
  // With target_tag: the planned path's poses, from where it was planned to
  // the current target (its last pose).
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
  // Fills _segment for transition _index out of _from, or for bridge _bridge
  // (with _index -1).
  void FillSegment(int _from, int _index, int _bridge, Segment* _segment) const;
  // Places the blend into next_ (blend_at_, next_.start), from _now in the
  // current segment's clip.
  void Plan(float _now);
  // Starts playing _segment's clip, crossfading over the playing ones.
  bool Begin(const Segment& _segment);
  // Frame of _features at _time (rounded, clamped).
  static int FeatureFrame(const PoseFeatures& _features, float _time);
  // Plans plan_ (and tag_path_) from _from, the playing transition _start
  // arriving there (-1: none), to the reachable tagged pose with the highest
  // tag share. Dijkstra over transitions (a clip arriving in a pose), each
  // costing 1: the shortest path. From a transition, the same clip's next
  // one (follow) or, through a small switch cost, any transition out of the
  // pose from another clip (switch, a blend): among equally short paths, the
  // fewest switches, then tagged clips. Bridges count as a transition and a
  // switch; after one, the plan follows its clip. Ties between targets: the
  // cheapest.
  // Stops at the first 100% tagged pose reached. False if none can be
  // reached.
  bool PlanTagPath(int _from, int _start);
  // Character control: the first step (a plan_ entry) of the greedy path out
  // of _from following goal (see TrajectoryGoal), the playing segment
  // carrying on until it reaches _from (unless _after_end or _jump). -1 if
  // there's no step. Fills match_.
  int SearchTrajectory(int _from, bool _after_end, bool _jump);
  // Measures facing_offset_ on result_'s clips.
  void MeasureFacingOffset();

  const CommonPoseResult* result_ = nullptr;
  Match match_;
  float facing_offset_ = 0.f;
  Segment segment_;
  // Picked next segment, waiting for the current clip to reach blend_at_.
  Segment next_;
  bool pending_ = false;
  float blend_at_ = 0.f;
  BlendStats stats_;
  // Go to tag: the plan, in order: transitions (indices in
  // CommonPoseResult::transitions) and bridges (transitions count + bridge
  // index); its id (changes with every plan); and its poses.
  std::vector<int> plan_;
  int plan_id_ = 0;
  std::vector<int> tag_path_;
  // The tag the plan was made for.
  std::string planned_tag_;
  // Plays the segments' clips (the last one is the current segment's).
  ClipCrossfade crossfade_;
  std::mt19937 rng_;
};

}  // namespace hyperpose
