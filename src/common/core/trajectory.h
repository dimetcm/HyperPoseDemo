#pragma once

// Desired trajectory from control input, as in motion matching: starting
// from where the character is and how it moves, its velocity and facing are
// pulled toward the desired ones by critically damped springs, predicted at
// a few times ahead; plus a short history of where the character was. All on
// the ground (y = 0); facings are angles about Y of the direction (x = sin,
// z = cos).

#include <vector>

#include "ozz/base/maths/vec_float.h"

namespace hyperpose {

struct TrajectoryPoint {
  float time;  // Relative to now (s): negative in the history.
  ozz::math::Float3 position;
  float facing;  // Radians about Y (see above).
  ozz::math::Float3 velocity = ozz::math::Float3::zero();  // m/s (future).
};

class Trajectory {
 public:
  // Settings.
  float velocity_halflife = .3f;  // How fast velocity reaches the goal (s).
  float facing_halflife = .2f;    // How fast facing does (s).
  std::vector<float> times = {1.f / 3.f, 2.f / 3.f, 1.f};  // Predicted (s).
  float history_length = 1.f;     // Kept (s).

  // Restarts from the character at _position.
  void Reset(const ozz::math::Float3& _position);

  // Advances by _dt (> 0): the character root is now at _position (its
  // velocity is estimated from it), facing _facing, and the input asks for
  // _desired_velocity (m/s, on the ground) and _desired_facing (radians, see
  // above). Then predicts the future points.
  void Update(float _dt, const ozz::math::Float3& _position, float _facing,
              const ozz::math::Float3& _desired_velocity, float _desired_facing);

  // Predicted points, at `times` ahead.
  const std::vector<TrajectoryPoint>& future() const { return future_; }
  // Past points, oldest first, the latest being now.
  const std::vector<TrajectoryPoint>& history() const { return history_; }

  const ozz::math::Float3& velocity() const { return velocity_; }
  float facing() const { return facing_; }

 private:
  bool started_ = false;
  ozz::math::Float3 position_ = ozz::math::Float3::zero();
  ozz::math::Float3 velocity_ = ozz::math::Float3::zero();  // Smoothed.
  float facing_ = 0.f;
  std::vector<TrajectoryPoint> future_;
  std::vector<TrajectoryPoint> history_;
};

// The facing of a ground direction (radians about Y, x = sin, z = cos).
float FacingOf(const ozz::math::Float3& _direction);

}  // namespace hyperpose
