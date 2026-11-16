#include "common/core/trajectory.h"

#include <algorithm>
#include <cmath>

namespace math = ozz::math;

namespace hyperpose {
namespace {

const float kPi = 3.14159265f;

// Critically damped spring damping from its half-life (time to halve the
// distance to the goal).
float Damping(float _halflife) { return 4.f * .69314718f / (_halflife + 1e-5f); }

float WrapAngle(float _angle) {
  while (_angle > kPi) {
    _angle -= 2.f * kPi;
  }
  while (_angle < -kPi) {
    _angle += 2.f * kPi;
  }
  return _angle;
}

}  // namespace

float FacingOf(const math::Float3& _direction) {
  return std::atan2(_direction.x, _direction.z);
}

void Trajectory::Reset(const math::Float3& _position) {
  started_ = true;
  position_ = math::Float3(_position.x, 0.f, _position.z);
  velocity_ = math::Float3::zero();
  history_.clear();
  future_.clear();
}

void Trajectory::Update(float _dt, const math::Float3& _position, float _facing,
                        const math::Float3& _desired_velocity, float _desired_facing) {
  const math::Float3 position(_position.x, 0.f, _position.z);
  facing_ = _facing;
  if (!started_ || _dt <= 0.f) {
    if (!started_) {
      Reset(position);
    }
    return;
  }
  // The character's velocity, smoothed (root motion is jerky per frame).
  const math::Float3 measured = (position - position_) * (1.f / _dt);
  const float smoothing = 1.f - std::exp(-_dt / .1f);
  velocity_ = velocity_ + (measured - velocity_) * smoothing;
  position_ = position;

  // History: a point per update, dropped past history_length.
  for (TrajectoryPoint& point : history_) {
    point.time -= _dt;
  }
  history_.push_back({0.f, position_, facing_});
  history_.erase(std::remove_if(history_.begin(), history_.end(),
                                [&](const TrajectoryPoint& _point) {
                                  return _point.time < -history_length;
                                }),
                 history_.end());

  // Prediction: velocity v toward goal g with a critically damped spring
  // (no initial acceleration): v(t) = g + (v0 - g) (1 + y t) e^-yt, whose
  // integral gives the position. Facing likewise, toward the desired one.
  const float y = Damping(velocity_halflife) * .5f;
  const math::Float3 j0 = velocity_ - _desired_velocity;
  const math::Float3 j1 = j0 * y;
  const float yf = Damping(facing_halflife) * .5f;
  const float facing_offset = WrapAngle(facing_ - _desired_facing);
  future_.clear();
  for (float t : times) {
    const float eydt = std::exp(-y * t);
    // x(t) = x0 + g t + j0/y + j1/y^2 - e^-yt ((j0 + j1 t)/y + j1/y^2)
    const math::Float3 offset =
        _desired_velocity * t + j0 * (1.f / y) + j1 * (1.f / (y * y)) -
        (j0 * (1.f / y) + j1 * (t / y) + j1 * (1.f / (y * y))) * eydt;
    // v(t) = g + (j0 + j1 t) e^-yt
    const math::Float3 velocity = _desired_velocity + (j0 + j1 * t) * eydt;
    const float facing =
        WrapAngle(_desired_facing + facing_offset * (1.f + yf * t) * std::exp(-yf * t));
    future_.push_back({t, position_ + offset, facing, velocity});
  }
}

}  // namespace hyperpose
