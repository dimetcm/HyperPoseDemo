// HyperPoseDemo: animation playback on top of the ozz sample framework.
// Loads --skeleton/--animation (.ozz archives) when given, otherwise plays a
// built-in 3-joint arm swing.

#include "framework/application.h"
#include "framework/imgui.h"
#include "framework/renderer.h"
#include "framework/utils.h"
#include "ozz/animation/offline/animation_builder.h"
#include "ozz/animation/offline/raw_animation.h"
#include "ozz/animation/offline/raw_skeleton.h"
#include "ozz/animation/offline/skeleton_builder.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/maths/box.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/options/options.h"

OZZ_OPTIONS_DECLARE_STRING(skeleton, "Path to the skeleton (ozz archive).", "",
                           false)
OZZ_OPTIONS_DECLARE_STRING(animation, "Path to the animation (ozz archive).",
                           "", false)

namespace offline = ozz::animation::offline;

static void BuildTestRig(ozz::animation::Skeleton* _skeleton,
                         ozz::animation::Animation* _animation) {
  offline::RawSkeleton raw_skeleton;
  raw_skeleton.roots.resize(1);
  offline::RawSkeleton::Joint* joint = &raw_skeleton.roots[0];
  joint->name = "root";
  joint->transform = ozz::math::Transform::identity();
  for (const char* name : {"elbow", "tip"}) {
    joint->children.resize(1);
    joint = &joint->children[0];
    joint->name = name;
    joint->transform = ozz::math::Transform::identity();
    joint->transform.translation = ozz::math::Float3(0.f, 1.f, 0.f);
  }
  *_skeleton = std::move(*offline::SkeletonBuilder()(raw_skeleton));

  offline::RawAnimation raw_animation;
  raw_animation.duration = 2.f;
  raw_animation.tracks.resize(_skeleton->num_joints());
  const auto swing = ozz::math::Quaternion::FromAxisAngle(
      ozz::math::Float3::z_axis(), ozz::math::kPi_4);
  for (int i = 0; i < _skeleton->num_joints(); ++i) {
    auto& track = raw_animation.tracks[i];
    track.translations.push_back(
        {0.f, i == 0 ? ozz::math::Float3::zero() : ozz::math::Float3(0.f, 1.f, 0.f)});
    track.rotations.push_back({0.f, ozz::math::Quaternion::identity()});
    track.rotations.push_back({1.f, swing});
    track.rotations.push_back({2.f, ozz::math::Quaternion::identity()});
  }
  *_animation = std::move(*offline::AnimationBuilder()(raw_animation));
}

class DemoApplication : public ozz::sample::Application {
 protected:
  bool OnInitialize() override {
    if (OPTIONS_skeleton.value()[0] && OPTIONS_animation.value()[0]) {
      if (!ozz::sample::LoadSkeleton(OPTIONS_skeleton, &skeleton_) ||
          !ozz::sample::LoadAnimation(OPTIONS_animation, &animation_)) {
        return false;
      }
    } else {
      BuildTestRig(&skeleton_, &animation_);
    }
    if (skeleton_.num_joints() != animation_.num_tracks()) {
      return false;
    }

    locals_.resize(skeleton_.num_soa_joints());
    models_.resize(skeleton_.num_joints());
    context_.Resize(skeleton_.num_joints());
    return true;
  }

  bool OnUpdate(float _dt, float) override {
    controller_.Update(animation_, _dt);

    ozz::animation::SamplingJob sampling_job;
    sampling_job.animation = &animation_;
    sampling_job.context = &context_;
    sampling_job.ratio = controller_.time_ratio();
    sampling_job.output = make_span(locals_);
    if (!sampling_job.Run()) {
      return false;
    }

    ozz::animation::LocalToModelJob ltm_job;
    ltm_job.skeleton = &skeleton_;
    ltm_job.input = make_span(locals_);
    ltm_job.output = make_span(models_);
    return ltm_job.Run();
  }

  bool OnDisplay(ozz::sample::Renderer* _renderer) override {
    return _renderer->DrawPosture(skeleton_, make_span(models_),
                                  ozz::math::Float4x4::identity());
  }

  bool OnGui(ozz::sample::ImGui* _im_gui) override {
    static bool open = true;
    ozz::sample::ImGui::OpenClose oc(_im_gui, "Animation control", &open);
    if (open) {
      controller_.OnGui(animation_, _im_gui);
    }
    return true;
  }

  ozz::math::Box GetSceneBounds() const override {
    return ozz::sample::ComputePostureBounds(make_span(models_),
                                             ozz::math::Float4x4::identity());
  }

 private:
  ozz::sample::PlaybackController controller_;
  ozz::animation::Skeleton skeleton_;
  ozz::animation::Animation animation_;
  ozz::animation::SamplingJob::Context context_;
  ozz::vector<ozz::math::SoaTransform> locals_;
  ozz::vector<ozz::math::Float4x4> models_;
};

int main(int _argc, const char** _argv) {
  return DemoApplication().Run(_argc, _argv, "0.1", "HyperPoseDemo");
}
