// pose_k_scan: finds common poses of a project for a range of cluster counts
// (k) and prints measures to choose k from (see ClusterCountMeasure):
//
//   rms         RMS pose distance of frames to their common pose (cm): falls
//               as k grows; the elbow is where adding clusters stops paying.
//   silhouette  -1..1, higher: better separated poses.
//   entropy     of each pose's next pose shares (bits): lower, a more
//               predictable transition graph.
//
// The settings (stride, seed, velocity weight) are the project's.
//
// Example:
//   pose_k_scan --project=Projects/medium_walk.hpproject --k_min=4 --k_max=64

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "common/core/clip_player.h"
#include "common/core/common_pose.h"
#include "common/core/pose_project.h"
#include "framework/utils.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/options/options.h"

namespace fs = std::filesystem;

OZZ_OPTIONS_DECLARE_STRING(project, "Project file (.hpproject).", "", true)
OZZ_OPTIONS_DECLARE_STRING(root, "Clip root the project's paths are relative to.",
                           HYPERPOSE_ASSETS_DIR "/ozz", false)
OZZ_OPTIONS_DECLARE_INT(k_min, "Smallest cluster count.", 2, false)
OZZ_OPTIONS_DECLARE_INT(k_max, "Largest cluster count.", 64, false)

int main(int _argc, const char** _argv) {
  if (ozz::options::ParseCommandLine(_argc, _argv, "1.0",
                                     "Measures common poses for a range of k.") !=
      ozz::options::kSuccess) {
    return EXIT_FAILURE;
  }
  hyperpose::PoseProject project;
  std::string error;
  if (!hyperpose::LoadPoseProject(OPTIONS_project.value(), OPTIONS_root.value(), &project,
                                  &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return EXIT_FAILURE;
  }
  if (project.clips.empty()) {
    std::fprintf(stderr, "Empty pose set.\n");
    return EXIT_FAILURE;
  }
  ozz::animation::Skeleton skeleton;
  const fs::path skeleton_path =
      hyperpose::FindClipSkeleton(project.clips.front().path, OPTIONS_root.value());
  if (skeleton_path.empty() ||
      !ozz::sample::LoadSkeleton(skeleton_path.string().c_str(), &skeleton)) {
    std::fprintf(stderr, "No skeleton for %s\n", project.clips.front().path.string().c_str());
    return EXIT_FAILURE;
  }
  const hyperpose::CommonPoseSettings& settings = project.settings;
  std::printf("%s: %d clips, stride %d, seed %d, velocity weight %.2fs\n",
              fs::path(OPTIONS_project.value()).stem().string().c_str(),
              static_cast<int>(project.clips.size()), settings.stride, settings.seed,
              settings.velocity_weight);

  const std::vector<hyperpose::ClusterCountMeasure> measures =
      hyperpose::MeasureClusterCounts(project.clips, &skeleton, settings, OPTIONS_k_min.value(),
                                      OPTIONS_k_max.value(), true, &error);
  if (measures.empty()) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return EXIT_FAILURE;
  }
  for (const hyperpose::ClusterCountMeasure& m : measures) {
    std::printf("k %2d  rms %5.2f cm  silhouette %6.3f  entropy %4.2f bits\n", m.clusters,
                m.rms * 100.f, m.silhouette, m.entropy);
  }
  const auto best_silhouette = std::max_element(
      measures.begin(), measures.end(),
      [](const auto& _a, const auto& _b) { return _a.silhouette < _b.silhouette; });
  std::printf("\nelbow of rms: k %d\nbest silhouette: k %d (%.3f)\n",
              hyperpose::ElbowClusterCount(measures), best_silhouette->clusters,
              best_silhouette->silhouette);
  return EXIT_SUCCESS;
}
