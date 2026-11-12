cmake_minimum_required(VERSION 3.30)
project(HyperPoseDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Patched FindFbx (supports FBX SDK 2020.3.11 layout); takes precedence over ozz's.
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ozz-animation: runtime + offline libraries, plus fbx2ozz (needs FBX SDK).
set(ozz_build_tools ON)
set(ozz_build_fbx ON)
set(ozz_build_gltf OFF)
set(ozz_build_samples OFF)
set(ozz_build_howtos OFF)
set(ozz_build_tests OFF)
set(ozz_build_postfix OFF)
set(ozz_build_msvc_rt_dll ON) # match CMake's default /MD runtime

FetchContent_Declare(ozz
  GIT_REPOSITORY https://github.com/guillaumeblanc/ozz-animation.git
  GIT_TAG 0.17.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ozz)

# dump2ozz only regenerates ozz's reference docs; its post-build step fails on
# the VS generator, so keep it out of the default build.
if(TARGET dump2ozz)
  set_target_properties(dump2ozz PROPERTIES EXCLUDE_FROM_ALL ON EXCLUDE_FROM_DEFAULT_BUILD ON)
endif()

add_subdirectory(external/ozz_framework)

# Shared core: animation helpers, retargeting, pose matching and clustering,
# the pose walk and projects (no UI).
add_library(hyperpose_core STATIC
  src/common/core/anim_utils.h
  src/common/core/anim_utils.cpp
  src/common/core/retarget.h
  src/common/core/retarget.cpp
  src/common/core/point_field.h
  src/common/core/point_field.cpp
  src/common/core/pose_match.h
  src/common/core/pose_match.cpp
  src/common/core/pose_cluster.h
  src/common/core/pose_cluster.cpp
  src/common/core/clip_player.h
  src/common/core/clip_player.cpp
  src/common/core/common_pose.h
  src/common/core/common_pose.cpp
  src/common/core/clip_crossfade.h
  src/common/core/clip_crossfade.cpp
  src/common/core/pose_walk.h
  src/common/core/pose_walk.cpp
  src/common/core/pose_project.h
  src/common/core/pose_project.cpp)
target_include_directories(hyperpose_core PUBLIC src)
target_link_libraries(hyperpose_core PUBLIC sample_framework)

# Shared UI and drawing (Dear ImGui, the framework renderer), on top of core.
add_library(hyperpose_gui STATIC
  src/common/gui/clip_library.h
  src/common/gui/clip_library.cpp
  src/common/gui/character_view.h
  src/common/gui/character_view.cpp
  src/common/gui/pose_gallery.h
  src/common/gui/pose_gallery.cpp
  src/common/gui/clip_preview.h
  src/common/gui/clip_preview.cpp
  src/common/gui/transition_graph.h
  src/common/gui/transition_graph.cpp
  src/common/gui/project_bar.h
  src/common/gui/project_bar.cpp)
target_link_libraries(hyperpose_gui PUBLIC hyperpose_core)

# Animation browser/preview.
add_executable(AnimPreview src/preview.cpp)
target_link_libraries(AnimPreview PRIVATE hyperpose_gui)
target_compile_definitions(AnimPreview PRIVATE HYPERPOSE_ASSETS_DIR="${CMAKE_SOURCE_DIR}/Assets")

# Minimal Pose Difference Blend: best transition between two clips.
add_executable(PoseBlend src/pose_blend.cpp)
target_link_libraries(PoseBlend PRIVATE hyperpose_gui)
target_compile_definitions(PoseBlend PRIVATE HYPERPOSE_ASSETS_DIR="${CMAKE_SOURCE_DIR}/Assets")

# Principal Poses Walk: clusters the poses of a set of clips and walks them.
add_executable(PrincipalPosesWalk src/principal_poses_walk.cpp)
target_link_libraries(PrincipalPosesWalk PRIVATE hyperpose_gui)
target_compile_definitions(PrincipalPosesWalk PRIVATE
  HYPERPOSE_ASSETS_DIR="${CMAKE_SOURCE_DIR}/Assets"
  HYPERPOSE_PROJECTS_DIR="${CMAKE_SOURCE_DIR}/Projects")
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT PrincipalPosesWalk)

# Asset pipeline: rescales/builds raw archives from fbx2ozz (see Tools/convert_fbx.py).
add_executable(ozz_fixup Tools/ozz_fixup/main.cpp)
target_link_libraries(ozz_fixup PRIVATE ozz_animation_offline ozz_animation ozz_options ozz_base)

# Point field generation (see src/common/core/point_field.h).
add_executable(pointfield_gen Tools/pointfield_gen/main.cpp)
target_link_libraries(pointfield_gen PRIVATE hyperpose_core)
add_executable(pointfield_stats Tools/pointfield_stats/main.cpp)
target_link_libraries(pointfield_stats PRIVATE hyperpose_core)

# Common pose cluster count scan (see Tools/pose_k_scan/main.cpp).
add_executable(pose_k_scan Tools/pose_k_scan/main.cpp)
target_link_libraries(pose_k_scan PRIVATE hyperpose_core)
target_compile_definitions(pose_k_scan PRIVATE HYPERPOSE_ASSETS_DIR="${CMAKE_SOURCE_DIR}/Assets")

# fbx2mesh (copied from ozz samples, patched): extracts skinned meshes to the
# framework's mesh.ozz format.
if(TARGET ozz_animation_fbx)
  add_executable(fbx2mesh
    external/ozz_framework/tools/fbx2mesh.cc
    external/ozz_framework/framework/mesh.cc)
  target_include_directories(fbx2mesh PRIVATE external/ozz_framework)
  target_link_libraries(fbx2mesh PRIVATE ozz_animation_fbx ozz_animation ozz_options ozz_base)
endif()
