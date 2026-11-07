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

# Code shared by the apps: animation helpers and retargeting.
add_library(hyperpose_common STATIC
  src/common/anim_utils.h
  src/common/anim_utils.cpp
  src/common/retarget.h
  src/common/retarget.cpp)
target_include_directories(hyperpose_common PUBLIC src)
target_link_libraries(hyperpose_common PUBLIC sample_framework)

add_executable(HyperPoseDemo src/main.cpp)
target_link_libraries(HyperPoseDemo PRIVATE hyperpose_common)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT HyperPoseDemo)

# Animation browser/preview.
add_executable(AnimPreview src/preview.cpp)
target_link_libraries(AnimPreview PRIVATE hyperpose_common)
target_compile_definitions(AnimPreview PRIVATE HYPERPOSE_ASSETS_DIR="${CMAKE_SOURCE_DIR}/Assets")

# Asset pipeline: rescales/builds raw archives from fbx2ozz (see Tools/convert_fbx.py).
add_executable(ozz_fixup Tools/ozz_fixup/main.cpp)
target_link_libraries(ozz_fixup PRIVATE ozz_animation_offline ozz_animation ozz_options ozz_base)

# fbx2mesh (copied from ozz samples, patched): extracts skinned meshes to the
# framework's mesh.ozz format.
if(TARGET ozz_animation_fbx)
  add_executable(fbx2mesh
    external/ozz_framework/tools/fbx2mesh.cc
    external/ozz_framework/framework/mesh.cc)
  target_include_directories(fbx2mesh PRIVATE external/ozz_framework)
  target_link_libraries(fbx2mesh PRIVATE ozz_animation_fbx ozz_animation ozz_options ozz_base)
endif()
