cmake_minimum_required(VERSION 3.30)
project(HyperPoseDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ozz-animation: build only the runtime + offline libraries.
set(ozz_build_tools OFF)
set(ozz_build_fbx OFF)
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

add_subdirectory(external/ozz_framework)

add_executable(HyperPoseDemo src/main.cpp)
target_link_libraries(HyperPoseDemo PRIVATE sample_framework)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT HyperPoseDemo)
