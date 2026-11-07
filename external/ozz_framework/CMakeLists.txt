# ozz-animation sample framework (copied from ozz 0.17.0 samples/framework).
# Provides window, camera, immediate-mode GUI and skeleton/mesh rendering.

file(GLOB_RECURSE FRAMEWORK_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/framework/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/framework/*.cc)
add_library(sample_framework STATIC ${FRAMEWORK_SOURCES})

find_package(OpenGL REQUIRED)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${ozz_SOURCE_DIR}/extern/glfw ${CMAKE_CURRENT_BINARY_DIR}/glfw)

target_link_libraries(sample_framework
  PUBLIC ozz_geometry ozz_animation_offline ozz_animation ozz_options ozz_base
  PRIVATE glfw OpenGL::GL)
target_compile_definitions(sample_framework PRIVATE GL_SILENCE_DEPRECATION)
target_include_directories(sample_framework
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${ozz_SOURCE_DIR}/extern/glfw/include) # bundled GL/glext.h
