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

# Dear ImGui (GLFW + OpenGL3 backends), drawn on top of the ozz gui.
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.93.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)
add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

target_link_libraries(sample_framework
  PUBLIC ozz_geometry ozz_animation_offline ozz_animation ozz_options ozz_base imgui
  PRIVATE glfw OpenGL::GL)
target_compile_definitions(sample_framework PRIVATE GL_SILENCE_DEPRECATION)
target_include_directories(sample_framework
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${ozz_SOURCE_DIR}/extern/glfw/include) # bundled GL/glext.h
