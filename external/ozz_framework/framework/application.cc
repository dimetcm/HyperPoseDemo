//----------------------------------------------------------------------------//
//                                                                            //
// ozz-animation is hosted at http://github.com/guillaumeblanc/ozz-animation  //
// and distributed under the MIT License (MIT).                               //
//                                                                            //
// Copyright (c) Guillaume Blanc                                              //
//                                                                            //
// Permission is hereby granted, free of charge, to any person obtaining a    //
// copy of this software and associated documentation files (the "Software"), //
// to deal in the Software without restriction, including without limitation  //
// the rights to use, copy, modify, merge, publish, distribute, sublicense,   //
// and/or sell copies of the Software, and to permit persons to whom the      //
// Software is furnished to do so, subject to the following conditions:       //
//                                                                            //
// The above copyright notice and this permission notice shall be included in //
// all copies or substantial portions of the Software.                        //
//                                                                            //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    //
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    //
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        //
// DEALINGS IN THE SOFTWARE.                                                  //
//                                                                            //
//----------------------------------------------------------------------------//

#define OZZ_INCLUDE_PRIVATE_HEADER  // Allows to include private headers.

#include "framework/application.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#ifdef __APPLE__
#include <unistd.h>
#endif  // __APPLE__

#if __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif  // __EMSCRIPTEN__

// [HyperPoseDemo change] Dear ImGui. Angle brackets so framework/imgui.h (the
// ozz gui) next to this file isn't picked instead.
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "framework/image.h"
#include "framework/internal/camera.h"
#include "framework/internal/imgui_impl.h"
#include "framework/internal/renderer_impl.h"
#include "framework/internal/shooter.h"
#include "framework/profile.h"
#include "framework/renderer.h"
#include "ozz/base/io/stream.h"
#include "ozz/base/log.h"
#include "ozz/base/maths/box.h"
#include "ozz/base/memory/allocator.h"
#include "ozz/options/options.h"

OZZ_OPTIONS_DECLARE_INT(
    max_idle_loops,
    "The maximum number of idle loops the sample application can perform."
    " Application automatically exit when this number of loops is reached."
    " A negative value disables this feature.",
    -1, false);

OZZ_OPTIONS_DECLARE_BOOL(render, "Enables sample redering.", true, false);

namespace {
// Screen resolution presets.
const ozz::sample::Resolution resolution_presets[] = {
    {640, 360},   {640, 480},  {800, 450},  {800, 600},   {1024, 576},
    {1024, 768},  {1280, 720}, {1280, 800}, {1280, 960},  {1280, 1024},
    {1400, 1050}, {1440, 900}, {1600, 900}, {1600, 1200}, {1680, 1050},
    {1920, 1080}, {1920, 1200}};
const int kNumPresets = OZZ_ARRAY_SIZE(resolution_presets);
}  // namespace

// Check resolution argument is within 0 - kNumPresets
static bool ResolutionCheck(const ozz::options::Option& _option,
                            int /*_argc*/) {
  const ozz::options::IntOption& option =
      static_cast<const ozz::options::IntOption&>(_option);
  return option >= -1 && option < kNumPresets;
}

// [HyperPoseDemo change] Defaults to -1: 80% of the monitor work area.
OZZ_OPTIONS_DECLARE_INT_FN(
    resolution, "Resolution index (0 to 16), or -1 to fit the monitor.", -1,
    false, &ResolutionCheck);

namespace ozz {
namespace sample {

Application::Application()
    : fps_(New<Record>(128)),
      update_time_(New<Record>(128)),
      render_time_(New<Record>(128)),
      window_size_(resolution_presets[0]),
      framebuffer_size_(resolution_presets[0]),
      content_scale_(1.f) {
#ifndef NDEBUG
  // Assert presets are correctly sorted.
  for (int i = 1; i < kNumPresets; ++i) {
    const Resolution& preset_m1 = resolution_presets[i - 1];
    const Resolution& preset = resolution_presets[i];
    assert(preset.width > preset_m1.width || preset.height > preset_m1.height);
  }
#endif  //  NDEBUG
}

Application::~Application() {}

int Application::Run(int _argc, const char** _argv, const char* _version,
                     const char* _title) {
  // Starting application
  log::Out() << "Starting sample \"" << _title << "\" version \"" << _version
             << "\"" << std::endl;
  log::Out() << "Ozz libraries were built with \""
             << math::SimdImplementationName() << "\" SIMD math implementation."
             << std::endl;

  // Parse command line arguments.
  const char* usage =
      "Ozz animation sample. See README.md file for more details.";
  ozz::options::ParseResult result =
      ozz::options::ParseCommandLine(_argc, _argv, _version, usage);
  if (result != ozz::options::kSuccess) {
    exit_ = true;
    return result == ozz::options::kExitSuccess ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  // Fetch initial resolution.
  // [HyperPoseDemo change] -1 is resolved from the monitor once glfw is up.
  window_size_ = resolution_presets[OPTIONS_resolution >= 0 ? OPTIONS_resolution : 5];

#ifdef __APPLE__
  // On OSX, when run from Finder, working path is the root path. This does not
  // allow to load resources from relative path.
  // The workaround is to change the working directory to application directory.
  // The proper solution would probably be to use bundles and load data from
  // resource folder.
  chdir(ozz::options::ParsedExecutablePath().c_str());
#endif  // __APPLE__

  // Initialize help.
  ParseReadme();

  // Open an OpenGL window
  bool success = true;
  if (OPTIONS_render) {
    auto error_callback = [](int error, const char* description) {
      ozz::log::Err() << "GLFW error " << error << ": " << description
                      << std::endl;
    };
    glfwSetErrorCallback(error_callback);

    // Initialize GLFW
    if (!glfwInit()) {
      return EXIT_FAILURE;
    }

    // Setup GL context.
    const int gl_version_major = 3, gl_version_minor = 2;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl_version_major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl_version_minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif  // NDEBUG

#ifdef __EMSCRIPTEN__
    emscripten::glfw3::SetNextWindowCanvasSelector("sample_canvas");
#endif  // __EMSCRIPTEN__

    // [HyperPoseDemo change] Fit the window to the primary monitor work area.
    int work_x = 0, work_y = 0, work_w = 0, work_h = 0;
    if (OPTIONS_resolution < 0) {
      if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) {
        glfwGetMonitorWorkarea(monitor, &work_x, &work_y, &work_w, &work_h);
      }
      if (work_w > 0 && work_h > 0) {
        window_size_ = {static_cast<int>(work_w * .8f),
                        static_cast<int>(work_h * .8f)};
      }
    }

    // Initializes rendering before looping.
    window_ = glfwCreateWindow(window_size_.width, window_size_.height, _title,
                               nullptr, nullptr);
    if (window_ && work_w > 0 && work_h > 0) {
      glfwSetWindowPos(window_, work_x + (work_w - window_size_.width) / 2,
                       work_y + (work_h - window_size_.height) / 2);
    }
    if (!window_) {
      log::Err() << "Failed to open OpenGL window. Required OpenGL version is "
                 << gl_version_major << "." << gl_version_minor << "."
                 << std::endl;
      success = false;
    } else {
      // So appllication can access window from static callbacks.
      glfwSetWindowUserPointer(window_, this);
      glfwMakeContextCurrent(window_);

      log::Out() << "Successfully opened OpenGL window version \""
                 << glGetString(GL_VERSION) << "\"." << std::endl;

      // Allocates and initializes camera
      camera_ = make_unique<internal::Camera>();
      math::Float3 camera_center;
      math::Float2 camera_angles;
      float distance;
      if (GetCameraInitialSetup(&camera_center, &camera_angles, &distance)) {
        camera_->Reset(camera_center, camera_angles, distance);
      }

      // Allocates and initializes renderer.
      renderer_ = make_unique<internal::RendererImpl>(camera_.get());
      success = renderer_->Initialize();

      if (success) {
        shooter_ = make_unique<internal::Shooter>();
        im_gui_ = make_unique<internal::ImGuiImpl>();

        // Setup the window and installs callbacks.
        glfwSwapInterval(vertical_sync_ ? swap_interval_ : 0);
        glfwSetWindowSizeCallback(window_, ResizeCbk);
        glfwSetFramebufferSizeCallback(window_, ResizeCbk);
        glfwSetWindowCloseCallback(window_, CloseCbk);

#ifdef __EMSCRIPTEN__
        // Makes the canvas resizable to the size of its div container
        emscripten::glfw3::MakeCanvasResizable(window_, "sample_container");
#endif  // __EMSCRIPTEN__

        // GLFW3 does not fire an initial resize event on window creation.
        // Trigger it manually to initialize viewport and camera projection
        // matrices.
        Resize();

        // [HyperPoseDemo change] Dear ImGui setup, scaled to monitor DPI.
        // IMGUI_CHECKVERSION() expanded: its unqualified ImGui:: would resolve
        // to ozz::sample::ImGui in this namespace.
        ::ImGui::DebugCheckVersionAndDataLayout(
            IMGUI_VERSION, sizeof(ImGuiIO), sizeof(ImGuiStyle), sizeof(ImVec2),
            sizeof(ImVec4), sizeof(ImDrawVert), sizeof(ImDrawIdx));
        ::ImGui::CreateContext();
        ::ImGui::GetIO().IniFilename = nullptr;
        ::ImGui::StyleColorsDark();
        float dpi_scale;
        glfwGetWindowContentScale(window_, &dpi_scale, nullptr);
        ::ImGui::GetStyle().ScaleAllSizes(dpi_scale);
        ::ImGui::GetStyle().FontScaleDpi = dpi_scale;
        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init("#version 150");

        // Loop the sample.
        success = Loop();

        // Ending sample, releases resources.
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ::ImGui::DestroyContext();
        shooter_.reset();
        im_gui_.reset();
      }
      renderer_.reset();
      camera_.reset();
    }

    // Closes window and terminates GLFW.
    glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();

  } else {
    // Loops without any rendering initialization.
    success = Loop();
  }

  // Notifies that an error occurred.
  if (!success) {
    log::Err() << "An error occurred during sample execution." << std::endl;
  }

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Helper function to detecte key pressed and released.
template <int _Key>
bool KeyPressed(GLFWwindow* _window) {
  if (!_window) {
    return false;
  }
  static int previous_key = GLFW_RELEASE;
  const int key = glfwGetKey(_window, _Key);
  const bool pressed = previous_key == GLFW_PRESS && key == GLFW_RELEASE;
  previous_key = key;
  return pressed;
}

Application::LoopStatus Application::OneLoop(int _loops) {
  Profiler profile(fps_.get());  // Profiles frame.

  // Tests for a manual exit request.
  if (exit_ || KeyPressed<GLFW_KEY_ESCAPE>(window_)) {
    return kBreak;
  }

  // Test for an exit request.
  if (OPTIONS_max_idle_loops >= 0 && _loops > OPTIONS_max_idle_loops) {
    return kBreak;
  }

#ifndef __EMSCRIPTEN__
  // Don't overload the cpu if the window is not active.
  if (OPTIONS_render && !glfwGetWindowAttrib(window_, GLFW_FOCUSED)) {
    glfwWaitEvents();  // Wait...

    // Reset last update time in order to stop the time while the app isn't
    // active.
    last_idle_time_ = glfwGetTime();

    return kContinue;  // ...but don't do anything.
  }
#endif  // __EMSCRIPTEN__

  // Enable/disable help on F1 key.
  show_help_ = show_help_ ^ KeyPressed<GLFW_KEY_F1>(window_);

  // Capture screenshot or video.
  capture_screenshot_ = KeyPressed<'S'>(window_);
  capture_video_ = capture_video_ ^ KeyPressed<'V'>(window_);

  // Do the main loop.
  if (!Idle(_loops == 0)) {
    return kBreakFailure;
  }

  // Skips display if "no_render" option is enabled.
  if (OPTIONS_render) {
    if (!Display()) {
      return kBreakFailure;
    }
  }

  return kContinue;
}

void OneLoopCbk(void* _arg) {
  Application* app = reinterpret_cast<Application*>(_arg);
  static int loops = 0;
  app->OneLoop(loops++);
}

bool Application::Loop() {
  // Initialize sample.
  bool success = OnInitialize();

// Emscripten requires to manage the main loop on their own, as browsers don't
// like infinite blocking functions.
#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop_arg(OneLoopCbk, this, 0, 1);
#else   // __EMSCRIPTEN__
  // Loops.
  for (int loops = 0; success; ++loops) {
    const LoopStatus status = OneLoop(loops);
    success = status != kBreakFailure;
    if (status != kContinue) {
      break;
    }
  }
#endif  // __EMSCRIPTEN__

  // De-initialize sample, even in case of initialization failure.
  OnDestroy();

  return success;
}

bool Application::Display() {
  assert(OPTIONS_render);

  bool success = true;

  {  // Profiles rendering excluding GUI.
    Profiler profile(render_time_.get());

    GL(ClearDepth(1.f));
    GL(ClearColor(.4f, .42f, .38f, 1.f));
    GL(Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    // Setup default states
    GL(Enable(GL_CULL_FACE));
    GL(CullFace(GL_BACK));
    GL(Enable(GL_DEPTH_TEST));
    GL(DepthMask(GL_TRUE));
    GL(DepthFunc(GL_LEQUAL));

    // Bind 3D camera matrices.
    camera_->Bind3D();

    // Forwards display event to the inheriting application.
    if (success) {
      success = OnDisplay(renderer_.get());
    }
  }  // Ends profiling.

  // Renders grid and axes at the end as they are transparent.
  if (show_grid_) {
    renderer_->DrawGrid(20, 1.f);
  }
  if (show_axes_) {
    renderer_->DrawAxes(ozz::math::Float4x4::identity());
  }

  // Bind 2D camera matrices.
  camera_->Bind2D();

  // Forwards gui event to the inheriting application.
  if (success) {
    success = Gui();
  }

  // [HyperPoseDemo change] Dear ImGui, on top of the ozz gui.
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ::ImGui::NewFrame();
  if (success) {
    success = OnDearImGui();
  }
  // [HyperPoseDemo change] Mouse wheel zooms the camera, unless Dear ImGui uses
  // the mouse (e.g. scrolling a list). Read within the frame, as EndFrame resets
  // the wheel; the camera view is rebuilt on next update.
  {
    const ImGuiIO& io = ::ImGui::GetIO();
    if (io.MouseWheel != 0.f && !io.WantCaptureMouse) {
      camera_->Zoom(io.MouseWheel);
    }
  }
  ::ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(::ImGui::GetDrawData());

  // Capture back buffer.
  if (capture_screenshot_ || capture_video_) {
    shooter_->Capture(GL_BACK);
    capture_screenshot_ = false;
  }

  // Swaps current window and polls events (GLFW3 no longer does this
  // implicitly on swap, unlike GLFW2's GLFW_AUTO_POLL_EVENTS).
  glfwSwapBuffers(window_);
  glfwPollEvents();

  return success;
}

bool Application::Idle(bool _first_frame) {
  // Early out if displaying help.
  if (show_help_) {
    last_idle_time_ = glfwGetTime();
    return true;
  }

  // Compute elapsed time since last idle, and delta time.
  float delta;
  double time = glfwGetTime();
  if (_first_frame ||  // Don't take into account time spent initializing.
      time == 0.) {    // Means glfw isn't initialized (rendering's disabled).
    delta = 1.f / 60.f;
  } else {
    delta = static_cast<float>(time - last_idle_time_);
  }
  last_idle_time_ = time;

  // Update dt, can be scaled, fixed, freezed...
  float update_delta;
  if (freeze_) {
    update_delta = 0.f;
  } else {
    if (fix_update_rate) {
      update_delta = time_factor_ / fixed_update_rate;
    } else {
      update_delta = delta * time_factor_;
    }
  }

  // Increment current application time
  time_ += update_delta;

  // Forwards update event to the inheriting application.
  bool update_result;
  {  // Profiles update scope.
    Profiler profile(update_time_.get());
    update_result = OnUpdate(update_delta, time_);
  }

  // Updates screen shooter object.
  if (shooter_) {
    shooter_->Update();
  }

  // Update camera model-view matrix.
  if (camera_) {
    const auto scene_bounds = GetSceneBounds();

    math::Float4x4 camera_transform;
    if (GetCameraOverride(&camera_transform)) {
      camera_->Update(window_, camera_transform, scene_bounds, delta,
                      _first_frame);
    } else {
      camera_->Update(window_, scene_bounds, delta, _first_frame);
    }
  }

  return update_result;
}

bool Application::Gui() {
  bool success = true;
  const float kFormWidth = 200.f;
  const float kHelpMargin = 16.f;

  // Finds gui area.
  const float kGuiMargin = 2.f;
  ozz::math::RectInt window_rect{0, 0, window_size_.width, window_size_.height};
  // [HyperPoseDemo change] Left edge of ozz forms, after app reserved space.
  const float left = gui_left_inset_ + kGuiMargin;

  // Fills ImGui's input structure.
  internal::ImGuiImpl::Inputs input;
  double mouse_dx, mouse_dy;
  glfwGetCursorPos(window_, &mouse_dx, &mouse_dy);
  input.mouse_x = static_cast<int>(mouse_dx);
  const int mouse_y = static_cast<int>(mouse_dy);
  input.mouse_y = window_rect.height - mouse_y;
  input.lmb_pressed =
      glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

  // Starts frame
  im_gui_->BeginFrame(input, window_rect, renderer_.get());

  // Downcast to public imgui.
  ImGui* im_gui = im_gui_.get();

  if (window_rect.width - gui_left_inset_ > (kGuiMargin + kFormWidth) * 2.f) {
    // Do floating gui.
    if (!show_help_) {
      success = OnFloatingGui(im_gui);
    }

    // Help gui.
    {
      math::RectFloat rect{left, kGuiMargin,
                           window_rect.width - left - kGuiMargin,
                           window_rect.height - kGuiMargin * 2.f};
      // Doesn't constrain form is it's opened, so it covers all screen.
      ImGui::Form form(im_gui, "Show help", rect, &show_help_, !show_help_);
      if (show_help_) {
        im_gui->DoLabel(help_.c_str(), ImGui::kLeft, false);
      }
    }

    if (!show_help_) {
      // Do framework gui.
      if (success) {
        static bool open = true;
        math::RectFloat rect{
            left, kGuiMargin, kFormWidth,
            window_rect.height - kGuiMargin * 2.f - kHelpMargin};
        ImGui::Form form(im_gui, "Framework", rect, &open, true);
        if (open) {
          success = FrameworkGui();
        }
      }

      // Do sample gui.
      if (success) {
        static bool open = true;
        math::RectFloat rect{window_rect.width - kFormWidth - kGuiMargin,
                             kGuiMargin, kFormWidth,
                             window_rect.height - kGuiMargin * 2 - kHelpMargin};
        ImGui::Form form(im_gui, "Sample", rect, &open, true);
        if (open) {
          // Forwards event to the inherited application.
          success = OnGui(im_gui);
        }
      }
    }
  }

  // Ends frame
  im_gui_->EndFrame();

  return success;
}

bool Application::FrameworkGui() {
  char label[64];
  // Downcast to public imgui.
  ImGui* im_gui = im_gui_.get();
  {  // Render statistics
    static bool open = true;
    ImGui::OpenClose stat_oc(im_gui, "Statistics", &open);
    if (open) {
      {  // FPS
        Record::Statistics statistics = fps_->GetStatistics();
        std::snprintf(label, sizeof(label), "FPS: %.0f",
                      statistics.mean == 0.f ? 0.f : 1000.f / statistics.mean);
        static bool fps_open = false;
        ImGui::OpenClose stats(im_gui, label, &fps_open);
        if (fps_open) {
          std::snprintf(label, sizeof(label), "Frame: %.2f ms",
                        statistics.mean);
          im_gui->DoGraph(label, 0.f, statistics.max, statistics.latest,
                          fps_->cursor(), fps_->record_begin(),
                          fps_->record_end());
        }
      }
      {  // Update time
        Record::Statistics statistics = update_time_->GetStatistics();
        std::snprintf(label, sizeof(label), "Update: %.2f ms", statistics.mean);
        static bool update_open = true;  // This is the most relevant for ozz.
        ImGui::OpenClose stats(im_gui, label, &update_open);
        if (update_open) {
          im_gui->DoGraph(nullptr, 0.f, statistics.max, statistics.latest,
                          update_time_->cursor(), update_time_->record_begin(),
                          update_time_->record_end());
        }
      }
      {  // Render time
        Record::Statistics statistics = render_time_->GetStatistics();
        std::snprintf(label, sizeof(label), "Render: %.2f ms", statistics.mean);
        static bool render_open = false;
        ImGui::OpenClose stats(im_gui, label, &render_open);
        if (render_open) {
          im_gui->DoGraph(nullptr, 0.f, statistics.max, statistics.latest,
                          render_time_->cursor(), render_time_->record_begin(),
                          render_time_->record_end());
        }
      }
    }
  }

  {  // Time control
    static bool open = false;
    ImGui::OpenClose stats(im_gui, "Time control", &open);
    if (open) {
      im_gui->DoButton("Freeze", true, &freeze_);
      im_gui->DoCheckBox("Fix update rate", &fix_update_rate, true);
      if (!fix_update_rate) {
        std::snprintf(label, sizeof(label), "Time factor: %.2f", time_factor_);
        im_gui->DoSlider(label, -5.f, 5.f, &time_factor_);
        if (im_gui->DoButton("Reset time factor", time_factor_ != 1.f)) {
          time_factor_ = 1.f;
        }
      } else {
        std::snprintf(label, sizeof(label), "Update rate: %.0f fps",
                      fixed_update_rate);
        im_gui->DoSlider(label, 1.f, 200.f, &fixed_update_rate, .5f, true);
        if (im_gui->DoButton("Reset update rate", fixed_update_rate != 60.f)) {
          fixed_update_rate = 60.f;
        }
      }
    }
  }

  {  // Rendering options
    static bool open = false;
    ImGui::OpenClose options(im_gui, "Options", &open);
    if (open) {
      // Multi-sampling.
      static bool fsaa_available =
          glfwGetWindowAttrib(window_, GLFW_SAMPLES) != 0;
      static bool fsaa_enabled = fsaa_available;
      if (im_gui->DoCheckBox("Anti-aliasing", &fsaa_enabled, fsaa_available)) {
        if (fsaa_enabled) {
          GL(Enable(GL_MULTISAMPLE));
        } else {
          GL(Disable(GL_MULTISAMPLE));
        }
      }
      // Vertical sync & swap interval
      bool changed = im_gui->DoCheckBox("Vertical sync", &vertical_sync_);
      std::snprintf(label, sizeof(label), "Swap interval: %d", swap_interval_);
      changed |=
          im_gui->DoSlider(label, 1, 4, &swap_interval_, 1.f, vertical_sync_);
      if (changed) {
        glfwSwapInterval(vertical_sync_ ? swap_interval_ : 0);
      }

      im_gui->DoCheckBox("Show grid", &show_grid_, true);
      im_gui->DoCheckBox("Show axes", &show_axes_, true);
    }

    // Searches for matching resolution settings.
    int preset_lookup = 0;
    for (; preset_lookup < kNumPresets - 1; ++preset_lookup) {
      const Resolution& preset = resolution_presets[preset_lookup];
      if (preset.width > window_size_.width) {
        break;
      } else if (preset.width == window_size_.width) {
        if (preset.height >= window_size_.height) {
          break;
        }
      }
    }

    std::snprintf(label, sizeof(label), "Resolution: %dx%d", window_size_.width,
                  window_size_.height);
    if (im_gui->DoSlider(label, 0, kNumPresets - 1, &preset_lookup)) {
      // Resolution changed.
      window_size_ = resolution_presets[preset_lookup];
      glfwSetWindowSize(window_, window_size_.width, window_size_.height);
    }
  }

  {  // Capture
    static bool open = false;
    ImGui::OpenClose controls(im_gui, "Capture", &open);
    if (open) {
      im_gui->DoButton("Capture video", true, &capture_video_);
      capture_screenshot_ |= im_gui->DoButton(
          "Capture screenshot", !capture_video_, &capture_screenshot_);
    }
  }

  {  // Controls
    static bool open = false;
    ImGui::OpenClose controls(im_gui, "Camera controls", &open);
    if (open) {
      camera_->OnGui(im_gui);
    }
  }
  return true;
}

bool Application::OnInitialize() { return true; }

void Application::OnDestroy() {}

bool Application::OnUpdate(float, float) { return true; }

bool Application::OnGui(ImGui*) { return true; }

bool Application::OnFloatingGui(ImGui*) { return true; }

bool Application::OnDearImGui() { return true; }

bool Application::OnDisplay(Renderer*) { return true; }

bool Application::GetCameraInitialSetup(math::Float3*, math::Float2*,
                                        float*) const {
  return false;
}

// Default implementation doesn't override camera location.
bool Application::GetCameraOverride(math::Float4x4*) const { return false; }

math::Box Application::GetSceneBounds() const { return {}; }

math::Float2 Application::WorldToScreen(const math::Float3& _world) const {
  const math::SimdFloat4 ndc =
      (camera_->projection() * camera_->view()) *
      math::simd_float4::Load(_world.x, _world.y, _world.z, 1.f);

  const math::SimdFloat4 resolution = math::simd_float4::FromInt(
      math::simd_int4::Load(window_size_.width, window_size_.height, 0, 0));
  const ozz::math::SimdFloat4 screen =
      resolution * ((ndc / math::SplatW(ndc)) + math::simd_float4::one()) /
      math::simd_float4::Load1(2.f);
  math::Float2 ret;
  math::Store2PtrU(screen, &ret.x);
  return ret;
}

/*static*/ void Application::ResizeCbk(GLFWwindow* _window, int, int) {
  Application* app =
      reinterpret_cast<Application*>(glfwGetWindowUserPointer(_window));
  app->Resize();
}

void Application::Resize() {
  Application* app =
      reinterpret_cast<Application*>(glfwGetWindowUserPointer(window_));

  glfwGetWindowSize(window_, &window_size_.width, &window_size_.height);

  glfwGetFramebufferSize(window_, &framebuffer_size_.width,
                         &framebuffer_size_.height);

  // GUI is laid out in window coordinates, so the 2D projection needs the
  // framebuffer/window ratio (2 on macOS retina, 1 on Windows), not the monitor
  // content scale (e.g. 1.5 at 150% on Windows, which pushed the GUI offscreen).
  // [HyperPoseDemo change]
  content_scale_.x = window_size_.width > 0
                         ? float(framebuffer_size_.width) / window_size_.width
                         : 1.f;
  content_scale_.y = window_size_.height > 0
                         ? float(framebuffer_size_.height) / window_size_.height
                         : 1.f;

  /*
  ozz::log::Out() << "Resize :" << window_size_.width << "x"
                  << window_size_.height << ", " << framebuffer_size_.width
                  << "x" << framebuffer_size_.height << ", " <<
  content_scale_.x
                  << "x" << content_scale_.y << std::endl;
  */

  // Uses the full viewport.
  GL(Viewport(0, 0, GLsizei(framebuffer_size_.width),
              GLsizei(framebuffer_size_.height)));

  // Forwards screen size to camera and shooter.
  app->camera_->Resize(framebuffer_size_.width, framebuffer_size_.height,
                       content_scale_);
  app->shooter_->Resize(framebuffer_size_.width, framebuffer_size_.height);
}

/*static*/ void Application::CloseCbk(GLFWwindow* _window) {
  Application* app =
      reinterpret_cast<Application*>(glfwGetWindowUserPointer(_window));

  app->exit_ = true;
  // Prevent GLFW from closing the window — we handle it at the end of the
  // loop.
  glfwSetWindowShouldClose(_window, GLFW_FALSE);
}

void Application::ParseReadme() {
  const char* error_message = "Unable to find README.md help file.";

  // Get README file, opens as binary to avoid conversions.
  ozz::io::File file("README.md", "rb");
  if (!file.opened()) {
    help_ = error_message;
    return;
  }

  // Allocate enough space to store the whole file.
  const size_t read_length = file.Size();
  ozz::memory::Allocator* allocator = ozz::memory::default_allocator();
  char* content = reinterpret_cast<char*>(allocator->Allocate(read_length, 4));

  // Read the content
  if (file.Read(content, read_length) == read_length) {
    help_ = ozz::string(content, content + read_length);
  } else {
    help_ = error_message;
  }

  // Deallocate temporary buffer;
  allocator->Deallocate(content);
}
}  // namespace sample
}  // namespace ozz
