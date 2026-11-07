// pointfield_stats: prints metrics of point field files, to compare methods
// and samplers.
//
//   pointfield_stats a.hppf            spacing evenness over the clip
//   pointfield_stats a.hppf b.hppf     also compares a and b frame by frame

#include <cstdio>
#include <cstring>
#include <fstream>

#include "common/point_field.h"

using hyperpose::PointFieldGenerator;
using hyperpose::PointFieldHeader;
using ozz::math::Float3;

namespace {

struct PointFieldFile {
  std::ifstream stream;
  PointFieldHeader header;

  bool Open(const char* _path) {
    stream.open(_path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
        std::memcmp(header.magic, "HPPF", 4) != 0) {
      std::fprintf(stderr, "%s isn't a point field file.\n", _path);
      return false;
    }
    return true;
  }

  bool ReadFrame(uint32_t _frame, ozz::vector<Float3>* _points) {
    _points->resize(header.num_points);
    stream.seekg(sizeof(header) + std::streamoff(_frame) * header.num_points *
                                      sizeof(Float3));
    return bool(stream.read(reinterpret_cast<char*>(_points->data()),
                            header.num_points * sizeof(Float3)));
  }
};

}  // namespace

int main(int _argc, const char** _argv) {
  if (_argc < 2 || _argc > 3) {
    std::fprintf(stderr, "Usage: pointfield_stats a.hppf [b.hppf]\n");
    return EXIT_FAILURE;
  }
  PointFieldFile files[2];
  const int num_files = _argc - 1;
  for (int f = 0; f < num_files; ++f) {
    if (!files[f].Open(_argv[f + 1])) {
      return EXIT_FAILURE;
    }
  }

  // Evaluates 5 frames spread over the clip.
  const uint32_t num_frames = files[0].header.num_frames;
  uint32_t frames[5];
  for (int i = 0; i < 5; ++i) {
    frames[i] = num_frames > 1 ? (num_frames - 1) * i / 4 : 0;
  }

  ozz::vector<Float3> points[2];
  for (int f = 0; f < num_files; ++f) {
    const PointFieldHeader& h = files[f].header;
    std::printf("%s: %u frames x %u points @ %.0f fps\n", _argv[f + 1],
                h.num_frames, h.num_points, h.frame_rate);
    std::printf("  frame   nn mean   nn cv   nn min\n");
    for (uint32_t frame : frames) {
      if (frame >= h.num_frames || !files[f].ReadFrame(frame, &points[f])) {
        continue;
      }
      const auto s = PointFieldGenerator::MeasureSpacing(make_span(points[f]));
      std::printf("  %5u  %6.1fmm   %.3f  %6.2fmm\n", frame, s.mean * 1000.f,
                  s.cv, s.min * 1000.f);
    }
  }

  if (num_files == 2) {
    std::printf("Distance between fields (mean nearest point):\n");
    std::printf("  frame     a->b      b->a\n");
    for (uint32_t frame : frames) {
      if (frame >= files[1].header.num_frames || !files[0].ReadFrame(frame, &points[0]) ||
          !files[1].ReadFrame(frame, &points[1])) {
        continue;
      }
      std::printf("  %5u  %6.1fmm  %6.1fmm\n", frame,
                  PointFieldGenerator::MeanNearestDistance(make_span(points[0]),
                                                           make_span(points[1])) * 1000.f,
                  PointFieldGenerator::MeanNearestDistance(make_span(points[1]),
                                                           make_span(points[0])) * 1000.f);
    }
  }
  return EXIT_SUCCESS;
}
