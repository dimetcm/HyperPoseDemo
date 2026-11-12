#include "common/core/point_field.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <random>
#include <unordered_map>

#include "common/core/anim_utils.h"
#include "framework/mesh.h"
#include "ozz/geometry/runtime/skinning_job.h"

namespace math = ozz::math;

namespace hyperpose {

void BuildRestMesh(const ozz::vector<ozz::sample::Mesh>& _meshes, TriMesh* _mesh) {
  _mesh->positions.clear();
  _mesh->indices.clear();
  for (const ozz::sample::Mesh& mesh : _meshes) {
    // Triangle indices address the mesh parts' vertices, concatenated.
    const uint32_t base = static_cast<uint32_t>(_mesh->positions.size());
    for (const auto& part : mesh.parts) {
      for (size_t i = 0; i + 2 < part.positions.size(); i += 3) {
        _mesh->positions.emplace_back(part.positions[i], part.positions[i + 1],
                                      part.positions[i + 2]);
      }
    }
    for (uint16_t index : mesh.triangle_indices) {
      _mesh->indices.push_back(base + index);
    }
  }
}

bool PoseMesh(const ozz::vector<ozz::sample::Mesh>& _meshes,
              ozz::span<const math::Float4x4> _joint_model_transforms,
              TriMesh* _mesh) {
  ozz::vector<math::Float4x4> skinning_matrices;
  size_t vertex = 0;
  for (const ozz::sample::Mesh& mesh : _meshes) {
    skinning_matrices.resize(mesh.joint_remaps.size());
    ComputeSkinningMatrices(mesh, _joint_model_transforms,
                            make_span(skinning_matrices));

    for (const auto& part : mesh.parts) {
      const int vertex_count = part.vertex_count();
      if (vertex_count == 0) {
        continue;
      }
      if (vertex + vertex_count > _mesh->positions.size()) {
        return false;  // Not built from these meshes.
      }
      const int influences = part.influences_count();
      ozz::geometry::SkinningJob job;
      job.vertex_count = vertex_count;
      job.influences_count = influences;
      job.joint_matrices = make_span(skinning_matrices);
      job.joint_indices = make_span(part.joint_indices);
      job.joint_indices_stride = sizeof(uint16_t) * influences;
      if (influences > 1) {
        job.joint_weights = make_span(part.joint_weights);
        job.joint_weights_stride = sizeof(float) * (influences - 1);
      }
      job.in_positions = make_span(part.positions);
      job.in_positions_stride = sizeof(float) * 3;
      float* out = &_mesh->positions[vertex].x;
      job.out_positions = {out, out + vertex_count * 3};
      job.out_positions_stride = sizeof(math::Float3);
      if (!job.Run()) {
        return false;
      }
      vertex += vertex_count;
    }
  }
  return true;
}

namespace {

void Triangle(const TriMesh& _mesh, uint32_t _triangle, math::Float3* _a,
              math::Float3* _b, math::Float3* _c) {
  *_a = _mesh.positions[_mesh.indices[_triangle * 3 + 0]];
  *_b = _mesh.positions[_mesh.indices[_triangle * 3 + 1]];
  *_c = _mesh.positions[_mesh.indices[_triangle * 3 + 2]];
}

float TriangleArea(const TriMesh& _mesh, uint32_t _triangle) {
  math::Float3 a, b, c;
  Triangle(_mesh, _triangle, &a, &b, &c);
  return .5f * Length(Cross(b - a, c - a));
}

double SurfaceArea(const TriMesh& _mesh) {
  double area = 0.;
  for (int t = 0; t < _mesh.num_triangles(); ++t) {
    area += TriangleArea(_mesh, t);
  }
  return area;
}

// Uniform grid over points, for fixed radius neighbor queries.
class PointGrid {
 public:
  PointGrid(ozz::span<const math::Float3> _points, float _cell)
      : points_(_points), inv_cell_(1.f / _cell) {
    for (size_t i = 0; i < _points.size(); ++i) {
      cells_[Key(Cell(_points[i]))].push_back(static_cast<int>(i));
    }
  }

  // Distance from _p to the nearest point (excluding index _exclude), searching
  // rings of cells outward until no closer point can exist.
  float Nearest(const math::Float3& _p, int _exclude = -1) const {
    const auto c = Cell(_p);
    const float cell = 1.f / inv_cell_;
    float best = std::numeric_limits<float>::max();
    for (int ring = 0; ring < 1024; ++ring) {
      for (int x = c[0] - ring; x <= c[0] + ring; ++x) {
        for (int y = c[1] - ring; y <= c[1] + ring; ++y) {
          for (int z = c[2] - ring; z <= c[2] + ring; ++z) {
            if (std::max({std::abs(x - c[0]), std::abs(y - c[1]),
                          std::abs(z - c[2])}) != ring) {
              continue;  // Inner rings were already visited.
            }
            const auto it = cells_.find(Key({x, y, z}));
            if (it == cells_.end()) continue;
            for (int j : it->second) {
              if (j != _exclude) best = std::min(best, Length(points_[j] - _p));
            }
          }
        }
      }
      if (best <= ring * cell) {
        break;
      }
    }
    return best;
  }

  // Calls _fct(index, distance) for points closer than _radius (<= cell size)
  // to point _i, excluding itself.
  template <typename _Fct>
  void ForNeighbors(int _i, float _radius, _Fct _fct) const {
    const auto c = Cell(points_[_i]);
    for (int x = c[0] - 1; x <= c[0] + 1; ++x) {
      for (int y = c[1] - 1; y <= c[1] + 1; ++y) {
        for (int z = c[2] - 1; z <= c[2] + 1; ++z) {
          const auto it = cells_.find(Key({x, y, z}));
          if (it == cells_.end()) continue;
          for (int j : it->second) {
            if (j == _i) continue;
            const float d = Length(points_[j] - points_[_i]);
            if (d < _radius) _fct(j, d);
          }
        }
      }
    }
  }

 private:
  std::array<int, 3> Cell(const math::Float3& _p) const {
    return {static_cast<int>(std::floor(_p.x * inv_cell_)),
            static_cast<int>(std::floor(_p.y * inv_cell_)),
            static_cast<int>(std::floor(_p.z * inv_cell_))};
  }
  static int64_t Key(const std::array<int, 3>& _c) {
    return (int64_t(_c[0]) * 73856093) ^ (int64_t(_c[1]) * 19349663) ^
           (int64_t(_c[2]) * 83492791);
  }

  ozz::span<const math::Float3> points_;
  float inv_cell_;
  std::unordered_map<int64_t, std::vector<int>> cells_;
};

// Grid cell size giving a few points per cell, from the bounding box of points
// assumed to lie on a surface.
float SurfaceCellSize(ozz::span<const math::Float3> _points) {
  math::Float3 lo(std::numeric_limits<float>::max()), hi(-lo.x);
  for (const auto& p : _points) {
    lo = Min(lo, p);
    hi = Max(hi, p);
  }
  const math::Float3 size = hi - lo;
  const float area = 2.f * (size.x * size.y + size.y * size.z + size.z * size.x);
  return std::max(std::sqrt(area / std::max<size_t>(_points.size(), 1)) * 2.f,
                  1e-4f);
}

}  // namespace

ozz::vector<SurfacePoint> PointFieldGenerator::SampleRandom(const TriMesh& _mesh,
                                                            int _count,
                                                            uint32_t _seed) {
  ozz::vector<SurfacePoint> points;
  const int num_triangles = _mesh.num_triangles();
  if (num_triangles == 0 || _count <= 0) {
    return points;
  }

  // Picks triangles proportionally to their area.
  std::vector<double> cumulative_areas(num_triangles);
  double total = 0.;
  for (int t = 0; t < num_triangles; ++t) {
    total += TriangleArea(_mesh, t);
    cumulative_areas[t] = total;
  }
  if (total <= 0.) {
    return points;
  }

  std::mt19937 rng(_seed);
  std::uniform_real_distribution<double> area_dist(0., total);
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  points.reserve(_count);
  for (int i = 0; i < _count; ++i) {
    const auto it = std::upper_bound(cumulative_areas.begin(),
                                     cumulative_areas.end(), area_dist(rng));
    const uint32_t triangle = static_cast<uint32_t>(
        std::min<ptrdiff_t>(it - cumulative_areas.begin(), num_triangles - 1));
    // Uniform point in triangle (square root parameterization).
    const float r1 = std::sqrt(unit(rng)), r2 = unit(rng);
    points.push_back({triangle, {1.f - r1, r1 * (1.f - r2), r1 * r2}});
  }
  return points;
}

ozz::vector<SurfacePoint> PointFieldGenerator::SampleUniform(const TriMesh& _mesh,
                                                             int _count,
                                                             uint32_t _seed) {
  // Weighted sample elimination, Yuksel 2015, "Sample Elimination for
  // Generating Poisson Disk Sample Sets".
  constexpr int kCandidatesRatio = 5;
  constexpr float kAlpha = 8.f, kBeta = .65f, kGamma = 1.5f;

  const ozz::vector<SurfacePoint> candidates =
      SampleRandom(_mesh, _count * kCandidatesRatio, _seed);
  const int num_candidates = static_cast<int>(candidates.size());
  if (num_candidates <= _count) {
    return candidates;
  }
  ozz::vector<math::Float3> positions;
  Evaluate(_mesh, make_span(candidates), &positions);

  // Max Poisson disk radius for _count points on a surface of that area, and
  // the lower bound that limits clustering penalty of very close candidates.
  const float area = static_cast<float>(SurfaceArea(_mesh));
  const float r_max = std::sqrt(area / (2.f * std::sqrt(3.f) * _count));
  const float r_min =
      r_max * (1.f - std::pow(float(_count) / num_candidates, kGamma)) * kBeta;
  const float radius = 2.f * r_max;
  auto weight = [&](float _d) {
    return std::pow(1.f - std::max(_d, r_min) / radius, kAlpha);
  };

  // Neighbor lists and initial weights.
  const PointGrid grid({positions.data(), positions.size()}, radius);
  std::vector<std::vector<std::pair<int, float>>> neighbors(num_candidates);
  std::vector<float> weights(num_candidates, 0.f);
  for (int i = 0; i < num_candidates; ++i) {
    grid.ForNeighbors(i, radius, [&](int _j, float _d) {
      const float w = weight(_d);
      neighbors[i].emplace_back(_j, w);
      weights[i] += w;
    });
  }

  // Repeatedly eliminates the most crowded sample. Heap entries are lazily
  // invalidated: an entry is stale if its weight doesn't match anymore.
  std::priority_queue<std::pair<float, int>> heap;
  for (int i = 0; i < num_candidates; ++i) {
    heap.emplace(weights[i], i);
  }
  std::vector<bool> removed(num_candidates, false);
  int remaining = num_candidates;
  while (remaining > _count && !heap.empty()) {
    const auto [w, i] = heap.top();
    heap.pop();
    if (removed[i] || w != weights[i]) {
      continue;
    }
    removed[i] = true;
    --remaining;
    for (const auto& [j, wij] : neighbors[i]) {
      if (!removed[j]) {
        weights[j] -= wij;
        heap.emplace(weights[j], j);
      }
    }
  }

  ozz::vector<SurfacePoint> points;
  points.reserve(_count);
  for (int i = 0; i < num_candidates; ++i) {
    if (!removed[i]) {
      points.push_back(candidates[i]);
    }
  }
  return points;
}

void PointFieldGenerator::Evaluate(const TriMesh& _mesh,
                                   ozz::span<const SurfacePoint> _points,
                                   ozz::vector<math::Float3>* _positions) {
  _positions->resize(_points.size());
  for (size_t i = 0; i < _points.size(); ++i) {
    const SurfacePoint& p = _points[i];
    math::Float3 a, b, c;
    Triangle(_mesh, p.triangle, &a, &b, &c);
    (*_positions)[i] =
        a * p.barycentrics[0] + b * p.barycentrics[1] + c * p.barycentrics[2];
  }
}

PointFieldGenerator::Spacing PointFieldGenerator::MeasureSpacing(
    ozz::span<const math::Float3> _points) {
  Spacing spacing = {0.f, 0.f, 0.f};
  if (_points.size() < 2) {
    return spacing;
  }
  const PointGrid grid(_points, SurfaceCellSize(_points));
  double sum = 0., sum_sq = 0.;
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < _points.size(); ++i) {
    const float d = grid.Nearest(_points[i], static_cast<int>(i));
    sum += d;
    sum_sq += double(d) * d;
    min = std::min(min, d);
  }
  const double n = static_cast<double>(_points.size());
  const double mean = sum / n;
  spacing.mean = static_cast<float>(mean);
  spacing.cv = static_cast<float>(
      mean > 0. ? std::sqrt(std::max(0., sum_sq / n - mean * mean)) / mean : 0.);
  spacing.min = min;
  return spacing;
}

float PointFieldGenerator::MeanNearestDistance(
    ozz::span<const math::Float3> _a, ozz::span<const math::Float3> _b) {
  if (_a.empty() || _b.empty()) {
    return 0.f;
  }
  const PointGrid grid(_b, SurfaceCellSize(_b));
  double sum = 0.;
  for (const auto& p : _a) {
    sum += grid.Nearest(p);
  }
  return static_cast<float>(sum / _a.size());
}

PointFieldGenerator::Sampler PointFieldGenerator::FindSampler(const char* _name) {
  if (std::strcmp(_name, "random") == 0) return &SampleRandom;
  if (std::strcmp(_name, "uniform") == 0) return &SampleUniform;
  return nullptr;
}

}  // namespace hyperpose
