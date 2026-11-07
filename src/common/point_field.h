#pragma once

// Point fields: points distributed on a skinned mesh surface, per animation
// frame. Samplers are static functions of PointFieldGenerator so algorithms can
// be swapped and compared easily.
//
// Point field file (.hppf), little endian:
//   PointFieldHeader, then num_frames * num_points * float3 (x, y, z).

#include <cstdint>

#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/vec_float.h"
#include "ozz/base/span.h"

namespace ozz::sample {
struct Mesh;
}

namespace hyperpose {

// Triangle mesh snapshot: all mesh parts flattened, positions at some pose.
struct TriMesh {
  ozz::vector<ozz::math::Float3> positions;
  ozz::vector<uint32_t> indices;  // 3 per triangle.
  int num_triangles() const { return static_cast<int>(indices.size() / 3); }
};

// A point on a mesh surface. Stays on the surface whatever the pose.
struct SurfacePoint {
  uint32_t triangle;
  float barycentrics[3];
};

// Flattens meshes (all parts) into a rest pose (bind pose) TriMesh.
void BuildRestMesh(const ozz::vector<ozz::sample::Mesh>& _meshes, TriMesh* _mesh);

// Skins meshes with joint model-space matrices into _mesh positions. _mesh must
// come from BuildRestMesh with the same meshes (only positions are updated).
bool PoseMesh(const ozz::vector<ozz::sample::Mesh>& _meshes,
              ozz::span<const ozz::math::Float4x4> _joint_model_transforms,
              TriMesh* _mesh);

class PointFieldGenerator {
 public:
  // Sampler signature: _count points on _mesh, deterministic for a _seed.
  using Sampler = ozz::vector<SurfacePoint> (*)(const TriMesh& _mesh,
                                                int _count, uint32_t _seed);

  // Area weighted random (white noise) sampling. Uniform density, but clumpy.
  static ozz::vector<SurfacePoint> SampleRandom(const TriMesh& _mesh,
                                                int _count, uint32_t _seed);

  // Evenly spaced (blue noise) sampling, by weighted sample elimination
  // (Yuksel 2015) of 5x random candidates.
  static ozz::vector<SurfacePoint> SampleUniform(const TriMesh& _mesh,
                                                 int _count, uint32_t _seed);

  // Positions of surface points on _mesh.
  static void Evaluate(const TriMesh& _mesh,
                       ozz::span<const SurfacePoint> _points,
                       ozz::vector<ozz::math::Float3>* _positions);

  // Returns the sampler with that name ("random", "uniform"), or nullptr.
  static Sampler FindSampler(const char* _name);

  // Metrics, to compare methods and samplers.

  // Nearest neighbor distances statistics. cv (std / mean) measures evenness:
  // ~0.5 for random sampling, lower is more even.
  struct Spacing {
    float mean, cv, min;
  };
  static Spacing MeasureSpacing(ozz::span<const ozz::math::Float3> _points);

  // Mean distance from each point of _a to its nearest point in _b.
  static float MeanNearestDistance(ozz::span<const ozz::math::Float3> _a,
                                   ozz::span<const ozz::math::Float3> _b);
};

struct PointFieldHeader {
  char magic[4] = {'H', 'P', 'P', 'F'};
  uint32_t version = 1;
  uint32_t num_frames = 0;
  uint32_t num_points = 0;  // Per frame.
  float frame_rate = 0.f;
};

}  // namespace hyperpose
