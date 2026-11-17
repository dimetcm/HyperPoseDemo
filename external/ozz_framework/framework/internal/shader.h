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

#ifndef OZZ_SAMPLES_FRAMEWORK_INTERNAL_SHADER_H_
#define OZZ_SAMPLES_FRAMEWORK_INTERNAL_SHADER_H_

#include "ozz/base/memory/unique_ptr.h"
#include "renderer_impl.h"

namespace ozz {
namespace math {
struct Float4x4;
}
namespace sample {
namespace internal {

// Declares a shader program.
class Shader {
 public:
  // Construct a fixed function pipeline shader. Use Shader::Build to specify
  // shader sources.
  Shader();

  // Destruct a shader.
  virtual ~Shader();

  // Returns the shader program that can be bound to the OpenGL context.
  GLuint program() const { return program_; }

  // Request an uniform location and pushes it to the uniform stack.
  // The uniform location is then accessible thought uniform().
  bool BindUniform(const char* _semantic);

  // Get an uniform location from the stack at index _index.
  GLint uniform(int _index) const { return uniforms_[_index]; }

  // Request an attribute location and pushes it to the uniform stack.
  // The varying location is then accessible thought attrib().
  bool FindAttrib(const char* _semantic);

  // Get an varying location from the stack at index _index.
  GLint attrib(int _index) const { return attribs_[_index]; }

  // Unblind shader.
  virtual void Unbind();

 protected:
  // Constructs a shader from _vertex and _fragment glsl sources.
  // Mutliple source files can be specified using the *count argument.
  bool BuildFromSource(int _vertex_count, const char** _vertex,
                       int _fragment_count, const char** _fragment);

 private:
  // Unbind all attribs from GL.
  void UnbindAttribs();

  // Shader program
  GLuint program_;

  // Vertex and fragment shaders
  GLuint vertex_;
  GLuint fragment_;

  // Uniform locations, in the order they were requested.
  ozz::vector<GLint> uniforms_;

  // Varying locations, in the order they were requested.
  ozz::vector<GLint> attribs_;
};

class ImmediatePCShader : public Shader {
 public:
  ImmediatePCShader() {}
  virtual ~ImmediatePCShader() {}

  // Constructs the shader.
  // Returns nullptr if shader compilation failed or a valid Shader pointer on
  // success. The shader must then be deleted using default allocator Delete
  // function.
  static ozz::unique_ptr<ImmediatePCShader> Build();

  // Binds the shader.
  void Bind(const math::Float4x4& _model, const math::Float4x4& _view_proj,
            GLsizei _pos_stride, GLsizei _pos_offset, GLsizei _color_stride,
            GLsizei _color_offset);
};

class ImmediatePTCShader : public Shader {
 public:
  ImmediatePTCShader() {}
  virtual ~ImmediatePTCShader() {}

  // Constructs the shader.
  // Returns nullptr if shader compilation failed or a valid Shader pointer on
  // success. The shader must then be deleted using default allocator Delete
  // function.
  static ozz::unique_ptr<ImmediatePTCShader> Build();

  // Binds the shader.
  void Bind(const math::Float4x4& _model, const math::Float4x4& _view_proj,
            GLsizei _pos_stride, GLsizei _pos_offset, GLsizei _tex_stride,
            GLsizei _tex_offset, GLsizei _color_stride, GLsizei _color_offset);
};

class PointsShader : public Shader {
 public:
  PointsShader() {}
  virtual ~PointsShader() {}

  // Constructs the shader.
  // Returns nullptr if shader compilation failed or a valid Shader pointer on
  // success. The shader must then be deleted using default allocator Delete
  // function.
  static ozz::unique_ptr<PointsShader> Build();

  // Binds the shader.
  struct GenericAttrib {
    GLint color;
    GLint size;
  };
  GenericAttrib Bind(const math::Float4x4& _model,
                     const math::Float4x4& _view_proj, GLsizei _pos_stride,
                     GLsizei _pos_offset, GLsizei _color_stride,
                     GLsizei _color_offset, GLsizei _size_stride,
                     GLsizei _size_offset, bool _screen_space);
};

class SkeletonShader : public Shader {
 public:
  SkeletonShader() {}
  virtual ~SkeletonShader() {}

  // Binds the shader.
  void Bind(const math::Float4x4& _model, const math::Float4x4& _view_proj,
            GLsizei _pos_stride, GLsizei _pos_offset, GLsizei _normal_stride,
            GLsizei _normal_offset, GLsizei _color_stride,
            GLsizei _color_offset);

  // Get an attribute location for the join, in cased of instanced rendering.
  GLint joint_instanced_attrib() const { return attrib(3); }

  // Get an uniform location for the join, in cased of non-instanced rendering.
  GLint joint_uniform() const { return uniform(2); }
};

class JointShader : public SkeletonShader {
 public:
  JointShader() {}
  virtual ~JointShader() {}

  // Constructs the shader.
  // Returns nullptr if shader compilation failed or a valid Shader pointer on
  // success. The shader must then be deleted using default allocator Delete
  // function.
  static ozz::unique_ptr<JointShader> Build();
};

class BoneShader : public SkeletonShader {
 public:
  BoneShader() {}
  virtual ~BoneShader() {}

  // Constructs the shader.
  // Returns nullptr if shader compilation failed or a valid Shader pointer on
  // success. The shader must then be deleted using default allocator Delete
  // function.
  static ozz::unique_ptr<BoneShader> Build();
};

class AmbientShader : public Shader {
 public:
  AmbientShader() {}
  virtual ~AmbientShader() {}

  // Constructs the shader.
  // Returns nullptr if shader compilation failed or a valid Shader pointer on
  // success. The shader must then be deleted using default allocator Delete
  // function.
  static ozz::unique_ptr<AmbientShader> Build();

  // Binds the shader.
  void Bind(const math::Float4x4& _model, const math::Float4x4& _view_proj,
            GLsizei _pos_stride, GLsizei _pos_offset, GLsizei _normal_stride,
            GLsizei _normal_offset, GLsizei _color_stride,
            GLsizei _color_offset, bool _color_float);

 protected:
  bool InternalBuild(int _vertex_count, const char** _vertex,
                     int _fragment_count, const char** _fragment);
};

// Ambient shader skinning vertices on the GPU (see Renderer::CreateGpuMesh):
// up to 8 joint influences per vertex, kMaxJoints joint matrices.
class SkinnedAmbientShader : public Shader {
 public:
  enum { kMaxJoints = 64 };

  SkinnedAmbientShader() {}
  virtual ~SkinnedAmbientShader() {}

  static ozz::unique_ptr<SkinnedAmbientShader> Build();

  // Binds the shader for the array buffer bound, interleaved: position (3
  // floats), normal (3 floats), joint indices (8 bytes), weights (8 floats).
  // _color (rgba) tints the mesh.
  void Bind(const math::Float4x4& _model, const math::Float4x4& _view_proj,
            span<const math::Float4x4> _joints, const float _color[4]);
};

// Instanced GPU skinning (HyperPoseDemo addition): the mesh layout of
// SkinnedAmbientShader, drawn once per instance, the instances read from an
// RGBA32F texture bound to unit 0, kTextureWidth texels a row: per instance,
// its skinning matrices in world space (4 texels each, the columns), then its
// color (1 texel).
class SkinnedInstancedShader : public Shader {
 public:
  enum { kTextureWidth = 1024 };

  SkinnedInstancedShader() {}
  virtual ~SkinnedInstancedShader() {}

  static ozz::unique_ptr<SkinnedInstancedShader> Build();

  // Binds the shader for the array buffer bound, _joints matrices an
  // instance.
  void Bind(const math::Float4x4& _view_proj, int _joints);
};

class AmbientShaderInstanced : public Shader {
 public:
  AmbientShaderInstanced() {}
  virtual ~AmbientShaderInstanced() {}

  // Constructs the shader.
  // Returns nullptr if shader compilation failed or a valid Shader pointer on
  // success. The shader must then be deleted using default allocator Delete
  // function.
  static ozz::unique_ptr<AmbientShaderInstanced> Build();

  // Binds the shader.
  void Bind(GLsizei _models_offset, const math::Float4x4& _view_proj,
            GLsizei _pos_stride, GLsizei _pos_offset, GLsizei _normal_stride,
            GLsizei _normal_offset, GLsizei _color_stride,
            GLsizei _color_offset, bool _color_float);

  virtual void Unbind();
};

class AmbientTexturedShader : public AmbientShader {
 public:
  // Constructs the shader.
  // Returns nullptr if shader compilation failed or a valid Shader pointer on
  // success. The shader must then be deleted using default allocator Delete
  // function.
  static ozz::unique_ptr<AmbientTexturedShader> Build();

  // Binds the shader.
  void Bind(const math::Float4x4& _model, const math::Float4x4& _view_proj,
            GLsizei _pos_stride, GLsizei _pos_offset, GLsizei _normal_stride,
            GLsizei _normal_offset, GLsizei _color_stride,
            GLsizei _color_offset, bool _color_float, GLsizei _uv_stride,
            GLsizei _uv_offset);
};
/*
class AmbientTexturedShaderInstanced : public AmbientShaderInstanced {
public:

  // Constructs the shader.
  // Returns nullptr if shader compilation failed or a valid Shader pointer on
  // success. The shader must then be deleted using default allocator Delete
  // function.
  static AmbientTexturedShaderInstanced* Build();

  // Binds the shader.
  void Bind(GLsizei _models_offset,
            const math::Float4x4& _view_proj,
            GLsizei _pos_stride, GLsizei _pos_offset,
            GLsizei _normal_stride, GLsizei _normal_offset,
            GLsizei _color_stride, GLsizei _color_offset,
            GLsizei _uv_stride, GLsizei _uv_offset);
};
*/
}  // namespace internal
}  // namespace sample
}  // namespace ozz
#endif  // OZZ_SAMPLES_FRAMEWORK_INTERNAL_SHADER_H_
