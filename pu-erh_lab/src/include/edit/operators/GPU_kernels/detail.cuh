// TODO: License header

// CUDA implementations of detail enhancement operators

#pragma once

#include <cuda_runtime.h>

#include "edit/operators/op_kernel.hpp"
#include "param.cuh"

namespace puerhlab {
namespace CUDA {
struct GPU_ClarityKernel : GPUNeighborOpTag {
  static __device__ __forceinline__ float gauss5(int dx, int dy) {
    constexpr float k[5][5] = {{1, 4, 7, 4, 1},
                               {4, 16, 26, 16, 4},
                               {7, 26, 41, 26, 7},
                               {4, 16, 26, 16, 4},
                               {1, 4, 7, 4, 1}};
    return k[dy + 2][dx + 2] * (1.0f / 273.0f);
  }

  __device__ __forceinline__ float luminance(const float4& c) const {
    // COLOR_RGB2GRAY
    return c.x * 0.114f + c.y * 0.587f + c.z * 0.299f;
  }

  __device__ void operator()(int x, int y, const float4* __restrict src, float4* __restrict dst,
                             int width, int height,
                             size_t             pitch_elems,  // pitch in float4 units
                             GPUOperatorParams& params) const {
    if (!params.clarity_enabled) {
      dst[y * pitch_elems + x] = src[y * pitch_elems + x];
      return;
    }

    float4 blur = make_float4(0, 0, 0, 0);
    for (int dy = -2; dy <= 2; ++dy) {
      int           yy  = std::min(std::max(y + dy, 0), height - 1);
      const float4* row = src + yy * pitch_elems;
      for (int dx = -2; dx <= 2; ++dx) {
        int    xx = std::min(std::max(x + dx, 0), width - 1);
        float  w  = gauss5(dx, dy);
        float4 v  = row[xx];
        blur.x += v.x * w;
        blur.y += v.y * w;
        blur.z += v.z * w;
        blur.w += v.w * w;
      }
    }

    const float4 orig = src[y * pitch_elems + x];
    float4       high = make_float4(orig.x - blur.x, orig.y - blur.y, orig.z - blur.z,
                                    orig.w);  // alpha unchanged
    // midtone mask: 1 - ((L-0.5)*2)^2
    float        lum  = luminance(orig);
    float        t    = (lum - 0.5f) * 2.0f;
    float        mask = 1.0f - t * t;

    float        w    = mask * params.clarity_offset;
    high.x *= w;
    high.y *= w;
    high.z *= w;

    float4 out               = make_float4(orig.x + high.x, orig.y + high.y, orig.z + high.z,
                                           orig.w);  // alpha unchanged
    dst[y * pitch_elems + x] = out;
  }
};

struct GPU_SharpenKernel : GPUNeighborOpTag {
  __device__ __forceinline__ float luminance(
      const float4& c) const { 
    return c.x * 0.114f + c.y * 0.587f + c.z * 0.299f;
  }

  __device__ __forceinline__ float gaussian(float dx, float dy, float sigma) const {
    const float inv2sigma2 = 0.5f / (sigma * sigma);
    return expf(-(dx * dx + dy * dy) * inv2sigma2);
  }

  __device__ void operator()(int x, int y, const float4* __restrict src, float4* __restrict dst,
                             int width, int height,
                             size_t             pitch_elems,  // pitch in float4 units
                             GPUOperatorParams& params) const {
    if (!params.sharpen_enabled || params.sharpen_offset == 0.0f || params.sharpen_radius <= 0.0f) {
      dst[y * pitch_elems + x] = src[y * pitch_elems + x];
      return;
    }

    const float sigma  = params.sharpen_radius;
    int         radius = (int)ceilf(3.0f * sigma);
    radius             = std::max(1, std::min(radius, 15));

    float4 blur        = make_float4(0, 0, 0, 0);
    float  sumW        = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
      int           yy  = std::min(std::max(y + dy, 0), height - 1);
      const float4* row = src + yy * pitch_elems;
      for (int dx = -radius; dx <= radius; ++dx) {
        int    xx = std::min(std::max(x + dx, 0), width - 1);
        float  w  = gaussian((float)dx, (float)dy, sigma);
        float4 v  = row[xx];
        blur.x += v.x * w;
        blur.y += v.y * w;
        blur.z += v.z * w;
        blur.w += v.w * w;
        sumW += w;
      }
    }
    const float invSum = 1.0f / sumW;
    blur.x *= invSum;
    blur.y *= invSum;
    blur.z *= invSum;
    blur.w *= invSum;

    const float4 orig = src[y * pitch_elems + x];
    float4       high = make_float4(orig.x - blur.x, orig.y - blur.y, orig.z - blur.z,
                                    orig.w - blur.w);

    if (params.sharpen_threshold > 0.0f) {
      float hp_gray = luminance(high);
      float mask    = (fabsf(hp_gray) > params.sharpen_threshold) ? 1.0f : 0.0f;
      high.x *= mask;
      high.y *= mask;
      high.z *= mask;
      high.w *= mask;
    }

    // USM: out = orig + high * offset
    float4 out = make_float4(
        orig.x + high.x * params.sharpen_offset, orig.y + high.y * params.sharpen_offset,
        orig.z + high.z * params.sharpen_offset, orig.w + high.w * params.sharpen_offset);
    dst[y * pitch_elems + x] = out;
  }
};
};  // namespace CUDA
};  // namespace puerhlab