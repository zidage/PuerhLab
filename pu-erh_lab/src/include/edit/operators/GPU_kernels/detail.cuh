//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// CUDA implementations of detail enhancement operators

#pragma once

#include <cuda_runtime.h>

#include "edit/operators/op_kernel.hpp"
#include "param.cuh"

namespace puerhlab {
namespace CUDA {

GPU_FUNC float detail_luminance(const float4& c) {
  // Match the CPU COLOR_BGR2GRAY coefficients.
  return c.x * 0.114f + c.y * 0.587f + c.z * 0.299f;
}

GPU_FUNC float4 detail_read_clamped(const float4* __restrict src, int x, int y, int width,
                                    int height, size_t pitch_elems) {
  const int clamped_x = min(max(x, 0), width - 1);
  const int clamped_y = min(max(y, 0), height - 1);
  return src[static_cast<size_t>(clamped_y) * pitch_elems + static_cast<size_t>(clamped_x)];
}

GPU_FUNC float4 detail_blur_horizontal(int x, int y, const float4* __restrict src, int width,
                                       int height, size_t pitch_elems, int tap_count,
                                       const float* __restrict weights) {
  if (tap_count <= 0) {
    return detail_read_clamped(src, x, y, width, height, pitch_elems);
  }

  const float4 center = detail_read_clamped(src, x, y, width, height, pitch_elems);
  float4       blur   = make_float4(center.x * weights[0], center.y * weights[0],
                                    center.z * weights[0], center.w * weights[0]);
  for (int tap = 1; tap < tap_count; ++tap) {
    const float  w = weights[tap];
    const float4 a = detail_read_clamped(src, x + tap, y, width, height, pitch_elems);
    const float4 b = detail_read_clamped(src, x - tap, y, width, height, pitch_elems);
    blur.x += (a.x + b.x) * w;
    blur.y += (a.y + b.y) * w;
    blur.z += (a.z + b.z) * w;
    blur.w += (a.w + b.w) * w;
  }
  return blur;
}

GPU_FUNC float4 detail_blur_vertical(int x, int y, const float4* __restrict src, int width,
                                     int height, size_t pitch_elems, int tap_count,
                                     const float* __restrict weights) {
  if (tap_count <= 0) {
    return detail_read_clamped(src, x, y, width, height, pitch_elems);
  }

  const float4 center = detail_read_clamped(src, x, y, width, height, pitch_elems);
  float4       blur   = make_float4(center.x * weights[0], center.y * weights[0],
                                    center.z * weights[0], center.w * weights[0]);
  for (int tap = 1; tap < tap_count; ++tap) {
    const float  w = weights[tap];
    const float4 a = detail_read_clamped(src, x, y + tap, width, height, pitch_elems);
    const float4 b = detail_read_clamped(src, x, y - tap, width, height, pitch_elems);
    blur.x += (a.x + b.x) * w;
    blur.y += (a.y + b.y) * w;
    blur.z += (a.z + b.z) * w;
    blur.w += (a.w + b.w) * w;
  }
  return blur;
}

struct GPU_ClarityBlurHorizontalKernel : GPUNeighborOpTag {
  __device__ __forceinline__ void operator()(int x, int y, const float4* __restrict src,
                                             float4* __restrict dst, int width, int height,
                                             size_t pitch_elems,
                                             GPUOperatorParams& params) const {
    const size_t offset = static_cast<size_t>(y) * pitch_elems + static_cast<size_t>(x);
    if (!params.clarity_enabled_ || params.clarity_offset_ == 0.0f || params.clarity_radius_ <= 0.0f ||
        params.clarity_gaussian_tap_count_ <= 0) {
      dst[offset] = src[offset];
      return;
    }

    dst[offset] = detail_blur_horizontal(x, y, src, width, height, pitch_elems,
                                         params.clarity_gaussian_tap_count_,
                                         params.clarity_gaussian_weights_);
  }
};

struct GPU_ClarityApplyVerticalKernel : GPUNeighborOpTag {
  __device__ __forceinline__ void operator()(int x, int y, const float4* __restrict src,
                                             float4* __restrict dst, int width, int height,
                                             size_t pitch_elems,
                                             GPUOperatorParams& params) const {
    const size_t offset = static_cast<size_t>(y) * pitch_elems + static_cast<size_t>(x);
    if (!params.clarity_enabled_ || params.clarity_offset_ == 0.0f || params.clarity_radius_ <= 0.0f ||
        params.clarity_gaussian_tap_count_ <= 0) {
      dst[offset] = src[offset];
      return;
    }

    const float4 orig = dst[offset];
    const float4 blur = detail_blur_vertical(x, y, src, width, height, pitch_elems,
                                             params.clarity_gaussian_tap_count_,
                                             params.clarity_gaussian_weights_);

    float4 high = make_float4(orig.x - blur.x, orig.y - blur.y, orig.z - blur.z, orig.w);

    const float lum  = detail_luminance(orig);
    const float t    = (lum - 0.5f) * 2.0f;
    const float mask = 1.0f - t * t;
    const float w    = mask * params.clarity_offset_;
    high.x *= w;
    high.y *= w;
    high.z *= w;

    dst[offset] = make_float4(orig.x + high.x, orig.y + high.y, orig.z + high.z, orig.w);
  }
};

struct GPU_SharpenBlurHorizontalKernel : GPUNeighborOpTag {
  __device__ __forceinline__ void operator()(int x, int y, const float4* __restrict src,
                                             float4* __restrict dst, int width, int height,
                                             size_t pitch_elems,
                                             GPUOperatorParams& params) const {
    const size_t offset = static_cast<size_t>(y) * pitch_elems + static_cast<size_t>(x);
    if (!params.sharpen_enabled_ || params.sharpen_offset_ == 0.0f || params.sharpen_radius_ <= 0.0f ||
        params.sharpen_gaussian_tap_count_ <= 0) {
      dst[offset] = src[offset];
      return;
    }

    dst[offset] = detail_blur_horizontal(x, y, src, width, height, pitch_elems,
                                         params.sharpen_gaussian_tap_count_,
                                         params.sharpen_gaussian_weights_);
  }
};

struct GPU_SharpenApplyVerticalKernel : GPUNeighborOpTag {
  __device__ __forceinline__ void operator()(int x, int y, const float4* __restrict src,
                                             float4* __restrict dst, int width, int height,
                                             size_t pitch_elems,
                                             GPUOperatorParams& params) const {
    const size_t offset = static_cast<size_t>(y) * pitch_elems + static_cast<size_t>(x);
    if (!params.sharpen_enabled_ || params.sharpen_offset_ == 0.0f || params.sharpen_radius_ <= 0.0f ||
        params.sharpen_gaussian_tap_count_ <= 0) {
      dst[offset] = src[offset];
      return;
    }

    const float4 orig = dst[offset];
    const float4 blur = detail_blur_vertical(x, y, src, width, height, pitch_elems,
                                             params.sharpen_gaussian_tap_count_,
                                             params.sharpen_gaussian_weights_);

    float4 high = make_float4(orig.x - blur.x, orig.y - blur.y, orig.z - blur.z,
                              orig.w - blur.w);

    if (params.sharpen_threshold_ > 0.0f) {
      const float hp_gray = detail_luminance(high);
      const float mask    = (fabsf(hp_gray) > params.sharpen_threshold_) ? 1.0f : 0.0f;
      high.x *= mask;
      high.y *= mask;
      high.z *= mask;
      high.w *= mask;
    }

    dst[offset] = make_float4(
        orig.x + high.x * params.sharpen_offset_, orig.y + high.y * params.sharpen_offset_,
        orig.z + high.z * params.sharpen_offset_, orig.w + high.w * params.sharpen_offset_);
  }
};

};  // namespace CUDA
};  // namespace puerhlab
