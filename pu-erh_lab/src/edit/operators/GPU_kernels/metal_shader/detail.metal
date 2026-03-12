//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "common.metal"

static inline float metal_detail_luminance(float4 c) {
  // Match the CUDA implementation's COLOR_BGR2GRAY coefficients.
  return c.x * 0.114f + c.y * 0.587f + c.z * 0.299f;
}

static inline float metal_detail_gaussian(float dx, float dy, float sigma) {
  const float inv2sigma2 = 0.5f / (sigma * sigma);
  return exp(-(dx * dx + dy * dy) * inv2sigma2);
}

static inline float4 metal_detail_read_clamped(texture2d<float, access::read> src, int2 coord) {
  const int x = clamp(coord.x, 0, static_cast<int>(src.get_width()) - 1);
  const int y = clamp(coord.y, 0, static_cast<int>(src.get_height()) - 1);
  return src.read(uint2(static_cast<uint>(x), static_cast<uint>(y)));
}

static inline float4 metal_detail_gaussian_blur(texture2d<float, access::read> src, uint2 gid,
                                                float sigma, int max_radius) {
  const float safe_sigma = fmax(sigma, 1.0e-4f);
  int radius             = static_cast<int>(ceil(3.0f * safe_sigma));
  radius                 = clamp(radius, 1, max_radius);

  float4 blur            = float4(0.0f);
  float sum_w            = 0.0f;
  const int2 center      = int2(static_cast<int>(gid.x), static_cast<int>(gid.y));

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const float  w = metal_detail_gaussian(static_cast<float>(dx), static_cast<float>(dy), safe_sigma);
      const float4 v = metal_detail_read_clamped(src, center + int2(dx, dy));
      blur += v * w;
      sum_w += w;
    }
  }

  return (sum_w > 0.0f) ? (blur / sum_w) : metal_detail_read_clamped(src, center);
}

static inline float4 GPU_SharpenKernel(texture2d<float, access::read> src, uint2 gid, float4 px,
                                       constant MetalFusedParams& params) {
  if (params.sharpen_enabled_ == 0u || params.sharpen_offset_ == 0.0f ||
      params.sharpen_radius_ <= 0.0f) {
    return px;
  }

  const float4 blur = metal_detail_gaussian_blur(src, gid, params.sharpen_radius_, 15);
  float4 high       = px - blur;

  if (params.sharpen_threshold_ > 0.0f) {
    const float hp_gray = metal_detail_luminance(high);
    const float mask    = (fabs(hp_gray) > params.sharpen_threshold_) ? 1.0f : 0.0f;
    high *= mask;
  }

  return px + high * params.sharpen_offset_;
}

static inline float4 GPU_ClarityKernel(texture2d<float, access::read> src, uint2 gid, float4 px,
                                       constant MetalFusedParams& params) {
  if (params.clarity_enabled_ == 0u) {
    return px;
  }

  const float4 blur = metal_detail_gaussian_blur(src, gid, params.clarity_radius_, 20);
  float4 high       = float4(px.x - blur.x, px.y - blur.y, px.z - blur.z, px.w);

  const float lum   = metal_detail_luminance(px);
  const float t     = (lum - 0.5f) * 2.0f;
  const float mask  = 1.0f - t * t;
  const float w     = mask * params.clarity_offset_;
  high.xyz *= w;

  return float4(px.x + high.x, px.y + high.y, px.z + high.z, px.w);
}

kernel void metal_detail_sharpen_rgba32f(texture2d<float, access::read> src [[texture(0)]],
                                         texture2d<float, access::write> dst [[texture(1)]],
                                         constant MetalFusedParams& params [[buffer(0)]],
                                         uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
    return;
  }

  const float4 px = src.read(gid);
  dst.write(GPU_SharpenKernel(src, gid, px, params), gid);
}

kernel void metal_detail_clarity_rgba32f(texture2d<float, access::read> src [[texture(0)]],
                                         texture2d<float, access::write> dst [[texture(1)]],
                                         constant MetalFusedParams& params [[buffer(0)]],
                                         uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
    return;
  }

  const float4 px = src.read(gid);
  dst.write(GPU_ClarityKernel(src, gid, px, params), gid);
}
