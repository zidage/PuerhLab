//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct WBParams {
  float black_level[4];
  float wb_multipliers[4];
};

struct ToLinearRefParams {
  float white_level_scale;
  uint  width;
  uint  height;
  uint  stride;
  uint  tile_width;
  uint  tile_height;
  uint  black_tile_width;
  uint  black_tile_height;
  uint  raw_fc[36];
};

static inline uint RawColorAt(constant ToLinearRefParams& params, uint y, uint x) {
  const uint tile_y = params.tile_height == 0u ? 0u : (y % params.tile_height);
  const uint tile_x = params.tile_width == 0u ? 0u : (x % params.tile_width);
  return params.raw_fc[tile_y * params.tile_width + tile_x];
}

static inline float PatternBlackAt(constant ToLinearRefParams& params, device const float* pattern_black,
                                   uint y, uint x) {
  if (params.black_tile_width == 0u || params.black_tile_height == 0u) {
    return 0.0f;
  }
  const uint tile_y = y % params.black_tile_height;
  const uint tile_x = x % params.black_tile_width;
  return pattern_black[tile_y * params.black_tile_width + tile_x];
}

kernel void to_linear_ref_r32f(device float*                image [[buffer(0)]],
                               constant ToLinearRefParams&  params [[buffer(1)]],
                               constant WBParams&           wb_params [[buffer(2)]],
                               device const float*          pattern_black [[buffer(3)]],
                               uint2                        gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const uint color_idx = RawColorAt(params, gid.y, gid.x);
  const uint index     = gid.y * params.stride + gid.x;

  float pixel_val      = image[index];
  pixel_val -= wb_params.black_level[color_idx] + PatternBlackAt(params, pattern_black, gid.y, gid.x);

  const float mask = (color_idx == 0u || color_idx == 2u) ? 1.0f : 0.0f;
  const float wb_mul = (wb_params.wb_multipliers[color_idx] / wb_params.wb_multipliers[1]) * mask +
                       (1.0f - mask);
  pixel_val *= wb_mul;
  pixel_val /= params.white_level_scale;

  image[index] = pixel_val;
}
