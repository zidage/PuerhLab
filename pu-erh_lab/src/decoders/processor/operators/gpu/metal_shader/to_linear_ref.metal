//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct WBParams {
  float black_level[4];
  float wb_scale[4];
};

struct ToLinearRefParams {
  float white_level_scale;
  uint  width;
  uint  height;
  uint  stride;
  uint  raw_fc[4];
};

kernel void to_linear_ref_r32f(device float*                image [[buffer(0)]],
                               constant ToLinearRefParams&  params [[buffer(1)]],
                               constant WBParams&           wb_params [[buffer(2)]],
                               uint2                        gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const uint color_idx = params.raw_fc[((gid.y & 1u) << 1u) | (gid.x & 1u)];
  const uint index     = gid.y * params.stride + gid.x;

  float pixel_val      = image[index];
  pixel_val -= wb_params.black_level[color_idx];
  pixel_val *= wb_params.wb_scale[color_idx];
  pixel_val /= params.white_level_scale;

  image[index] = pixel_val;
}
