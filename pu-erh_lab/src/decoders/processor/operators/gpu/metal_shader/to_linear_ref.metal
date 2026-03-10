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
};

constant ushort kRemap[4] = {0, 1, 3, 2};

kernel void to_linear_ref_r32f(device float*                image [[buffer(0)]],
                               constant ToLinearRefParams&  params [[buffer(1)]],
                               constant WBParams&           wb_params [[buffer(2)]],
                               uint2                        gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const uint color_idx = kRemap[((gid.y & 1u) << 1u) | (gid.x & 1u)];
  const uint index     = gid.y * params.stride + gid.x;

  float pixel_val      = image[index];
  pixel_val -= wb_params.black_level[color_idx];

  const float mask   = (color_idx == 0u || color_idx == 2u) ? 1.0f : 0.0f;
  const float wb_mul =
      (wb_params.wb_multipliers[color_idx] / wb_params.wb_multipliers[1]) * mask + (1.0f - mask);
  pixel_val *= wb_mul;
  pixel_val /= params.white_level_scale;

  image[index] = pixel_val;
}
