//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct InverseCamMulParams {
  float scale_r;
  float scale_g;
  float scale_b;
  float scale_a;
  uint  width;
  uint  height;
  uint  stride;
};

kernel void apply_inverse_cam_mul_rgba32f(device float4* image [[buffer(0)]],
                                          constant InverseCamMulParams& params [[buffer(1)]],
                                          uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const uint index = gid.y * params.stride + gid.x;
  float4 rgba      = image[index];
  rgba.r *= params.scale_r;
  rgba.g *= params.scale_g;
  rgba.b *= params.scale_b;
  rgba.a *= params.scale_a;
  image[index] = rgba;
}
