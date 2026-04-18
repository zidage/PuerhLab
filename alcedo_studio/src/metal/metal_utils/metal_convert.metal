//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct ConvertParams {
  float    alpha;
  float    beta;
  uint     width;
  uint     height;
  uint     src_stride;
  uint     dst_stride;
};

struct CropParams {
  uint origin_x;
  uint origin_y;
  uint crop_width;
  uint crop_height;
  uint src_stride;
  uint dst_stride;
};

struct ClampParams {
  float lo;
  float hi;
  uint  width;
  uint  height;
  uint  stride;
};

struct RotateParams {
  uint src_width;
  uint src_height;
  uint dst_width;
  uint dst_height;
  uint src_stride;
  uint dst_stride;
};

static inline ushort SaturateU16(float value) {
  return static_cast<ushort>(clamp(value, 0.0f, 65535.0f));
}

static inline ushort4 SaturateU16(float4 value) {
  return ushort4(SaturateU16(value.x), SaturateU16(value.y), SaturateU16(value.z),
                 SaturateU16(value.w));
}

kernel void convert_r16u_to_r16u(device const ushort* src [[buffer(0)]],
                                 device ushort*       dst [[buffer(1)]],
                                 constant ConvertParams& params [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint src_index = gid.y * params.src_stride + gid.x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = SaturateU16(float(src[src_index]) * params.alpha + params.beta);
}

kernel void convert_r16u_to_r32f(device const ushort* src [[buffer(0)]],
                                 device float*        dst [[buffer(1)]],
                                 constant ConvertParams& params [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint src_index = gid.y * params.src_stride + gid.x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = float(src[src_index]) * params.alpha + params.beta;
}

kernel void convert_r32f_to_r16u(device const float* src [[buffer(0)]],
                                 device ushort*      dst [[buffer(1)]],
                                 constant ConvertParams& params [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint src_index = gid.y * params.src_stride + gid.x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = SaturateU16(src[src_index] * params.alpha + params.beta);
}

kernel void convert_r32f_to_r32f(device const float* src [[buffer(0)]],
                                 device float*       dst [[buffer(1)]],
                                 constant ConvertParams& params [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint src_index = gid.y * params.src_stride + gid.x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = src[src_index] * params.alpha + params.beta;
}

kernel void convert_rgba16u_to_rgba16u(device const ushort4* src [[buffer(0)]],
                                       device ushort4*       dst [[buffer(1)]],
                                       constant ConvertParams& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint src_index = gid.y * params.src_stride + gid.x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = SaturateU16(float4(src[src_index]) * params.alpha + params.beta);
}

kernel void convert_rgba16u_to_rgba32f(device const ushort4* src [[buffer(0)]],
                                       device float4*        dst [[buffer(1)]],
                                       constant ConvertParams& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint src_index = gid.y * params.src_stride + gid.x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = float4(src[src_index]) * params.alpha + params.beta;
}

kernel void convert_rgba32f_to_rgba16u(device const float4* src [[buffer(0)]],
                                       device ushort4*      dst [[buffer(1)]],
                                       constant ConvertParams& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint src_index = gid.y * params.src_stride + gid.x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = SaturateU16(src[src_index] * params.alpha + params.beta);
}

kernel void convert_rgba32f_to_rgba32f(device const float4* src [[buffer(0)]],
                                       device float4*       dst [[buffer(1)]],
                                       constant ConvertParams& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint src_index = gid.y * params.src_stride + gid.x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = src[src_index] * params.alpha + params.beta;
}

kernel void crop_r32f(device const float* src [[buffer(0)]],
                      device float*       dst [[buffer(1)]],
                      constant CropParams& params [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.crop_width || gid.y >= params.crop_height) {
    return;
  }

  const uint src_index = (gid.y + params.origin_y) * params.src_stride + gid.x + params.origin_x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = src[src_index];
}

kernel void crop_rgba32f(device const float4* src [[buffer(0)]],
                         device float4*       dst [[buffer(1)]],
                         constant CropParams& params [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.crop_width || gid.y >= params.crop_height) {
    return;
  }

  const uint src_index = (gid.y + params.origin_y) * params.src_stride + gid.x + params.origin_x;
  const uint dst_index = gid.y * params.dst_stride + gid.x;
  dst[dst_index]       = src[src_index];
}

kernel void clamp_r32f(device float*       image [[buffer(0)]],
                       constant ClampParams& params [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const uint index = gid.y * params.stride + gid.x;
  image[index]     = clamp(image[index], params.lo, params.hi);
}

kernel void clamp_rgba32f(device float4*       image [[buffer(0)]],
                          constant ClampParams& params [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const uint index = gid.y * params.stride + gid.x;
  image[index]     = clamp(image[index], float4(params.lo), float4(params.hi));
}

kernel void rotate180_r32f(device const float* src [[buffer(0)]],
                           device float*       dst [[buffer(1)]],
                           constant RotateParams& params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }

  const uint src_x   = params.src_width - 1u - gid.x;
  const uint src_y   = params.src_height - 1u - gid.y;
  const uint src_idx = src_y * params.src_stride + src_x;
  const uint dst_idx = gid.y * params.dst_stride + gid.x;
  dst[dst_idx]       = src[src_idx];
}

kernel void rotate180_rgba32f(device const float4* src [[buffer(0)]],
                              device float4*       dst [[buffer(1)]],
                              constant RotateParams& params [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }

  const uint src_x   = params.src_width - 1u - gid.x;
  const uint src_y   = params.src_height - 1u - gid.y;
  const uint src_idx = src_y * params.src_stride + src_x;
  const uint dst_idx = gid.y * params.dst_stride + gid.x;
  dst[dst_idx]       = src[src_idx];
}

kernel void rotate90cw_r32f(device const float* src [[buffer(0)]],
                            device float*       dst [[buffer(1)]],
                            constant RotateParams& params [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }

  const uint src_x   = gid.y;
  const uint src_y   = params.src_height - 1u - gid.x;
  const uint src_idx = src_y * params.src_stride + src_x;
  const uint dst_idx = gid.y * params.dst_stride + gid.x;
  dst[dst_idx]       = src[src_idx];
}

kernel void rotate90cw_rgba32f(device const float4* src [[buffer(0)]],
                               device float4*       dst [[buffer(1)]],
                               constant RotateParams& params [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }

  const uint src_x   = gid.y;
  const uint src_y   = params.src_height - 1u - gid.x;
  const uint src_idx = src_y * params.src_stride + src_x;
  const uint dst_idx = gid.y * params.dst_stride + gid.x;
  dst[dst_idx]       = src[src_idx];
}

kernel void rotate90ccw_r32f(device const float* src [[buffer(0)]],
                             device float*       dst [[buffer(1)]],
                             constant RotateParams& params [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }

  const uint src_x   = params.src_width - 1u - gid.y;
  const uint src_y   = gid.x;
  const uint src_idx = src_y * params.src_stride + src_x;
  const uint dst_idx = gid.y * params.dst_stride + gid.x;
  dst[dst_idx]       = src[src_idx];
}

kernel void rotate90ccw_rgba32f(device const float4* src [[buffer(0)]],
                                device float4*       dst [[buffer(1)]],
                                constant RotateParams& params [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }

  const uint src_x   = params.src_width - 1u - gid.y;
  const uint src_y   = gid.x;
  const uint src_idx = src_y * params.src_stride + src_x;
  const uint dst_idx = gid.y * params.dst_stride + gid.x;
  dst[dst_idx]       = src[src_idx];
}
