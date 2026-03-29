//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct HighlightParams {
  float clips[4];
  float clipdark[4];
  float chrominance[4];
  uint  width;
  uint  height;
  uint  stride;
};

constant uint kDilateRadius  = 3u;
constant uint kPlaneBaseR    = 0u;
constant uint kPlaneBaseG    = 1u;
constant uint kPlaneBaseB    = 2u;
constant uint kPlaneDilatedR = 3u;
constant uint kPlaneDilatedG = 4u;
constant uint kPlaneDilatedB = 5u;
constant uint kPlaneBaseMulti = 6u;
constant uint kPlaneDilMulti  = 7u;

static inline float Cube(float value) { return value * value * value; }

static inline float3 MaxRgb(float4 value) { return max(value.rgb, float3(0.0f)); }

static inline int CountClippedChannels(float3 pixel, constant HighlightParams& params) {
  int count = 0;
  count += pixel.x >= params.clips[0] ? 1 : 0;
  count += pixel.y >= params.clips[1] ? 1 : 0;
  count += pixel.z >= params.clips[2] ? 1 : 0;
  return count;
}

static inline float3 CalcRefavg(device const float4* input, int row, int col,
                                constant HighlightParams& params) {
  float mean[3] = {0.0f, 0.0f, 0.0f};
  float cnt[3]  = {0.0f, 0.0f, 0.0f};

  const int dymin = max(0, row - 1);
  const int dxmin = max(0, col - 1);
  const int dymax = min(static_cast<int>(params.height) - 1, row + 1);
  const int dxmax = min(static_cast<int>(params.width) - 1, col + 1);

  for (int dy = dymin; dy <= dymax; ++dy) {
    for (int dx = dxmin; dx <= dxmax; ++dx) {
      const float3 sample =
          MaxRgb(input[static_cast<uint>(dy) * params.stride + static_cast<uint>(dx)]);
      mean[0] += sample.x;
      mean[1] += sample.y;
      mean[2] += sample.z;
      cnt[0] += 1.0f;
      cnt[1] += 1.0f;
      cnt[2] += 1.0f;
    }
  }

  for (uint c = 0; c < 3u; ++c) {
    mean[c] = (cnt[c] > 0.0f) ? pow(mean[c] / cnt[c], 1.0f / 3.0f) : 0.0f;
  }

  return float3(Cube(0.5f * (mean[1] + mean[2])), Cube(0.5f * (mean[0] + mean[2])),
                Cube(0.5f * (mean[0] + mean[1])));
}

static inline uchar DilateMaskAt(device const uchar* plane, uint width, uint height, int row,
                                 int col, int radius) {
  const int y0 = max(0, row - radius);
  const int x0 = max(0, col - radius);
  const int y1 = min(static_cast<int>(height) - 1, row + radius);
  const int x1 = min(static_cast<int>(width) - 1, col + radius);

  for (int y = y0; y <= y1; ++y) {
    const int row_offset = y * static_cast<int>(width);
    for (int x = x0; x <= x1; ++x) {
      if (plane[row_offset + x] != 0) {
        return 1;
      }
    }
  }

  return 0;
}

kernel void hlr_build_mask(device const float4* input [[buffer(0)]],
                           device uchar*        mask_buf [[buffer(1)]],
                           device atomic_uint*  anyclipped [[buffer(2)]],
                           constant HighlightParams& params [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint   size  = params.width * params.height;
  const uint   index = gid.y * params.stride + gid.x;
  const uint   idx   = gid.y * params.width + gid.x;
  const float3 pixel = MaxRgb(input[index]);
  const int    count = CountClippedChannels(pixel, params);

  mask_buf[kPlaneBaseR * size + idx]    = pixel.x >= params.clips[0] ? 1 : 0;
  mask_buf[kPlaneBaseG * size + idx]    = pixel.y >= params.clips[1] ? 1 : 0;
  mask_buf[kPlaneBaseB * size + idx]    = pixel.z >= params.clips[2] ? 1 : 0;
  mask_buf[kPlaneBaseMulti * size + idx] = count >= 2 ? 1 : 0;

  if (count > 0) {
    atomic_store_explicit(anyclipped, 1u, memory_order_relaxed);
  }
}

kernel void hlr_dilate_mask(device const uchar* mask_buf [[buffer(0)]],
                            device uchar*       dilated_mask_buf [[buffer(1)]],
                            constant HighlightParams& params [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const uint size = params.width * params.height;
  const uint idx  = gid.y * params.width + gid.x;

  dilated_mask_buf[kPlaneDilatedR * size + idx] =
      DilateMaskAt(mask_buf + kPlaneBaseR * size, params.width, params.height,
                   static_cast<int>(gid.y), static_cast<int>(gid.x), static_cast<int>(kDilateRadius));
  dilated_mask_buf[kPlaneDilatedG * size + idx] =
      DilateMaskAt(mask_buf + kPlaneBaseG * size, params.width, params.height,
                   static_cast<int>(gid.y), static_cast<int>(gid.x), static_cast<int>(kDilateRadius));
  dilated_mask_buf[kPlaneDilatedB * size + idx] =
      DilateMaskAt(mask_buf + kPlaneBaseB * size, params.width, params.height,
                   static_cast<int>(gid.y), static_cast<int>(gid.x), static_cast<int>(kDilateRadius));
  dilated_mask_buf[kPlaneDilMulti * size + idx] =
      DilateMaskAt(mask_buf + kPlaneBaseMulti * size, params.width, params.height,
                   static_cast<int>(gid.y), static_cast<int>(gid.x), static_cast<int>(kDilateRadius));
}

kernel void hlr_chrominance_contrib(device const float4* input [[buffer(0)]],
                                    device const uchar*  mask_buf [[buffer(1)]],
                                    device float4*       contrib [[buffer(2)]],
                                    device float4*       counts [[buffer(3)]],
                                    constant HighlightParams& params [[buffer(4)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const uint   size  = params.width * params.height;
  const uint   index = gid.y * params.stride + gid.x;
  const uint   idx   = gid.y * params.width + gid.x;
  const float3 pixel = MaxRgb(input[index]);
  const float3 ref   = CalcRefavg(input, static_cast<int>(gid.y), static_cast<int>(gid.x), params);

  float4 contrib_value = float4(0.0f);
  float4 count_value   = float4(0.0f);

  if (mask_buf[kPlaneDilatedR * size + idx] && pixel.x > params.clipdark[0] &&
      pixel.x < params.clips[0]) {
    contrib_value.x = pixel.x - ref.x;
    count_value.x   = 1.0f;
  }
  if (mask_buf[kPlaneDilatedG * size + idx] && pixel.y > params.clipdark[1] &&
      pixel.y < params.clips[1]) {
    contrib_value.y = pixel.y - ref.y;
    count_value.y   = 1.0f;
  }
  if (mask_buf[kPlaneDilatedB * size + idx] && pixel.z > params.clipdark[2] &&
      pixel.z < params.clips[2]) {
    contrib_value.z = pixel.z - ref.z;
    count_value.z   = 1.0f;
  }

  contrib[index] = contrib_value;
  counts[index]  = count_value;
}

kernel void hlr_reconstruct(device const float4* input [[buffer(0)]],
                            device float4*       output [[buffer(1)]],
                            constant HighlightParams& params [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const uint  index = gid.y * params.stride + gid.x;
  const float4 input_pixel = input[index];
  const float3 pixel       = MaxRgb(input_pixel);
  const float3 ref         = CalcRefavg(input, static_cast<int>(gid.y), static_cast<int>(gid.x), params);

  float3 result = pixel;
  if (pixel.x >= params.clips[0]) {
    result.x = max(pixel.x, ref.x + params.chrominance[0]);
  }
  if (pixel.y >= params.clips[1]) {
    result.y = max(pixel.y, ref.y + params.chrominance[1]);
  }
  if (pixel.z >= params.clips[2]) {
    result.z = max(pixel.z, ref.z + params.chrominance[2]);
  }

  output[index] = float4(result, input_pixel.w);
}
