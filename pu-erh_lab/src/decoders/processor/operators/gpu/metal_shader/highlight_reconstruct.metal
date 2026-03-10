//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct HighlightParams {
  float correction[4];
  float chrominance[4];
  float clip_val;
  float lo_clip_val;
  uint  width;
  uint  height;
  uint  stride;
  uint  m_width;
  uint  m_height;
  uint  m_size;
};

constant float kHLPowerF    = 3.0f;
constant float kInvHLPowerF = 1.0f / kHLPowerF;

static inline uint FC(int y, int x) {
  return (y & 1) ? ((x & 1) ? 1u : 2u) : ((x & 1) ? 1u : 0u);
}

static inline uint RawToCmap(uint m_width, int row, int col) {
  return static_cast<uint>(row / 3) * m_width + static_cast<uint>(col / 3);
}

static inline uchar MaskDilate(device const uchar* in, uint w1) {
  if (in[0]) {
    return 1;
  }

  if (in[-static_cast<int>(w1) - 1] | in[-static_cast<int>(w1)] | in[-static_cast<int>(w1) + 1] |
      in[-1] | in[1] | in[static_cast<int>(w1) - 1] | in[static_cast<int>(w1)] |
      in[static_cast<int>(w1) + 1]) {
    return 1;
  }

  const int w2 = 2 * static_cast<int>(w1);
  const int w3 = 3 * static_cast<int>(w1);
  return (in[-w3 - 2] | in[-w3 - 1] | in[-w3] | in[-w3 + 1] | in[-w3 + 2] | in[-w2 - 3] |
          in[-w2 - 2] | in[-w2 - 1] | in[-w2] | in[-w2 + 1] | in[-w2 + 2] | in[-w2 + 3] |
          in[-static_cast<int>(w1) - 3] | in[-static_cast<int>(w1) - 2] |
          in[-static_cast<int>(w1) + 2] | in[-static_cast<int>(w1) + 3] | in[-3] | in[-2] |
          in[2] | in[3] | in[static_cast<int>(w1) - 3] | in[static_cast<int>(w1) - 2] |
          in[static_cast<int>(w1) + 2] | in[static_cast<int>(w1) + 3] | in[w2 - 3] |
          in[w2 - 2] | in[w2 - 1] | in[w2] | in[w2 + 1] | in[w2 + 2] | in[w2 + 3] |
          in[w3 - 2] | in[w3 - 1] | in[w3] | in[w3 + 1] | in[w3 + 2])
             ? 1
             : 0;
}

static inline float CalcRefavg(device const float* in, int row, int col,
                               constant HighlightParams& params) {
  const uint color = FC(row, col);
  float      mean[3] = {0.0f, 0.0f, 0.0f};
  float      cnt[3]  = {0.0f, 0.0f, 0.0f};

  const int dymin = (row > 0) ? (row - 1) : 0;
  const int dxmin = (col > 0) ? (col - 1) : 0;
  const int dymax = (row + 2 < static_cast<int>(params.height) - 1) ? (row + 2)
                                                                     : (static_cast<int>(params.height) - 1);
  const int dxmax = (col + 2 < static_cast<int>(params.width) - 1) ? (col + 2)
                                                                    : (static_cast<int>(params.width) - 1);

  for (int dy = dymin; dy < dymax; ++dy) {
    for (int dx = dxmin; dx < dxmax; ++dx) {
      const float val = max(0.0f, in[static_cast<uint>(dy) * params.stride + static_cast<uint>(dx)]);
      const uint  c   = FC(dy, dx);
      mean[c] += val;
      cnt[c] += 1.0f;
    }
  }

  for (uint c = 0; c < 3u; ++c) {
    mean[c] = (cnt[c] > 0.0f) ? pow(params.correction[c] * mean[c] / cnt[c], kInvHLPowerF) : 0.0f;
  }

  const float croot_refavg[3] = {0.5f * (mean[1] + mean[2]), 0.5f * (mean[0] + mean[2]),
                                 0.5f * (mean[0] + mean[1])};
  return pow(croot_refavg[color], kHLPowerF);
}

kernel void hlr_build_mask(device const float* input [[buffer(0)]],
                           device uchar*       mask_buf [[buffer(1)]],
                           constant HighlightParams& params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.m_width || gid.y >= params.m_height) {
    return;
  }
  if (gid.x < 1u || gid.x >= params.m_width - 1u || gid.y < 1u || gid.y >= params.m_height - 1u) {
    return;
  }

  const int row = static_cast<int>(gid.y);
  const int col = static_cast<int>(gid.x);

  uchar mbuff[3] = {0, 0, 0};
  const int base_raw_y = 3 * row;
  const int base_raw_x = 3 * col;

  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      const int raw_y   = base_raw_y + y;
      const int raw_x   = base_raw_x + x;
      const float val   = input[static_cast<uint>(raw_y) * params.stride + static_cast<uint>(raw_x)];
      const uint  color = FC(row + y, col + x);
      mbuff[color] += (val >= params.clip_val) ? 1 : 0;
    }
  }

  const uint idx = gid.y * params.m_width + gid.x;
  for (uint c = 0; c < 3u; ++c) {
    if (mbuff[c]) {
      mask_buf[c * params.m_size + idx] = 1;
    }
  }
}

kernel void hlr_dilate_mask(device uchar* mask_buf [[buffer(0)]],
                            constant HighlightParams& params [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.m_width || gid.y >= params.m_height) {
    return;
  }
  if (gid.x < 3u || gid.x >= params.m_width - 3u || gid.y < 3u || gid.y >= params.m_height - 3u) {
    return;
  }

  const uint idx = gid.y * params.m_width + gid.x;
  mask_buf[3u * params.m_size + idx] = MaskDilate(mask_buf + 0u * params.m_size + idx, params.m_width);
  mask_buf[4u * params.m_size + idx] = MaskDilate(mask_buf + 1u * params.m_size + idx, params.m_width);
  mask_buf[5u * params.m_size + idx] = MaskDilate(mask_buf + 2u * params.m_size + idx, params.m_width);
}

kernel void hlr_chrominance_contrib(device const float* input [[buffer(0)]],
                                    device const uchar* mask_buf [[buffer(1)]],
                                    device float*       contrib [[buffer(2)]],
                                    device uint*        flags [[buffer(3)]],
                                    constant HighlightParams& params [[buffer(4)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  if (gid.x < 3u || gid.x >= params.width - 3u || gid.y < 3u || gid.y >= params.height - 3u) {
    return;
  }

  const int   row   = static_cast<int>(gid.y);
  const int   col   = static_cast<int>(gid.x);
  const uint  color = FC(row, col);
  const uint  index = gid.y * params.stride + gid.x;
  const float inval = input[index];

  if ((inval < params.clip_val) && (inval > params.lo_clip_val) &&
      mask_buf[(color + 3u) * params.m_size + RawToCmap(params.m_width, row, col)]) {
    contrib[index] = inval - CalcRefavg(input, row, col, params);
    flags[index]   = 1u;
  }
}

kernel void hlr_reconstruct(device const float* input [[buffer(0)]],
                            device float*       output [[buffer(1)]],
                            constant HighlightParams& params [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const int   row   = static_cast<int>(gid.y);
  const int   col   = static_cast<int>(gid.x);
  const uint  color = FC(row, col);
  const uint  index = gid.y * params.stride + gid.x;
  const float inval = max(0.0f, input[index]);

  if (inval >= params.clip_val) {
    const float ref = CalcRefavg(input, row, col, params);
    output[index]   = max(inval, ref + params.chrominance[color]);
  } else {
    output[index] = inval;
  }
}
