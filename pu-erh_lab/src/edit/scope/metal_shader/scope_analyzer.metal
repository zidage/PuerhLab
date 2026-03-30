//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct ScopeAnalysisParams {
  uint input_width;
  uint input_height;
  uint sample_step;
  uint histogram_bins;
  uint waveform_width;
  uint waveform_height;
  uint waveform_stride;
};

kernel void scope_accumulate_histogram(texture2d<float, access::read> input_texture [[texture(0)]],
                                       device atomic_uint* histogram_counts [[buffer(0)]],
                                       constant ScopeAnalysisParams& params [[buffer(1)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  const uint x = gid.x * params.sample_step;
  const uint y = gid.y * params.sample_step;
  if (x >= params.input_width || y >= params.input_height || params.histogram_bins == 0) {
    return;
  }

  const float4 pixel = input_texture.read(uint2(x, y));
  const float3 rgb   = clamp(pixel.rgb, 0.0f, 1.0f);

  for (uint channel = 0; channel < 3; ++channel) {
    const uint bin = min(static_cast<uint>(rgb[channel] * float(params.histogram_bins - 1u) + 0.5f),
                         params.histogram_bins - 1u);
    atomic_fetch_add_explicit(&histogram_counts[channel * params.histogram_bins + bin], 1u,
                              memory_order_relaxed);
  }
}

kernel void scope_accumulate_waveform(texture2d<float, access::read> input_texture [[texture(0)]],
                                      device atomic_uint* waveform_counts [[buffer(0)]],
                                      constant ScopeAnalysisParams& params [[buffer(1)]],
                                      uint2 gid [[thread_position_in_grid]]) {
  const uint x = gid.x * params.sample_step;
  const uint y = gid.y * params.sample_step;
  if (x >= params.input_width || y >= params.input_height || params.waveform_width == 0 ||
      params.waveform_height == 0 || params.waveform_stride == 0) {
    return;
  }

  const float4 pixel = input_texture.read(uint2(x, y));
  const float3 rgb   = clamp(pixel.rgb, 0.0f, 1.0f);

  const float width_denom = float(max(params.input_width - 1u, 1u));
  const uint x_bin =
      min(static_cast<uint>((float(x) / width_denom) * float(params.waveform_width - 1u) + 0.5f),
          params.waveform_width - 1u);

  for (uint channel = 0; channel < 3; ++channel) {
    const uint y_bin =
        params.waveform_height - 1u -
        min(static_cast<uint>(rgb[channel] * float(params.waveform_height - 1u) + 0.5f),
            params.waveform_height - 1u);
    const uint base = (y_bin * params.waveform_stride + x_bin) * 4u;
    atomic_fetch_add_explicit(&waveform_counts[base + channel], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&waveform_counts[base + 3u], 1u, memory_order_relaxed);
  }
}
