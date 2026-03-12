//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "common.metal"

constant int kMetalHlsProfileCount = 8;

static inline float metal_wrap_hue(float h) {
  h = fmod(h, 360.0f);
  if (h < 0.0f) {
    h += 360.0f;
  }
  return h;
}

static inline float metal_hue2rgb(float p, float q, float t) {
  if (t < 0.0f) {
    t += 1.0f;
  }
  if (t > 1.0f) {
    t -= 1.0f;
  }
  if (t < (1.0f / 6.0f)) {
    return p + (q - p) * 6.0f * t;
  }
  if (t < 0.5f) {
    return q;
  }
  if (t < (2.0f / 3.0f)) {
    return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
  }
  return p;
}

static inline float4 GPU_SaturationOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.saturation_enabled_ == 0u) {
    return px;
  }

  const float luma = 0.2126f * px.x + 0.7152f * px.y + 0.0722f * px.z;
  px.x             = luma + (px.x - luma) * params.saturation_offset_;
  px.y             = luma + (px.y - luma) * params.saturation_offset_;
  px.z             = luma + (px.z - luma) * params.saturation_offset_;
  return px;
}

static inline float4 GPU_TintOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.tint_enabled_ == 0u) {
    return px;
  }

  px.y += params.tint_offset_;
  return px;
}

static inline float4 GPU_VibranceOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.vibrance_enabled_ == 0u) {
    return px;
  }

  const float max_val = fmax(fmax(px.x, px.y), px.z);
  const float min_val = fmin(fmin(px.x, px.y), px.z);
  const float chroma  = max_val - min_val;
  const float strength = params.vibrance_offset_;
  const float falloff  = exp(-3.0f * chroma);
  const float scale    = 1.0f + strength * falloff;

  if (params.vibrance_offset_ >= 0.0f) {
    const float luma = px.x * 0.299f + px.y * 0.587f + px.z * 0.114f;
    px.x             = luma + (px.x - luma) * scale;
    px.y             = luma + (px.y - luma) * scale;
    px.z             = luma + (px.z - luma) * scale;
  } else {
    const float avg = (px.x + px.y + px.z) / 3.0f;
    px.x += (avg - px.x) * (1.0f - scale);
    px.y += (avg - px.y) * (1.0f - scale);
    px.z += (avg - px.z) * (1.0f - scale);
  }
  return px;
}

static inline float4 GPU_ColorWheelOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.color_wheel_enabled_ == 0u) {
    return px;
  }

  constexpr float kEps = 1e-6f;
  const float offset_r = params.lift_color_offset_[0] + params.lift_luminance_offset_;
  const float offset_g = params.lift_color_offset_[1] + params.lift_luminance_offset_;
  const float offset_b = params.lift_color_offset_[2] + params.lift_luminance_offset_;

  const float slope_r  = fmax(params.gain_color_offset_[0] + params.gain_luminance_offset_, kEps);
  const float slope_g  = fmax(params.gain_color_offset_[1] + params.gain_luminance_offset_, kEps);
  const float slope_b  = fmax(params.gain_color_offset_[2] + params.gain_luminance_offset_, kEps);

  const float power_r  = fmax(params.gamma_color_offset_[0] + params.gamma_luminance_offset_, kEps);
  const float power_g  = fmax(params.gamma_color_offset_[1] + params.gamma_luminance_offset_, kEps);
  const float power_b  = fmax(params.gamma_color_offset_[2] + params.gamma_luminance_offset_, kEps);

  const float base_r   = fmax(px.x * slope_r + offset_r, 0.0f);
  const float base_g   = fmax(px.y * slope_g + offset_g, 0.0f);
  const float base_b   = fmax(px.z * slope_b + offset_b, 0.0f);

  px.x                 = clamp(pow(base_r, power_r), 0.0f, 1.0f);
  px.y                 = clamp(pow(base_g, power_g), 0.0f, 1.0f);
  px.z                 = clamp(pow(base_b, power_b), 0.0f, 1.0f);
  return px;
}

static inline float4 GPU_HLSOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.hls_enabled_ == 0u) {
    return px;
  }

  constexpr float kEps = 1e-6f;
  const float r        = clamp(px.x, 0.0f, 1.0f);
  const float g        = clamp(px.y, 0.0f, 1.0f);
  const float b        = clamp(px.z, 0.0f, 1.0f);

  const float max_c    = fmax(fmax(r, g), b);
  const float min_c    = fmin(fmin(r, g), b);
  const float L        = (max_c + min_c) * 0.5f;
  float H              = 0.0f;
  float S              = 0.0f;
  const float d        = max_c - min_c;
  if (d > kEps) {
    const float denom = fmax(1.0f - fabs(2.0f * L - 1.0f), kEps);
    S                 = clamp(d / denom, 0.0f, 1.0f);
    if (max_c == r) {
      H = (g - b) / d + (g < b ? 6.0f : 0.0f);
    } else if (max_c == g) {
      H = (b - r) / d + 2.0f;
    } else {
      H = (r - g) / d + 4.0f;
    }
    H *= 60.0f;
  }

  int profile_count = params.hls_profile_count_;
  if (profile_count < 1) {
    profile_count = 1;
  }
  if (profile_count > kMetalHlsProfileCount) {
    profile_count = kMetalHlsProfileCount;
  }

  const float h = metal_wrap_hue(H);
  float accum_h = 0.0f;
  float accum_l = 0.0f;
  float accum_s = 0.0f;
  bool has_contribution = false;

  for (int i = 0; i < kMetalHlsProfileCount; ++i) {
    if (i >= profile_count) {
      continue;
    }

    const float adj_h = params.hls_profile_adjustments_[i][0];
    const float adj_l = params.hls_profile_adjustments_[i][1];
    const float adj_s = params.hls_profile_adjustments_[i][2];
    if (fabs(adj_h) <= kEps && fabs(adj_l) <= kEps && fabs(adj_s) <= kEps) {
      continue;
    }

    const float hue_range = fmax(params.hls_profile_hue_ranges_[i], kEps);
    const float target_h  = metal_wrap_hue(params.hls_profile_hues_[i]);
    const float hue_diff  = fabs(h - target_h);
    const float hue_dist  = fmin(hue_diff, 360.0f - hue_diff);
    if (hue_dist >= hue_range) {
      continue;
    }

    const float weight = 1.0f - hue_dist / hue_range;
    accum_h += adj_h * weight;
    accum_l += adj_l * weight;
    accum_s += adj_s * weight;
    has_contribution = true;
  }

  if (!has_contribution) {
    return px;
  }

  const float h_adjusted = metal_wrap_hue(h + accum_h);
  const float l_adjusted = clamp(L + accum_l, 0.0f, 1.0f);
  const float s_adjusted = clamp(S + accum_s, 0.0f, 1.0f);
  if (s_adjusted <= kEps) {
    px.x = l_adjusted;
    px.y = l_adjusted;
    px.z = l_adjusted;
  } else {
    const float q = (l_adjusted < 0.5f) ? (l_adjusted * (1.0f + s_adjusted))
                                        : (l_adjusted + s_adjusted - l_adjusted * s_adjusted);
    const float p = 2.0f * l_adjusted - q;

    px.x = metal_hue2rgb(p, q, h_adjusted / 360.0f + 1.0f / 3.0f);
    px.y = metal_hue2rgb(p, q, h_adjusted / 360.0f);
    px.z = metal_hue2rgb(p, q, h_adjusted / 360.0f - 1.0f / 3.0f);
  }

  px.xyz = clamp(px.xyz, float3(0.0f), float3(1.0f));
  return px;
}
