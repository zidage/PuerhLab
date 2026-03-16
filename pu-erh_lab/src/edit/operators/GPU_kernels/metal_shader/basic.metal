//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "common.metal"

static inline float metal_sigmoid(float t) { return 1.0f / (1.0f + exp(-t)); }

static inline float metal_contrast_sigmoid_01(float x, float k) {
  const float a = metal_sigmoid(-0.5f * k);
  const float b = metal_sigmoid(0.5f * k);
  const float y = metal_sigmoid(k * (x - 0.5f));
  return (y - a) / (b - a);
}

static inline float3 metal_contrast_on_luma_acescc(float3 rgb_acescc, float k = 6.0f,
                                                   float pivot = 0.5f, float range = 0.35f,
                                                   float eps = 1e-6f) {
  const float Y = 0.2126f * rgb_acescc.x + 0.7152f * rgb_acescc.y + 0.0722f * rgb_acescc.z;
  const float lo = pivot - range;
  const float hi = pivot + range;
  const float t  = (Y - lo) / (hi - lo);
  if (t <= 0.0f || t >= 1.0f) {
    return rgb_acescc;
  }

  const float t2    = metal_contrast_sigmoid_01(t, k);
  const float Y2    = lo + (hi - lo) * t2;
  const float scale = Y2 / fmax(Y, eps);
  return rgb_acescc * scale;
}

static inline float metal_luma(float3 rgb) {
  return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

static inline float metal_evaluate_curve_hermite(float x, constant MetalFusedParams& params) {
  const int curve_count = params.curve_ctrl_pts_size_;
  if (curve_count <= 0) {
    return x;
  }
  if (curve_count == 1) {
    return params.curve_ctrl_pts_y_[0];
  }
  if (x >= params.curve_ctrl_pts_x_[curve_count - 1]) {
    return params.curve_ctrl_pts_y_[curve_count - 1];
  }

  int idx = curve_count - 2;
  for (int i = 0; i < curve_count - 1; ++i) {
    if (x < params.curve_ctrl_pts_x_[i + 1]) {
      idx = i;
      break;
    }
  }

  const float dx = params.curve_h_[idx];
  if (fabs(dx) <= 1e-8f) {
    return params.curve_ctrl_pts_y_[idx];
  }

  const float t   = (x - params.curve_ctrl_pts_x_[idx]) / dx;
  const float h00 = 2.0f * t * t * t - 3.0f * t * t + 1.0f;
  const float h10 = t * t * t - 2.0f * t * t + t;
  const float h01 = -2.0f * t * t * t + 3.0f * t * t;
  const float h11 = t * t * t - t * t;
  const float y   = h00 * params.curve_ctrl_pts_y_[idx] + h10 * dx * params.curve_m_[idx] +
                  h01 * params.curve_ctrl_pts_y_[idx + 1] + h11 * dx * params.curve_m_[idx + 1];
  return y;
}

static inline float4 GPU_ExposureOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.exposure_enabled_ == 0u) {
    return px;
  }
  px.xyz += params.exposure_offset_;
  return px;
}

static inline float4 GPU_ToneOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.white_enabled_ == 0u && params.black_enabled_ == 0u) {
    return px;
  }
  px.xyz = px.xyz * params.slope_ + params.black_point_;
  return px;
}

static inline float4 GPU_ContrastOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.contrast_enabled_ == 0u) {
    return px;
  }
  px.xyz = metal_contrast_on_luma_acescc(px.xyz, params.contrast_scale_);
  return px;
}

static inline float4 GPU_HighlightOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.highlights_enabled_ == 0u) {
    return px;
  }

  const float L = metal_luma(px.xyz);
  float outL    = L;
  if (L <= params.highlights_k_) {
    outL = L;
  } else if (L < 1.0f) {
    const float t   = (L - params.highlights_k_) / params.highlights_dx_;
    const float H00 = 2.0f * t * t * t - 3.0f * t * t + 1.0f;
    const float H10 = t * t * t - 2.0f * t * t + t;
    const float H01 = -2.0f * t * t * t + 3.0f * t * t;
    const float H11 = t * t * t - t * t;
    outL            = H00 * params.highlights_k_ +
           H10 * (params.highlights_dx_ * params.highlights_m0_) + H01 * 1.0f +
           H11 * (params.highlights_dx_ * params.highlights_m1_);
  } else {
    const float x  = L - 1.0f;
    const float m1 = params.highlights_m1_;

    if (m1 < 1.0f) {
      const float rolloff_strength = 1.0f;
      const float b                = fmax(1e-6f, (1.0f - m1) * rolloff_strength);
      outL                         = 1.0f + (m1 * x) / (1.0f + b * x);
    } else {
      outL = 1.0f + x * m1;
    }
  }

  if (!isfinite(outL)) {
    outL = L;
  }
  const float scale = (L > 1e-8f) ? (outL / L) : 1.0f;
  px.xyz *= scale;
  return px;
}

static inline float4 GPU_ShadowOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.shadows_enabled_ == 0u) {
    return px;
  }

  const float L = metal_luma(px.xyz);
  if (L < params.shadows_x1_) {
    const float t    = (L - params.shadows_x0_) / params.shadows_dx_;
    const float H00  = 2.0f * t * t * t - 3.0f * t * t + 1.0f;
    const float H10  = t * t * t - 2.0f * t * t + t;
    const float H01  = -2.0f * t * t * t + 3.0f * t * t;
    const float H11  = t * t * t - t * t;
    const float outL = H00 * params.shadows_y0_ + H10 * (params.shadows_dx_ * params.shadows_m0_) +
                     H01 * params.shadows_y1_ + H11 * (params.shadows_dx_ * params.shadows_m1_);
    const float scale = (L > 1e-8f) ? (outL / L) : 1.0f;
    px.xyz *= scale;
  }
  return px;
}

static inline float4 GPU_CurveOpKernel(float4 px, constant MetalFusedParams& params) {
  if (params.curve_enabled_ == 0u || params.curve_ctrl_pts_size_ <= 0) {
    return px;
  }

  constexpr float kCurveInfluence = 0.65f;
  const float lum                 = metal_luma(px.xyz);
  const float mapped_lum          = metal_evaluate_curve_hermite(lum, params);
  const float new_lum             = lum + (mapped_lum - lum) * kCurveInfluence;
  const float ratio               = (lum > 1e-5f) ? (new_lum / lum) : 0.0f;
  px.xyz *= ratio;
  return px;
}
