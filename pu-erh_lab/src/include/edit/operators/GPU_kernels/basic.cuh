//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// CUDA implementations of basic image operations

#pragma once
#include <cuda_runtime.h>

#include "edit/operators/op_kernel.hpp"
#include "param.cuh"

namespace puerhlab {
namespace CUDA {
// Basic point operation kernel
struct GPU_ExposureOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) {
    if (!params.exposure_enabled_) return;
    p->x = p->x + (params.exposure_offset_);
    p->y = p->y + (params.exposure_offset_);
    p->z = p->z + (params.exposure_offset_);
  }
};

struct GPU_ToneOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.white_enabled_ && !params.black_enabled_) return;
    p->x = p->x * params.slope_ + params.black_point_;
    p->y = p->y * params.slope_ + params.black_point_;
    p->z = p->z * params.slope_ + params.black_point_;
  }
};

GPU_FUNC float sigmoid(float t) { return 1.0f / (1.0f + expf(-t)); }

GPU_FUNC float contrast_sigmoid_01(float x, float k) {
  // x is assumed in [0,1]
  const float a = sigmoid(-0.5f * k);
  const float b = sigmoid(0.5f * k);
  const float y = sigmoid(k * (x - 0.5f));
  return (y - a) / (b - a);
}

GPU_FUNC float3 contrast_on_luma_acescc(float3 rgb_acescc, float k = 6.0f, float pivot = 0.5f,
                                        float range = 0.35f, float eps = 1e-6f) {
  // AP1 luma in ACEScc domain (approx approach; "professional-looking" for gentle ops)
  const float Y =
      0.2126f * rgb_acescc.x + 0.7152f * rgb_acescc.y + 0.0722f * rgb_acescc.z;

  const float lo = pivot - range;
  const float hi = pivot + range;

  // Map Y -> t in [0,1] over [lo, hi]
  const float t  = (Y - lo) / (hi - lo);

  // Outside window: identical
  if (t <= 0.0f || t >= 1.0f) return rgb_acescc;

  const float t2    = contrast_sigmoid_01(t, k);
  const float Y2    = lo + (hi - lo) * t2;

  const float scale = Y2 / fmaxf(Y, eps);

  return {rgb_acescc.x * scale, rgb_acescc.y * scale, rgb_acescc.z * scale};
}

struct GPU_ContrastOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.contrast_enabled_) return;
    float3 acescc_color = make_float3(p->x, p->y, p->z);
    acescc_color        = contrast_on_luma_acescc(acescc_color, params.contrast_scale_);
    p->x                = acescc_color.x;;
    p->y                = acescc_color.y;
    p->z                = acescc_color.z;
  }
};

struct GPU_HighlightOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.highlights_enabled_) return;
    float L    = 0.2126f * p->x + 0.7152f * p->y + 0.0722f * p->z;
    float outL = L;
    if (L <= params.highlights_k_) {
      // below knee_start: identity
      outL = L;
    } else if (L < 1.0f) {
      // inside the Hermite segment: parameterize t in [0,1]
      float t   = (L - params.highlights_k_) / params.highlights_dx_;
      // Hermite interpolation:
      float H00 = 2 * t * t * t - 3 * t * t + 1;
      float H10 = t * t * t - 2 * t * t + t;
      float H01 = -2 * t * t * t + 3 * t * t;
      float H11 = t * t * t - t * t;
      // note: tangents in Hermite are (dx * m0) and (dx * m1)
      outL = H00 * params.highlights_k_ + H10 * (params.highlights_dx_ * params.highlights_m0_) +
             H01 * 1.0f + H11 * (params.highlights_dx_ * params.highlights_m1_);
    } else {
      // L >= whitepoint: prefer a soft roll-off so "extreme highlights" compress more
      const float x  = L - 1.0f;
      const float m1 = params.highlights_m1_;

      if (m1 < 1.0f) {
        // Rational soft-clip:
        // outL = 1 + (m1*x) / (1 + b*x)
        // - slope at x=0 is m1 (continuous at whitepoint)
        // - as x->inf, outL asymptotes to 1 + m1/b (stronger "extreme highlight" suppression)
        const float rolloff_strength = 1.0f;  // tune: 0.5 weaker, 1.0 default, 2.0 stronger
        const float b                = fmaxf(1e-6f, (1.0f - m1) * rolloff_strength);
        outL                         = 1.0f + (m1 * x) / (1.0f + b * x);
      } else {
        // if boosting highlights, keep the linear extrapolation
        outL = 1.0f + x * m1;
      }
    }

    // avoid negative or NaN
    if (!isfinite(outL)) outL = L;
    // Preserve hue/chroma by scaling RGB by ratio outL/L (guard L==0)
    float scale = (L > 1e-8f) ? (outL / L) : 1.0f;
    p->x *= (scale);
    p->y *= (scale);
    p->z *= (scale);
  }
};

struct GPU_ShadowOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.shadows_enabled_) return;
    float L = 0.2126f * p->x + 0.7152f * p->y + 0.0722f * p->z;
    if (L < params.shadows_x1_) {
      float t    = (L - params.shadows_x0_) / params.shadows_dx_;
      float H00  = 2 * t * t * t - 3 * t * t + 1;
      float H10  = t * t * t - 2 * t * t + t;
      float H01  = -2 * t * t * t + 3 * t * t;
      float H11  = t * t * t - t * t;
      float outL = H00 * params.shadows_y0_ + H10 * (params.shadows_dx_ * params.shadows_m0_) +
                   H01 * params.shadows_y1_ + H11 * (params.shadows_dx_ * params.shadows_m1_);

      float scale = (L > 1e-8f) ? (outL / L) : 1.0f;
      p->x *= scale;
      p->y *= scale;
      p->z *= scale;
    }
  }
};

GPU_FUNC float evaluate_curve_hermite(float x, const GPUOperatorParams& params) {
  const int curve_count = params.curve_ctrl_pts_size_;
  if (curve_count <= 0) return x;
  if (curve_count == 1) {
    return fminf(fmaxf(params.curve_ctrl_pts_y_[0], 0.0f), 1.0f);
  }

  if (x <= params.curve_ctrl_pts_x_[0]) return params.curve_ctrl_pts_y_[0];
  if (x >= params.curve_ctrl_pts_x_[curve_count - 1]) return params.curve_ctrl_pts_y_[curve_count - 1];

  int idx = curve_count - 2;
  for (int i = 0; i < curve_count - 1; ++i) {
    if (x < params.curve_ctrl_pts_x_[i + 1]) {
      idx = i;
      break;
    }
  }

  const float dx = params.curve_h_[idx];
  if (fabsf(dx) <= 1e-8f) {
    return fminf(fmaxf(params.curve_ctrl_pts_y_[idx], 0.0f), 1.0f);
  }

  const float t   = (x - params.curve_ctrl_pts_x_[idx]) / dx;

  // Hermite interpolation
  const float h00 = 2.0f * t * t * t - 3.0f * t * t + 1.0f;
  const float h10 = t * t * t - 2.0f * t * t + t;
  const float h01 = -2.0f * t * t * t + 3.0f * t * t;
  const float h11 = t * t * t - t * t;

  const float y = h00 * params.curve_ctrl_pts_y_[idx] + h10 * dx * params.curve_m_[idx] +
                  h01 * params.curve_ctrl_pts_y_[idx + 1] + h11 * dx * params.curve_m_[idx + 1];
  return fminf(fmaxf(y, 0.0f), 1.0f);
}

struct GPU_CurveOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.curve_enabled_) return;
    if (params.curve_ctrl_pts_size_ <= 0) return;

    constexpr float kCurveInfluence = 0.65f;

    const float lum     = 0.2126f * p->x + 0.7152f * p->y + 0.0722f * p->z;
    const float mapped_lum = evaluate_curve_hermite(lum, params);
    const float new_lum    = lum + (mapped_lum - lum) * kCurveInfluence;
    const float ratio   = (lum > 1e-5f) ? new_lum / lum : 0.0f;
    p->x *= ratio;
    p->y *= ratio;
    p->z *= ratio;
  }
};
};  // namespace CUDA
};  // namespace puerhlab
