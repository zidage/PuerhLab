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
// CUDA implementations of color adjustment operators

#pragma once

#include <cuda_runtime.h>
#include <device_types.h>

#include "edit/operators/op_kernel.hpp"
#include "param.cuh"

namespace puerhlab {
namespace CUDA {
struct GPU_HLSOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.hls_enabled_) return;

    const float kEps = 1e-6f;
    auto WrapHue      = [](float h) -> float {
      h = fmodf(h, 360.0f);
      if (h < 0.0f) h += 360.0f;
      return h;
    };

    // Convert RGB to HLS.
    float r = fminf(fmaxf(p->x, 0.0f), 1.0f);
    float g = fminf(fmaxf(p->y, 0.0f), 1.0f);
    float b = fminf(fmaxf(p->z, 0.0f), 1.0f);

    float max_c = fmaxf(fmaxf(r, g), b);
    float min_c = fminf(fminf(r, g), b);
    float L     = (max_c + min_c) * 0.5f;
    float H = 0.0f;
    float S = 0.0f;
    float d = max_c - min_c;
    if (d > kEps) {
      const float denom = fmaxf(1.0f - fabsf(2.0f * L - 1.0f), kEps);
      S                 = fminf(fmaxf(d / denom, 0.0f), 1.0f);
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
    if (profile_count > OperatorParams::kHlsProfileCount) {
      profile_count = OperatorParams::kHlsProfileCount;
    }

    const float h = WrapHue(H);
    float       accum_h = 0.0f;
    float       accum_l = 0.0f;
    float       accum_s = 0.0f;
    bool        has_contribution = false;

#pragma unroll
    for (int i = 0; i < OperatorParams::kHlsProfileCount; ++i) {
      if (i >= profile_count) {
        continue;
      }

      const float adj_h = params.hls_profile_adjustments_[i][0];
      const float adj_l = params.hls_profile_adjustments_[i][1];
      const float adj_s = params.hls_profile_adjustments_[i][2];
      if (fabsf(adj_h) <= kEps && fabsf(adj_l) <= kEps && fabsf(adj_s) <= kEps) {
        continue;
      }

      const float hue_range = fmaxf(params.hls_profile_hue_ranges_[i], kEps);
      const float target_h  = WrapHue(params.hls_profile_hues_[i]);
      const float hue_diff  = fabsf(h - target_h);
      const float hue_dist  = fminf(hue_diff, 360.0f - hue_diff);
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
      return;
    }

    float h_adjusted = WrapHue(h + accum_h);
    float l_adjusted = fminf(fmaxf(L + accum_l, 0.0f), 1.0f);
    float s_adjusted = fminf(fmaxf(S + accum_s, 0.0f), 1.0f);
    // Convert HLS back to RGB
    if (s_adjusted <= kEps) {
      p->x = l_adjusted;
      p->y = l_adjusted;
      p->z = l_adjusted;
    } else {
      float q       = (l_adjusted < 0.5f) ? (l_adjusted * (1.0f + s_adjusted))
                                          : (l_adjusted + s_adjusted - l_adjusted * s_adjusted);
      float _p      = 2.0f * l_adjusted - q;

      auto  hue2rgb = [](float p, float q, float t) -> float {
        if (t < 0.0f) t += 1.0f;
        if (t > 1.0f) t -= 1.0f;
        if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
        if (t < 1.0f / 2.0f) return q;
        if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
        return p;
      };

      p->x = hue2rgb(_p, q, h_adjusted / 360.0f + 1.0f / 3.0f);
      p->y = hue2rgb(_p, q, h_adjusted / 360.0f);
      p->z = hue2rgb(_p, q, h_adjusted / 360.0f - 1.0f / 3.0f);
    }

    p->x = fminf(fmaxf(p->x, 0.0f), 1.0f);
    p->y = fminf(fmaxf(p->y, 0.0f), 1.0f);
    p->z = fminf(fmaxf(p->z, 0.0f), 1.0f);
  }
};

struct GPU_SaturationOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.saturation_enabled_) return;

    float luma = 0.2126f * p->x + 0.7152f * p->y + 0.0722f * p->z;
    p->x       = luma + (p->x - luma) * params.saturation_offset_;
    p->y       = luma + (p->y - luma) * params.saturation_offset_;
    p->z       = luma + (p->z - luma) * params.saturation_offset_;
  }
};

struct GPU_TintOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.tint_enabled_) return;

    p->y += params.tint_offset_;
  }
};

struct GPU_VibranceOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.vibrance_enabled_) return;

    float max_val  = fmaxf(fmaxf(p->x, p->y), p->z);
    float min_val  = fminf(fminf(p->x, p->y), p->z);
    float chroma   = max_val - min_val;

    // chroma in [0, max], vibrance_offset in [-100, 100]
    float strength = params.vibrance_offset_;

    // Protect already highly saturated color
    float falloff  = expf(-3.0f * chroma);

    float scale    = 1.0f + strength * falloff;

    if (params.vibrance_offset_ >= 0.0f) {
      float luma = p->x * 0.299f + p->y * 0.587f + p->z * 0.114f;

      p->x          = luma + (p->x - luma) * scale;
      p->y          = luma + (p->y - luma) * scale;
      p->z          = luma + (p->z - luma) * scale;

    } else {
      float avg = (p->x + p->y + p->z) / 3.0f;
      p->x += (avg - p->x) * (1.0f - scale);
      p->y += (avg - p->y) * (1.0f - scale);
      p->z += (avg - p->z) * (1.0f - scale);
    }
  }
};

struct GPU_ColorWheelOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.color_wheel_enabled_) return;

    constexpr float kEps = 1e-6f;

    const float offset_r = params.lift_color_offset_[0] + params.lift_luminance_offset_;
    const float offset_g = params.lift_color_offset_[1] + params.lift_luminance_offset_;
    const float offset_b = params.lift_color_offset_[2] + params.lift_luminance_offset_;

    const float slope_r  = fmaxf(params.gain_color_offset_[0] + params.gain_luminance_offset_, kEps);
    const float slope_g  = fmaxf(params.gain_color_offset_[1] + params.gain_luminance_offset_, kEps);
    const float slope_b  = fmaxf(params.gain_color_offset_[2] + params.gain_luminance_offset_, kEps);

    const float power_r  =
        fmaxf(params.gamma_color_offset_[0] + params.gamma_luminance_offset_, kEps);
    const float power_g  =
        fmaxf(params.gamma_color_offset_[1] + params.gamma_luminance_offset_, kEps);
    const float power_b  =
        fmaxf(params.gamma_color_offset_[2] + params.gamma_luminance_offset_, kEps);

    const float base_r = fmaxf(p->x * slope_r + offset_r, 0.0f);
    const float base_g = fmaxf(p->y * slope_g + offset_g, 0.0f);
    const float base_b = fmaxf(p->z * slope_b + offset_b, 0.0f);

    p->x               = fminf(fmaxf(powf(base_r, power_r), 0.0f), 1.0f);
    p->y               = fminf(fmaxf(powf(base_g, power_g), 0.0f), 1.0f);
    p->z               = fminf(fmaxf(powf(base_b, power_b), 0.0f), 1.0f);
  }
};
};  // namespace CUDA
};  // namespace puerhlab
