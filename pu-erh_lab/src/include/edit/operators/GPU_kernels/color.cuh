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
  __device__ __forceinline__ inline void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.hls_enabled) return;

    // Convert RGB to HLS
    float r = p->x, g = p->y, b = p->z;
    float max_c = fmaxf(fmaxf(r, g), b);
    float min_c = fminf(fminf(r, g), b);
    float L     = (max_c + min_c) * 0.5f;
    float H = 0.0f, S = 0.0f;
    float d = max_c - min_c == 0.0f ? 1e-10f : max_c - min_c;

    S       = (L < 0.5f) ? (d / (max_c + min_c)) : (d / (2.0f - max_c - min_c));
    if (max_c == r) {
      H = (g - b) / d + (g < b ? 6.0f : 0.0f);
    } else if (max_c == g) {
      H = (b - r) / d + 2.0f;
    } else if (max_c == b) {
      H = (r - g) / d + 4.0f;
    }
    H *= 60.0f;

    float target_h = params.target_hls[0];
    float target_l = params.target_hls[1];
    float target_s = params.target_hls[2];

    // Compute mask
    float h        = H;
    float l        = L;
    float s        = S;
    float hue_diff = fabsf(h - target_h);
    float hue_dist = fminf(hue_diff, 360.0f - hue_diff);

    float weight =
        fmaxf(0.0f, 1.0f - hue_dist / params.hue_range) *                      // hue_w
        fmaxf(0.0f, 1.0f - fabsf(l - target_l) / params.lightness_range) *  // lightness_w
        fmaxf(0.0f, 1.0f - fabsf(s - target_s) / params.saturation_range);  // saturation_w

    float adj_h      = params.hls_adjustment[0];
    float adj_l      = params.hls_adjustment[1];
    float adj_s      = params.hls_adjustment[2];

    float h_adjusted = fmodf(h + adj_h * weight, 360.0f);
    if (h_adjusted < 0) h_adjusted += 360.0f;

    float l_adjusted = fminf(fmaxf(l + adj_l * weight, 0.0f), 1.0f);
    float s_adjusted = fminf(fmaxf(s + adj_s * weight, 0.0f), 1.0f);
    // Convert HLS back to RGB
    if (s_adjusted == 0.0f) {
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
  }
};

struct GPU_SaturationOpKernel : GPUPointOpTag {
  __device__ __forceinline__ inline void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.saturation_enabled) return;

    float luma = 0.2126f * p->x + 0.7152f * p->y + 0.0722f * p->z;
    p->x       = luma + (p->x - luma) * params.saturation_offset;
    p->y       = luma + (p->y - luma) * params.saturation_offset;
    p->z       = luma + (p->z - luma) * params.saturation_offset;
  }
};

struct GPU_TintOpKernel : GPUPointOpTag {
  __device__ __forceinline__ inline void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.tint_enabled) return;

    p->y += params.tint_offset;
  }
};

struct GPU_VibranceOpKernel : GPUPointOpTag {
  __device__ __forceinline__ inline void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.vibrance_enabled) return;

    float max_val  = fmaxf(fmaxf(p->x, p->y), p->z);
    float min_val  = fminf(fminf(p->x, p->y), p->z);
    float chroma   = max_val - min_val;

    // chroma in [0, max], vibrance_offset in [-100, 100]
    float strength = params.vibrance_offset / 100.0f;

    // Protect already highly saturated color
    float falloff  = expf(-3.0f * chroma);

    float scale    = 1.0f + strength * falloff;

    if (params.vibrance_offset >= 0.0f) {
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
};  // namespace CUDA
};  // namespace puerhlab