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

struct GPU_BlackOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
      if (!params.black_enabled_) return;
      p->x = p->x * params.slope_ + params.black_point_;
      p->y = p->y * params.slope_ + params.black_point_;
      p->z = p->z * params.slope_ + params.black_point_;
  }
};

struct GPU_WhiteOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
      if (!params.white_enabled_) return;
      p->x = p->x * params.slope_ + params.black_point_;
      p->y = p->y * params.slope_ + params.black_point_;
      p->z = p->z * params.slope_ + params.black_point_;
  }
};

struct GPU_ContrastOpKernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.contrast_enabled_) return;
    p->x = (p->x - 0.05707762557f) * params.contrast_scale_ + 0.05707762557f;  // 1 stop = 1/17.52
    p->y = (p->y - 0.05707762557f) * params.contrast_scale_ + 0.05707762557f;
    p->z = (p->z - 0.05707762557f) * params.contrast_scale_ + 0.05707762557f;
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
      outL      = H00 * params.highlights_k_ + H10 * (params.highlights_dx_ * params.highlights_m0_) +
             H01 * 1.0f + H11 * (params.highlights_dx_ * params.highlights_m1_);
    } else {
      // L >= whitepoint: linear extrapolate using slope m1
      outL = 1.0f + (L - 1.0f) * params.highlights_m1_;
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
}; // namespace CUDA
}; // namespace puerhlab