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

// CUDA implementations of helper functions in Lib.Academy.Tonescale.ctl
// Reference:
// https://github.com/aces-aswf/aces-core/blob/main/lib/Lib.Academy.Tonescale.ctl

#pragma once

#include <cuda_runtime.h>
#include "edit/operators/GPU_kernels/param.cuh"

namespace puerhlab {
namespace CUDA {

GPU_FUNC float Tonescale_fwd(float x, GPU_TSParams& params) {
  // Forward MM tone scale
  // Guard against 0/0 when x == -s_2_ and against negative denominators.
  const float denom = x + params.s_2_;
  const float ratio = (denom > 1e-7f) ? (fmaxf(0.f, x) / denom) : 0.0f;
  float f = params.m_2_ * powf(ratio, params.g_);

  float h = fmaxf(0.f, f * f / (f + params.t_1_));
  return h * params.n_r_;
}
}
}