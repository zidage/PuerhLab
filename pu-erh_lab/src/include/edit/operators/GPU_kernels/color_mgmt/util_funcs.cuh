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

// CUDA implementations of helper functions in Lib.Academy.DisplayEncoding.ctl
// Reference:
// https://github.com/aces-aswf/aces-core/blob/main/lib/Lib.Academy.DisplayEncoding.ctl

#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include "edit/operators/GPU_kernels/param.cuh"

namespace puerhlab {
namespace CUDA {
GPU_FUNC float3 mult_f3_f33(const float3& v, const float* m) {
  return make_float3(v.x * m[0] + v.y * m[3] + v.z * m[6], v.x * m[1] + v.y * m[4] + v.z * m[7],
                     v.x * m[2] + v.y * m[5] + v.z * m[8]);
}

GPU_FUNC float3 mult_f_f3(const float3& v, float s) {
  return make_float3(v.x * s, v.y * s, v.z * s);
}
GPU_FUNC float3 clamp_f3(const float3& v, float min_val, float max_val) {
  return make_float3(fminf(fmaxf(v.x, min_val), max_val), fminf(fmaxf(v.y, min_val), max_val),
                     fminf(fmaxf(v.z, min_val), max_val));
}
GPU_FUNC float3 pow_f3(const float3& v, float exp) {
  return make_float3(powf(v.x, exp), powf(v.y, exp), powf(v.z, exp));
}
}  // namespace CUDA
}  // namespace puerhlab