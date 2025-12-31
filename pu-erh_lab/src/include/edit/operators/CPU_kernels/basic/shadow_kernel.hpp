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

#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {

struct ShadowsOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.shadows_enabled) return;
    float L = 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b;
    if (L <= params.shadows_x0) {
      p.r = 0.0f;
      p.g = 0.0f;
      p.b = 0.0f;
    } else if (L < params.shadows_x1) {
      float t    = (L - params.shadows_x0) / params.shadows_dx;
      float H00  = 2 * t * t * t - 3 * t * t + 1;
      float H10  = t * t * t - 2 * t * t + t;
      float H01  = -2 * t * t * t + 3 * t * t;
      float H11  = t * t * t - t * t;
      float outL = H00 * params.shadows_y0 + H10 * (params.shadows_dx * params.shadows_m0) +
                   H01 * params.shadows_y1 + H11 * (params.shadows_dx * params.shadows_m1);

      if (!std::isfinite(outL)) outL = L;
      float scale = (L > 1e-8f) ? (outL / L) : 1.0f;
      p.r *= scale;
      p.g *= scale;
      p.b *= scale;
    }
  }
};
}  // namespace puerhlab