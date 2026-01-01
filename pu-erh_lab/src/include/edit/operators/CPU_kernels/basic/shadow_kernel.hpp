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
    if (!params.shadows_enabled_) return;
    float L = 0.2126f * p.r_ + 0.7152f * p.g_ + 0.0722f * p.b_;
    if (L <= params.shadows_x0_) {
      p.r_ = 0.0f;
      p.g_ = 0.0f;
      p.b_ = 0.0f;
    } else if (L < params.shadows_x1_) {
      float t    = (L - params.shadows_x0_) / params.shadows_dx_;
      float H00  = 2 * t * t * t - 3 * t * t + 1;
      float H10  = t * t * t - 2 * t * t + t;
      float H01  = -2 * t * t * t + 3 * t * t;
      float H11  = t * t * t - t * t;
      float outL = H00 * params.shadows_y0_ + H10 * (params.shadows_dx_ * params.shadows_m0_) +
                   H01 * params.shadows_y1_ + H11 * (params.shadows_dx_ * params.shadows_m1_);

      if (!std::isfinite(outL)) outL = L;
      float scale = (L > 1e-8f) ? (outL / L) : 1.0f;
      p.r_ *= scale;
      p.g_ *= scale;
      p.b_ *= scale;
    }
  }
};
}  // namespace puerhlab