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
#include "edit/operators/op_kernel.hpp"

namespace puerhlab {
struct CurveOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.curve_enabled) return;
    float lum = 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b;
    float x   = lum;

    /* Evaluate Curve */
    if (x <= params.curve_ctrl_pts.front().x)
      x = params.curve_ctrl_pts.front().y;
    else if (x >= params.curve_ctrl_pts.back().x)
      x = params.curve_ctrl_pts.back().y;
    else {
      // Find segment
      int idx = 0;
      for (int i = 0; i < static_cast<int>(params.curve_ctrl_pts.size()) - 1; ++i) {
        if (x < params.curve_ctrl_pts[i + 1].x) {
          idx = i;
          break;
        }
      }

      float t   = (x - params.curve_ctrl_pts[idx].x) / params.curve_h[idx];

      // Hermite interpolation
      float h00 = (2 * t * t * t - 3 * t * t + 1);
      float h10 = (t * t * t - 2 * t * t + t);
      float h01 = (-2 * t * t * t + 3 * t * t);
      float h11 = (t * t * t - t * t);

      x = h00 * params.curve_ctrl_pts[idx].y + h10 * params.curve_h[idx] * params.curve_m[idx] +
          h01 * params.curve_ctrl_pts[idx + 1].y +
          h11 * params.curve_h[idx] * params.curve_m[idx + 1];
    }

    /* Apply the curve adjustment to the pixel's luminance */
    float ratio = (lum > 1e-5f) ? x / lum : 0.0f;
    p.r *= ratio;
    p.g *= ratio;
    p.b *= ratio;
  }
};
};  // namespace puerhlab
