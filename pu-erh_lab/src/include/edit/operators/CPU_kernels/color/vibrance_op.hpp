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
struct VibranceOpKernel : PointOpTag {
  inline void operator()(Pixel& in, OperatorParams& params) const {
    if (!params.vibrance_enabled) return;

    float r = in.r, g = in.g, b = in.b;

    float max_val  = std::max({r, g, b});
    float min_val  = std::min({r, g, b});
    float chroma   = max_val - min_val;

    // chroma in [0, max], vibrance_offset in [-100, 100]
    float strength = params.vibrance_offset / 100.0f;

    // Protect already highly saturated color
    float falloff  = std::exp(-3.0f * chroma);

    float scale    = 1.0f + strength * falloff;

    if (params.vibrance_offset >= 0.0f) {
      float luma = r * 0.299f + g * 0.587f + b * 0.114f;

      r          = luma + (r - luma) * scale;
      g          = luma + (g - luma) * scale;
      b          = luma + (b - luma) * scale;

    } else {
      float avg = (r + g + b) / 3.0f;
      r += (avg - r) * (1.0f - scale);
      g += (avg - g) * (1.0f - scale);
      b += (avg - b) * (1.0f - scale);
    }

    in.r = r;
    in.g = g;
    in.b = b;
  }
};
}  // namespace puerhlab