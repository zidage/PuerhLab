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

struct HLSOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.hls_enabled) return;

    // Convert RGB to HLS
    float r = p.r, g = p.g, b = p.b;
    float max_c = std::max({r, g, b});
    float min_c = std::min({r, g, b});
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
    float hue_diff = std::abs(h - target_h);
    float hue_dist = std::min(hue_diff, 360.0f - hue_diff);

    float weight =
        std::max(0.0f, 1.0f - hue_dist / params.hue_range) *                      // hue_w
        std::max(0.0f, 1.0f - std::abs(l - target_l) / params.lightness_range) *  // lightness_w
        std::max(0.0f, 1.0f - std::abs(s - target_s) / params.saturation_range);  // saturation_w

    float adj_h      = params.hls_adjustment[0];
    float adj_l      = params.hls_adjustment[1];
    float adj_s      = params.hls_adjustment[2];

    float h_adjusted = std::fmod(h + adj_h * weight, 360.0f);
    if (h_adjusted < 0) h_adjusted += 360.0f;

    float l_adjusted = std::clamp(l + adj_l * weight, 0.0f, 1.0f);
    float s_adjusted = std::clamp(s + adj_s * weight, 0.0f, 1.0f);

    // Convert HLS back to RGB
    if (s_adjusted == 0.0f) {
      p.r = l_adjusted;
      p.g = l_adjusted;
      p.b = l_adjusted;
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

      p.r = hue2rgb(_p, q, h_adjusted / 360.0f + 1.0f / 3.0f);
      ;
      p.g = hue2rgb(_p, q, h_adjusted / 360.0f);
      p.b = hue2rgb(_p, q, h_adjusted / 360.0f - 1.0f / 3.0f);
    }
  }
};
};  // namespace puerhlab