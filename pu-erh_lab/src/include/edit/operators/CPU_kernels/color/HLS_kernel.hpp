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

#include <algorithm>
#include <cmath>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

struct HLSOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.hls_enabled_) return;

    // Convert RGB to HLS
    float r = p.r_, g = p.g_, b = p.b_;
    float max_c = std::max({r, g, b});
    float min_c = std::min({r, g, b});
    float L     = (max_c + min_c) * 0.5f;
    float H = 0.0f;
    float S = 0.0f;
    float d = max_c - min_c;
    if (d > 1e-6f) {
      const float denom = std::max(1.0f - std::abs(2.0f * L - 1.0f), 1e-6f);
      S                 = std::clamp(d / denom, 0.0f, 1.0f);
      if (max_c == r) {
        H = (g - b) / d + (g < b ? 6.0f : 0.0f);
      } else if (max_c == g) {
        H = (b - r) / d + 2.0f;
      } else {
        H = (r - g) / d + 4.0f;
      }
      H *= 60.0f;
    }

    const auto WrapHue = [](float hue) -> float {
      hue = std::fmod(hue, 360.0f);
      if (hue < 0.0f) {
        hue += 360.0f;
      }
      return hue;
    };

    const int profile_count = std::clamp(params.hls_profile_count_, 1, OperatorParams::kHlsProfileCount);

    float      best_dist    = 360.0f;
    int        best_idx     = 0;
    const float h           = WrapHue(H);
    for (int i = 0; i < profile_count; ++i) {
      const float target_h = WrapHue(params.hls_profile_hues_[i]);
      const float hue_diff = std::abs(h - target_h);
      const float hue_dist = std::min(hue_diff, 360.0f - hue_diff);
      if (hue_dist < best_dist) {
        best_dist = hue_dist;
        best_idx  = i;
      }
    }

    const float safe_hue_range = std::max(params.hls_profile_hue_ranges_[best_idx], 1e-6f);
    const float weight         = std::max(0.0f, 1.0f - best_dist / safe_hue_range);
    if (weight <= 0.0f) {
      return;
    }

    const float adj_h = params.hls_profile_adjustments_[best_idx][0];
    const float adj_l = params.hls_profile_adjustments_[best_idx][1];
    const float adj_s = params.hls_profile_adjustments_[best_idx][2];
    if (std::abs(adj_h) <= 1e-6f && std::abs(adj_l) <= 1e-6f && std::abs(adj_s) <= 1e-6f) {
      return;
    }

    float l_adjusted = std::clamp(L + adj_l * weight, 0.0f, 1.0f);
    float s_adjusted = std::clamp(S + adj_s * weight, 0.0f, 1.0f);

    float h_adjusted = std::fmod(h + adj_h * weight, 360.0f);
    if (h_adjusted < 0) h_adjusted += 360.0f;

    // Convert HLS back to RGB
    if (s_adjusted == 0.0f) {
      p.r_ = l_adjusted;
      p.g_ = l_adjusted;
      p.b_ = l_adjusted;
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

      p.r_ = hue2rgb(_p, q, h_adjusted / 360.0f + 1.0f / 3.0f);
      p.g_ = hue2rgb(_p, q, h_adjusted / 360.0f);
      p.b_ = hue2rgb(_p, q, h_adjusted / 360.0f - 1.0f / 3.0f);
    }
  }
};
};  // namespace puerhlab
