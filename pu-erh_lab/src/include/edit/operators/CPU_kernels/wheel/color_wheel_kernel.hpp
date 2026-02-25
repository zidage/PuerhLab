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
struct ColorWheelOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.color_wheel_enabled_) return;
    constexpr float kEps = 1e-6f;

    const float offset_r = params.lift_color_offset_[0] + params.lift_luminance_offset_;
    const float offset_g = params.lift_color_offset_[1] + params.lift_luminance_offset_;
    const float offset_b = params.lift_color_offset_[2] + params.lift_luminance_offset_;

    const float slope_r  = std::max(params.gain_color_offset_[0] + params.gain_luminance_offset_, kEps);
    const float slope_g  = std::max(params.gain_color_offset_[1] + params.gain_luminance_offset_, kEps);
    const float slope_b  = std::max(params.gain_color_offset_[2] + params.gain_luminance_offset_, kEps);

    const float power_r  =
        std::max(params.gamma_color_offset_[0] + params.gamma_luminance_offset_, kEps);
    const float power_g  =
        std::max(params.gamma_color_offset_[1] + params.gamma_luminance_offset_, kEps);
    const float power_b  =
        std::max(params.gamma_color_offset_[2] + params.gamma_luminance_offset_, kEps);

    const float base_r = std::max(p.r_ * slope_r + offset_r, 0.0f);
    const float base_g = std::max(p.g_ * slope_g + offset_g, 0.0f);
    const float base_b = std::max(p.b_ * slope_b + offset_b, 0.0f);

    p.r_               = std::clamp(std::pow(base_r, power_r), 0.0f, 1.0f);
    p.g_               = std::clamp(std::pow(base_g, power_g), 0.0f, 1.0f);
    p.b_               = std::clamp(std::pow(base_b, power_b), 0.0f, 1.0f);
  }
};
};  // namespace puerhlab
