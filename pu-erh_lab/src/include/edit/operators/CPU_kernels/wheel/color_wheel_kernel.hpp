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
#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct ColorWheelOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.color_wheel_enabled_) return;
    const float lum     = 0.2126f * p.r_ + 0.7152f * p.g_ + 0.0722f * p.b_;
    const float lift_w  = std::clamp(std::exp(-(lum * lum) / (0.45f * 0.45f)), 0.0f, 1.0f);
    const float gamma_w = 1.0f;
    const float gain_w =
        std::clamp(std::exp(-((lum - 1.0f) * (lum - 1.0f)) / (0.45f * 0.45f)), 0.0f, 1.0f);

    const float lift_x      = params.lift_color_offset_[0] + params.lift_luminance_offset_;
    const float lift_y      = params.lift_color_offset_[1] + params.lift_luminance_offset_;
    const float lift_z      = params.lift_color_offset_[2] + params.lift_luminance_offset_;

    const float gain_x      = params.gain_color_offset_[0] + params.gain_luminance_offset_;
    const float gain_y      = params.gain_color_offset_[1] + params.gain_luminance_offset_;
    const float gain_z      = params.gain_color_offset_[2] + params.gain_luminance_offset_;

    const float gamma_inv_x = 1.0f / (params.gamma_color_offset_[0] + params.gamma_luminance_offset_);
    const float gamma_inv_y = 1.0f / (params.gamma_color_offset_[1] + params.gamma_luminance_offset_);
    const float gamma_inv_z = 1.0f / (params.gamma_color_offset_[2] + params.gamma_luminance_offset_);

    const float gain_dx     = gain_x - 1.0f;
    const float gain_dy     = gain_y - 1.0f;
    const float gain_dz     = gain_z - 1.0f;

    const float r0 = p.r_, g0 = p.g_, b0 = p.b_;

    const float pr = std::pow(r0, gamma_inv_x);
    const float pg = std::pow(g0, gamma_inv_y);
    const float pb = std::pow(b0, gamma_inv_z);

    float       r  = r0 + lift_w * lift_x + gain_w * (r0 * gain_dx) + gamma_w * (pr - r0);
    float       g  = g0 + lift_w * lift_y + gain_w * (g0 * gain_dy) + gamma_w * (pg - g0);
    float       b  = b0 + lift_w * lift_z + gain_w * (b0 * gain_dz) + gamma_w * (pb - b0);

    p.r_            = std::clamp(r, 0.0f, 1.0f);
    p.g_            = std::clamp(g, 0.0f, 1.0f);
    p.b_            = std::clamp(b, 0.0f, 1.0f);
  }
};
};  // namespace puerhlab