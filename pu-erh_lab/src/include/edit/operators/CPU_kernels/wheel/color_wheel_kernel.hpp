#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct ColorWheelOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    const float lum     = 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b;
    const float lift_w  = std::clamp(std::exp(-(lum * lum) / (0.45f * 0.45f)), 0.0f, 1.0f);
    const float gamma_w = 1.0f;
    const float gain_w =
        std::clamp(std::exp(-((lum - 1.0f) * (lum - 1.0f)) / (0.45f * 0.45f)), 0.0f, 1.0f);

    const float lift_x      = params.lift_color_offset[0] + params.lift_luminance_offset;
    const float lift_y      = params.lift_color_offset[1] + params.lift_luminance_offset;
    const float lift_z      = params.lift_color_offset[2] + params.lift_luminance_offset;

    const float gain_x      = params.gain_color_offset[0] + params.gain_luminance_offset;
    const float gain_y      = params.gain_color_offset[1] + params.gain_luminance_offset;
    const float gain_z      = params.gain_color_offset[2] + params.gain_luminance_offset;

    const float gamma_inv_x = 1.0f / (params.gamma_color_offset[0] + params.gamma_luminance_offset);
    const float gamma_inv_y = 1.0f / (params.gamma_color_offset[1] + params.gamma_luminance_offset);
    const float gamma_inv_z = 1.0f / (params.gamma_color_offset[2] + params.gamma_luminance_offset);

    const float gain_dx     = gain_x - 1.0f;
    const float gain_dy     = gain_y - 1.0f;
    const float gain_dz     = gain_z - 1.0f;

    const float r0 = p.r, g0 = p.g, b0 = p.b;

    const float pr = std::pow(r0, gamma_inv_x);
    const float pg = std::pow(g0, gamma_inv_y);
    const float pb = std::pow(b0, gamma_inv_z);

    float       r  = r0 + lift_w * lift_x + gain_w * (r0 * gain_dx) + gamma_w * (pr - r0);
    float       g  = g0 + lift_w * lift_y + gain_w * (g0 * gain_dy) + gamma_w * (pg - g0);
    float       b  = b0 + lift_w * lift_z + gain_w * (b0 * gain_dz) + gamma_w * (pb - b0);

    p.r           = cv::saturate_cast<float>(r);
    p.g           = cv::saturate_cast<float>(g);
    p.b           = cv::saturate_cast<float>(b);
  }
};
};  // namespace puerhlab