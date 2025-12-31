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

#include "edit/operators/wheel/color_wheel_op.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"
#include "utils/simd/simple_simd.hpp"

namespace puerhlab {

ColorWheelOp::ColorWheelOp()
    : _lift(), _gamma(), _gain(), _lift_crossover(0.25f), _gain_crossover(0.75f) {}

ColorWheelOp::ColorWheelOp(const nlohmann::json& params) { SetParams(params); }

void Clamp(cv::Mat& input) {
  cv::threshold(input, input, 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(input, input, 0.0f, 0.0f, cv::THRESH_TOZERO);
}

float bell(float L, float center, float width) {
  float x = (L - center) / width;
  return exp(-x * x);  // Gaussian
}

// FIXME： Migrate to ASC CDL style color grading
void ColorWheelOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();
  if (img.empty()) {
    throw std::invalid_argument("Color Wheel: Invalid input image");
  }

  // Get luminance graph
  cv::Mat img_Lab;
  cv::cvtColor(img, img_Lab, cv::COLOR_BGR2Lab);
  std::vector<cv::Mat> Lab_channels;
  cv::split(img_Lab, Lab_channels);
  cv::Mat   lightness = Lab_channels[0] / 100.0f;  // L

  // BGR
  cv::Vec3f lift_offset(_lift.color_offset.x + _lift.luminance_offset,
                        _lift.color_offset.y + _lift.luminance_offset,
                        _lift.color_offset.z + _lift.luminance_offset);
  cv::Vec3f gain_factor(_gain.color_offset.x + _gain.luminance_offset,
                        _gain.color_offset.y + _gain.luminance_offset,
                        _gain.color_offset.z + _gain.luminance_offset);
  cv::Vec3f gamma_inv(1.0f / (_gamma.color_offset.x + _gamma.luminance_offset),
                      1.0f / (_gamma.color_offset.y + _gamma.luminance_offset),
                      1.0f / (_gamma.color_offset.z + _gamma.luminance_offset));

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* pos) {
    float     L              = lightness.at<float>(pos[0], pos[1]);
    float     lift_w         = std::clamp(bell(L, 0.0f, 0.45f), 0.0f, 1.0f);
    float     gamma_w        = 1.0f;
    float     gain_w         = std::clamp(bell(L, 1.0f, 0.45f), 0.0f, 1.0f);

    // float total_w = lift_w + gamma_w + gain_w + 1e-6fff
    // lift_w /= total_w;
    // gamma_w = 1.0f;
    // gain_w /= total_w;

    cv::Vec3f original_pixel = pixel;
    cv::Vec3f lifted_pixel   = original_pixel + lift_offset;
    cv::Vec3f gained_pixel   = original_pixel.mul(gain_factor);
    cv::Vec3f gamma_pixel;
    gamma_pixel[0] = std::pow(original_pixel[0], gamma_inv[0]);
    gamma_pixel[1] = std::pow(original_pixel[1], gamma_inv[1]);
    gamma_pixel[2] = std::pow(original_pixel[2], gamma_inv[2]);
    pixel          = pixel + lift_w * (lifted_pixel - pixel) + gain_w * (gained_pixel - pixel) +
            gamma_w * (gamma_pixel - pixel);

    cv::saturate_cast<float>(pixel[0]);
    cv::saturate_cast<float>(pixel[1]);
    cv::saturate_cast<float>(pixel[2]);
  });
}

auto ColorWheelOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["lift"]       = _lift;
  inner["gamma"]      = _gamma;
  inner["gain"]       = _gain;
  inner["crossovers"] = {{"lift", _lift_crossover}, {"gain", _gain_crossover}};

  o[_script_name]     = inner;
  return o;
}

void ColorWheelOp::SetParams(const nlohmann::json& params) {
  nlohmann::json inner = params.at(_script_name);
  if (inner.contains("lift")) inner.at("lift").get_to(_lift);
  if (inner.contains("gamma")) inner.at("gamma").get_to(_gamma);
  if (inner.contains("gain")) inner.at("gain").get_to(_gain);
  if (inner.contains("crossovers")) {
    const auto& crossovers = inner.at("crossovers");
    if (crossovers.contains("lift")) crossovers.at("lift").get_to(_lift_crossover);
    if (crossovers.contains("gain")) crossovers.at("gain").get_to(_gain_crossover);
  }
}

void ColorWheelOp::SetGlobalParams(OperatorParams& params) const {
  params.lift_color_offset[0]  = _lift.color_offset.x;
  params.lift_color_offset[1]  = _lift.color_offset.y;
  params.lift_color_offset[2]  = _lift.color_offset.z;
  params.lift_luminance_offset = _lift.luminance_offset;

  params.gain_color_offset[0]  = _gain.color_offset.x;
  ;
  params.gain_color_offset[1]   = _gain.color_offset.y;
  params.gain_color_offset[2]   = _gain.color_offset.z;
  params.gain_luminance_offset  = _gain.luminance_offset;

  params.gamma_color_offset[0]  = _gamma.color_offset.x;
  params.gamma_color_offset[1]  = _gamma.color_offset.y;
  params.gamma_color_offset[2]  = _gamma.color_offset.z;
  params.gamma_luminance_offset = _gamma.luminance_offset;
}
};  // namespace puerhlab