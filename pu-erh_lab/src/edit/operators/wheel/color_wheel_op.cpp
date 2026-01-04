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

#include "image/image_buffer.hpp"

namespace puerhlab {

ColorWheelOp::ColorWheelOp()
    : lift_(), gamma_(), gain_(), lift_crossover_(0.25f), gain_crossover_(0.75f) {}

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
  cv::Vec3f lift_offset(lift_.color_offset_.x + lift_.luminance_offset_,
                        lift_.color_offset_.y + lift_.luminance_offset_,
                        lift_.color_offset_.z + lift_.luminance_offset_);
  cv::Vec3f gain_factor(gain_.color_offset_.x + gain_.luminance_offset_,
                        gain_.color_offset_.y + gain_.luminance_offset_,
                        gain_.color_offset_.z + gain_.luminance_offset_);
  cv::Vec3f gamma_inv(1.0f / (gamma_.color_offset_.x + gamma_.luminance_offset_),
                      1.0f / (gamma_.color_offset_.y + gamma_.luminance_offset_),
                      1.0f / (gamma_.color_offset_.z + gamma_.luminance_offset_));

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

  inner["lift"]       = lift_;
  inner["gamma"]      = gamma_;
  inner["gain"]       = gain_;
  inner["crossovers"] = {{"lift", lift_crossover_}, {"gain", gain_crossover_}};

  o[script_name_]     = inner;
  return o;
}

void ColorWheelOp::SetParams(const nlohmann::json& params) {
  nlohmann::json inner = params.at(script_name_);
  if (inner.contains("lift")) inner.at("lift").get_to(lift_);
  if (inner.contains("gamma")) inner.at("gamma").get_to(gamma_);
  if (inner.contains("gain")) inner.at("gain").get_to(gain_);
  if (inner.contains("crossovers")) {
    const auto& crossovers = inner.at("crossovers");
    if (crossovers.contains("lift")) crossovers.at("lift").get_to(lift_crossover_);
    if (crossovers.contains("gain")) crossovers.at("gain").get_to(gain_crossover_);
  }
}

void ColorWheelOp::SetGlobalParams(OperatorParams& params) const {
  params.lift_color_offset_[0]  = lift_.color_offset_.x;
  params.lift_color_offset_[1]  = lift_.color_offset_.y;
  params.lift_color_offset_[2]  = lift_.color_offset_.z;
  params.lift_luminance_offset_ = lift_.luminance_offset_;

  params.gain_color_offset_[0]  = gain_.color_offset_.x;
  ;
  params.gain_color_offset_[1]   = gain_.color_offset_.y;
  params.gain_color_offset_[2]   = gain_.color_offset_.z;
  params.gain_luminance_offset_  = gain_.luminance_offset_;

  params.gamma_color_offset_[0]  = gamma_.color_offset_.x;
  params.gamma_color_offset_[1]  = gamma_.color_offset_.y;
  params.gamma_color_offset_[2]  = gamma_.color_offset_.z;
  params.gamma_luminance_offset_ = gamma_.luminance_offset_;
}
};  // namespace puerhlab