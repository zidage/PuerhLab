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

#include "edit/operators/basic/contrast_op.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core/hal/interface.h>

#include <opencv2/core.hpp>
#include <stdexcept>

#include "edit/operators/color/conversion/Oklab_cvt.hpp"
#include "edit/operators/operator_factory.hpp"

namespace puerhlab {

/**
 * @brief Default construct a new Contrast Op:: Contrast Op object
 *
 */
namespace {
inline float contrast_scale_from_slider(float v) {
  const float s = std::clamp(v, -100.0f, 100.0f) / 100.0f;
  return std::exp(s);
}

inline float contrast_k_from_slider(float v) {
  const float s = std::clamp(v, -100.0f, 100.0f) / 100.0f;
  if (std::abs(s) <= 1e-6f) {
    return 0.0f;
  }
  return 4.0f * s * std::abs(s);
}
}  // namespace

ContrastOp::ContrastOp() : contrast_offset_(0.0f) {
  scale_          = 1.0f;
  gpu_strength_   = 0.0f;
  slider_enabled_ = false;
}

/**
 * @brief Construct a new Contrast Op:: Contrast Op object
 *
 * @param contrast_offset
 */
ContrastOp::ContrastOp(float contrast_offset) : contrast_offset_(contrast_offset) {
  scale_          = contrast_scale_from_slider(contrast_offset_);
  gpu_strength_   = contrast_k_from_slider(contrast_offset_);
  slider_enabled_ = std::abs(gpu_strength_) > 1e-6f;
}

ContrastOp::ContrastOp(const nlohmann::json& params) { SetParams(params); }

/**
 * @brief Apply the contrast adjustment
 *
 * @param input
 * @return ImageBuffer
 */
void ContrastOp::Apply(std::shared_ptr<ImageBuffer> input) {
  if (!runtime_enabled_ || !slider_enabled_ || std::abs(scale_ - 1.0f) <= 1e-6f) {
    return;
  }

  cv::Mat& linear_image = input->GetCPUData();

  linear_image.forEach<cv::Vec3f>([this](cv::Vec3f& pixel, const int*) -> void {
    auto lab = OklabCvt::ACESRGB2Oklab(pixel);
    lab.l_    = (lab.l_ - 0.5f) * scale_ + 0.5f;
    pixel    = OklabCvt::Oklab2ACESRGB(lab);
  });

  // linear_image          = (linear_image - 0.5f) * _scale + 0.5f;
  // clamp
  // cv::min(linear_image, 100.0f, linear_image);
  // cv::max(linear_image, 0.0f, linear_image);
}

void ContrastOp::ApplyGPU(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error("ContrastOp: ApplyGPU not implemented");
}

auto ContrastOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[GetScriptName()] = contrast_offset_;
  return o;
}

void ContrastOp::SetParams(const nlohmann::json& params) {
  if (params.contains(GetScriptName())) {
    contrast_offset_ = params[GetScriptName()];
  } else {
    contrast_offset_ = 0.0f;
  }
  scale_          = contrast_scale_from_slider(contrast_offset_);
  gpu_strength_   = contrast_k_from_slider(contrast_offset_);
  slider_enabled_ = std::abs(gpu_strength_) > 1e-6f;
}

void ContrastOp::SetGlobalParams(OperatorParams& params) const {
  params.contrast_enabled_ = runtime_enabled_ && slider_enabled_;
  params.contrast_scale_   = params.contrast_enabled_ ? gpu_strength_ : 0.0f;
}

void ContrastOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  runtime_enabled_         = enable;
  params.contrast_enabled_ = runtime_enabled_ && slider_enabled_;
  params.contrast_scale_   = params.contrast_enabled_ ? gpu_strength_ : 0.0f;
}
};  // namespace puerhlab
