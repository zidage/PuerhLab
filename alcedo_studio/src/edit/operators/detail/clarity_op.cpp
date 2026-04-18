//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "edit/operators/detail/clarity_op.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core/hal/interface.h>

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "edit/operators/curve/curve_op.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"

namespace alcedo {
namespace {

void BuildGaussianKernel(float sigma, int max_radius, int& tap_count,
                         float (&weights)[OperatorParams::kDetailMaxGaussianTapCount]) {
  std::fill_n(weights, OperatorParams::kDetailMaxGaussianTapCount, 0.0f);
  tap_count = 0;

  if (sigma <= 0.0f) {
    return;
  }

  const float safe_sigma = std::max(sigma, 1.0e-4f);
  const int   radius =
      std::clamp(static_cast<int>(std::ceil(3.0f * safe_sigma)), 1, max_radius);
  tap_count = std::min(radius + 1, OperatorParams::kDetailMaxGaussianTapCount);

  const double inv2sigma2  = 0.5 / (static_cast<double>(safe_sigma) * safe_sigma);
  double       full_weight = 1.0;
  weights[0]               = 1.0f;
  for (int tap = 1; tap < tap_count; ++tap) {
    const double w = std::exp(-(static_cast<double>(tap) * static_cast<double>(tap)) * inv2sigma2);
    weights[tap]   = static_cast<float>(w);
    full_weight += 2.0 * w;
  }

  if (full_weight > 0.0) {
    for (int tap = 0; tap < tap_count; ++tap) {
      weights[tap] = static_cast<float>(static_cast<double>(weights[tap]) / full_weight);
    }
  }
}

}  // namespace

float ClarityOp::usm_radius_ = 5.0f;

ClarityOp::ClarityOp() : clarity_offset_(0) { scale_ = 1.0f; }
ClarityOp::ClarityOp(float clarity_offset) : clarity_offset_(clarity_offset) {
  scale_ = clarity_offset / 300.0f;
}

ClarityOp::ClarityOp(const nlohmann::json& params) { SetParams(params); }

void ClarityOp::CreateMidtoneMask(cv::Mat& input, cv::Mat& mask) const {
  cv::Mat luminosity_mask;
  cv::cvtColor(input, luminosity_mask, cv::COLOR_BGR2GRAY);

  // Apply a "U" shape curve
  luminosity_mask = luminosity_mask - 0.5f;
  luminosity_mask = luminosity_mask * 2.0f;
  cv::pow(luminosity_mask, 2.0, luminosity_mask);
  mask = 1.0f - luminosity_mask;

  // if (_blur_sigma > 0) {
  //   cv::GaussianBlur(mask, mask, cv::Size(), _blur_sigma, _blur_sigma);
  // }
}

void ClarityOp::Apply(std::shared_ptr<ImageBuffer> input) {
  // Adpated from
  // https://community.adobe.com/t5/photoshop-ecosystem-discussions/what-exactly-is-clarity/td-p/8957968
  cv::Mat& img = input->GetCPUData();

  cv::Mat  midtone_mask;
  CreateMidtoneMask(img, midtone_mask);

  cv::Mat blurred;
  // Use reflect padding to avoid brightness seams at tile boundaries
  cv::GaussianBlur(img, blurred, cv::Size(), usm_radius_, usm_radius_, cv::BORDER_REFLECT101);

  cv::Mat    high_pass  = img - blurred;

  const bool continuous = high_pass.isContinuous() && midtone_mask.isContinuous();
  const int  rows       = high_pass.rows;
  const int  cols       = high_pass.cols;

  if (continuous) {
    const int total    = rows * cols;
    auto*     hp_ptr   = high_pass.ptr<cv::Vec3f>();
    auto*     mask_ptr = midtone_mask.ptr<float>();
    for (int i = 0; i < total; ++i) {
      const float w = mask_ptr[i] * scale_;
      hp_ptr[i][0] *= w;
      hp_ptr[i][1] *= w;
      hp_ptr[i][2] *= w;
    }
  } else {
    for (int r = 0; r < rows; ++r) {
      auto*        hp_ptr = high_pass.ptr<cv::Vec3f>(r);
      const float* m      = midtone_mask.ptr<float>(r);
      for (int c = 0; c < cols; ++c) {
        const float w = m[c] * scale_;
        hp_ptr[c][0] *= w;
        hp_ptr[c][1] *= w;
        hp_ptr[c][2] *= w;
      }
    }
  }

  img += high_pass;
}

void ClarityOp::ApplyGPU(std::shared_ptr<ImageBuffer>) {
  // GPU implementation not provided
  throw std::runtime_error("ClarityOp: ApplyGPU not implemented");
}

auto ClarityOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[script_name_] = clarity_offset_;

  return o;
}

void ClarityOp::SetParams(const nlohmann::json& params) {
  if (params.contains(script_name_)) {
    clarity_offset_ = params[script_name_];
  } else {
    clarity_offset_ = 0.0f;
  }
  scale_ = clarity_offset_ / 300.0f;
}

void ClarityOp::SetGlobalParams(OperatorParams& params) const {
  params.clarity_offset_ = scale_;
  params.clarity_radius_ = usm_radius_;
  BuildGaussianKernel(usm_radius_, 20, params.clarity_gaussian_tap_count_,
                      params.clarity_gaussian_weights_);
}

void ClarityOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.clarity_enabled_ = enable;
}
};  // namespace alcedo
