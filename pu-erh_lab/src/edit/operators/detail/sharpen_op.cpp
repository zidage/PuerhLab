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

#include "edit/operators/detail/sharpen_op.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {
SharpenOp::SharpenOp(float offset, float radius, float threshold)
    : offset_(offset), radius_(radius), threshold_(threshold) {
  ComputeScale();
  threshold_ /= 100.0f;
}

SharpenOp::SharpenOp(const nlohmann::json& params) { SetParams(params); }

void SharpenOp::ComputeScale() { scale_ = offset_ / 100.0f; }

auto SharpenOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["offset"]    = offset_;
  inner["radius"]    = radius_;
  inner["threshold"] = threshold_;

  o[script_name_]    = inner;
  return o;
}

void SharpenOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(script_name_)) return;
  nlohmann::json inner = params[script_name_].get<nlohmann::json>();
  if (inner.contains("offset")) {
    offset_ = inner["offset"].get<float>();
  }
  if (inner.contains("radius")) {
    radius_ = inner["radius"].get<float>();
  } else {
    radius_ = 3.0f;
  }
  if (inner.contains("threshold")) {
    threshold_ = inner["threshold"].get<float>();
    threshold_ /= 100.0f;
  }
  ComputeScale();
}

void SharpenOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();

  // Use USM to sharpen the image
  cv::Mat  blurred;
  cv::GaussianBlur(img, blurred, cv::Size(), radius_, radius_, cv::BORDER_REPLICATE);

  cv::Mat high_pass = img - blurred;
  if (threshold_ > 0.0f) {
    cv::Mat high_pass_gray;
    cv::cvtColor(high_pass, high_pass_gray, cv::COLOR_BGR2GRAY);

    cv::Mat abs_high_pass_gray = cv::abs(high_pass_gray);

    cv::Mat mask;
    cv::threshold(abs_high_pass_gray, mask, threshold_, 1.0f, cv::THRESH_BINARY);

    cv::Mat mask_3channel;
    cv::cvtColor(mask, mask_3channel, cv::COLOR_GRAY2BGR);
    cv::multiply(high_pass, mask_3channel, high_pass);
  }

  cv::scaleAdd(high_pass, scale_, img, img);
  cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
}


void SharpenOp::SetGlobalParams(OperatorParams& params) const {
  params.sharpen_offset_    = scale_;
  params.sharpen_radius_    = radius_;
  params.sharpen_threshold_ = threshold_;
}

void SharpenOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.sharpen_enabled_ = enable;
}
};  // namespace puerhlab