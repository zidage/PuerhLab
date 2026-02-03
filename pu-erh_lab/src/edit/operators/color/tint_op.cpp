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

#include "edit/operators/color/tint_op.hpp"

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "edit/operators/operator_factory.hpp"
#include "json.hpp"

namespace puerhlab {

TintOp::TintOp() : tint_offset_(0.0f) { scale_ = 0.0f; }

TintOp::TintOp(float tint_offset) : tint_offset_(tint_offset) {
  // In OpenCV, the value of a channel lies between -127 to 127
  scale_ = tint_offset / 1000.0f;
}

TintOp::TintOp(const nlohmann::json& params) { SetParams(params); }

void TintOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat&             img = input->GetCPUData();
  std::vector<cv::Mat> bgr_channels;

  cv::split(img, bgr_channels);

  bgr_channels[1] += scale_;
  // Thresholding
  cv::threshold(bgr_channels[1], bgr_channels[1], 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(bgr_channels[1], bgr_channels[1], 0.0f, 0.0f, cv::THRESH_TOZERO);

  cv::merge(bgr_channels, img);
}



auto TintOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[script_name_] = tint_offset_;
  return o;
}

void TintOp::SetParams(const nlohmann::json& params) {
  if (params.contains(script_name_)) {
    tint_offset_ = params.at(script_name_).get<float>();
    scale_       = tint_offset_ / 5000.0f;
  } else {
    tint_offset_ = 0.0f;
    scale_       = 0.0f;
  }
}

void TintOp::SetGlobalParams(OperatorParams& params) const { params.tint_offset_ = scale_; }

void TintOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.tint_enabled_ = enable;
}
};  // namespace puerhlab