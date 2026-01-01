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

#include "edit/operators/cst/cv_cvt_op.hpp"

namespace puerhlab {
CVCvtColorOp::CVCvtColorOp(int code, std::optional<size_t> channel_index)
    : code_(code), channel_index_(channel_index) {}

CVCvtColorOp::CVCvtColorOp(const nlohmann::json& params) { SetParams(params); }

void CVCvtColorOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();
  cv::UMat src;
  img.copyTo(src);

  if (img.empty()) {
    throw std::runtime_error("CVCvtColorOp: Input image is empty");
  }
  cv::cvtColor(src, src, code_);

  cv::Mat dst;
  src.copyTo(dst);

  if (channel_index_.has_value()) {
    std::vector<cv::Mat> channels;
    cv::split(dst, channels);
    img = channels.at(channel_index_.value());
  }
}
};  // namespace puerhlab