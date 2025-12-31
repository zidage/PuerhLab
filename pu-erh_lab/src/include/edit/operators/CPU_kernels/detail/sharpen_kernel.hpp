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

#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct SharpenOpKernel : NeighborOpTag {
  inline void operator()(Tile& in, OperatorParams& params) const {
    if (!params.sharpen_enabled) return;
    cv::Mat tile_mat(in._height, in._width, CV_32FC4, in._ptr);
    cv::Mat blurred;
    cv::GaussianBlur(tile_mat, blurred, cv::Size(), params.sharpen_radius, params.sharpen_radius,
                     cv::BORDER_REPLICATE);
    cv::Mat high_pass = tile_mat - blurred;
    if (params.sharpen_threshold > 0.0f) {
      cv::Mat high_pass_gray;
      cv::cvtColor(high_pass, high_pass_gray, cv::COLOR_BGR2GRAY);
      cv::Mat abs_high_pass_gray = cv::abs(high_pass_gray);
      cv::Mat mask;
      cv::threshold(abs_high_pass_gray, mask, params.sharpen_threshold, 1.0f, cv::THRESH_BINARY);
      cv::Mat mask_3channel;
      cv::cvtColor(mask, mask_3channel, cv::COLOR_GRAY2BGR);
      cv::multiply(high_pass, mask_3channel, high_pass);
    }
    cv::scaleAdd(high_pass, params.sharpen_offset, tile_mat, tile_mat);
  }
};
};  // namespace puerhlab