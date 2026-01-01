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
#include "image/image_buffer.hpp"

namespace puerhlab {
struct ClarityOpKernel : NeighborOpTag {
  inline void operator()(Tile& in, OperatorParams& params) const {
    if (!params.clarity_enabled_) return;
    cv::Mat tile_mat(in.height_, in.width_, CV_32FC4, in.ptr_);
    cv::Mat midtone_mask;

    cv::Mat luminosity_mask;
    cv::cvtColor(tile_mat, luminosity_mask, cv::COLOR_BGR2GRAY);

    // Apply a "U" shape curve
    luminosity_mask = luminosity_mask - 0.5f;
    luminosity_mask = luminosity_mask * 2.0f;
    cv::pow(luminosity_mask, 2.0, luminosity_mask);
    midtone_mask = 1.0f - luminosity_mask;
    cv::Mat blurred;
    // Reflect padding keeps gradients continuous across tile borders when halos are stitched
    cv::GaussianBlur(tile_mat, blurred, cv::Size(), 5.0f, 5.0f, cv::BORDER_REFLECT101);
    cv::Mat    high_pass  = tile_mat - blurred;
    const bool continuous = high_pass.isContinuous() && midtone_mask.isContinuous();
    const int  rows       = high_pass.rows;
    const int  cols       = high_pass.cols;

    if (continuous) {
      const int total    = rows * cols;
      auto*     hp_ptr   = high_pass.ptr<cv::Vec4f>();
      auto*     mask_ptr = midtone_mask.ptr<float>();
      for (int i = 0; i < total; ++i) {
        const float w = mask_ptr[i] * params.clarity_offset_;
        hp_ptr[i][0] *= w;
        hp_ptr[i][1] *= w;
        hp_ptr[i][2] *= w;  // leave alpha untouched
      }
    } else {
      for (int r = 0; r < rows; ++r) {
        auto*        hp_ptr = high_pass.ptr<cv::Vec4f>(r);
        const float* m      = midtone_mask.ptr<float>(r);
        for (int c = 0; c < cols; ++c) {
          const float w = m[c] * params.clarity_offset_;
          hp_ptr[c][0] *= w;
          hp_ptr[c][1] *= w;
          hp_ptr[c][2] *= w;  // leave alpha untouched
        }
      }
    }
    tile_mat += high_pass;
  }
};
}  // namespace puerhlab