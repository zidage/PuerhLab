//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <opencv2/core/cuda.hpp>

#include "decoders/processor/raw_processor_pattern.hpp"

namespace alcedo {
namespace CUDA {
struct RcdWorkspace {
  cv::cuda::GpuMat r;
  cv::cuda::GpuMat g;
  cv::cuda::GpuMat b;
  cv::cuda::GpuMat vh_dir;
  cv::cuda::GpuMat pq_dir;

  void Reserve(const cv::Size& size) {
    if (size.width <= 0 || size.height <= 0) {
      return;
    }
    r.create(size, CV_32FC1);
    g.create(size, CV_32FC1);
    b.create(size, CV_32FC1);
    vh_dir.create(size, CV_16UC1);
    pq_dir.create(size, CV_16UC1);
  }
};

void Bayer2x2ToRGB_RCD(cv::cuda::GpuMat& image, const BayerPattern2x2& pattern,
                       RcdWorkspace* workspace = nullptr, cv::cuda::Stream* stream = nullptr);

void Bayer2x2ToPlanarRGB_RCD(const cv::cuda::GpuMat& raw, const BayerPattern2x2& pattern,
                             RcdWorkspace* workspace = nullptr,
                             cv::cuda::Stream* stream = nullptr);
};
};  // namespace alcedo
