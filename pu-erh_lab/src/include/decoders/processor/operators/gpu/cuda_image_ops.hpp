//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <opencv2/core/cuda.hpp>

namespace puerhlab {
namespace CUDA {

void MergeRGB(const cv::cuda::GpuMat& red, const cv::cuda::GpuMat& green,
              const cv::cuda::GpuMat& blue, cv::cuda::GpuMat& dst,
              cv::cuda::Stream* stream = nullptr);

void RGBToRGBA(cv::cuda::GpuMat& img, cv::cuda::Stream* stream = nullptr);

}  // namespace CUDA
}  // namespace puerhlab

