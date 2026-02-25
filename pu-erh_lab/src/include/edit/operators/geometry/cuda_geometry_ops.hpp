//  Copyright 2026 Yurun Zi
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

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>

namespace puerhlab {
namespace CUDA {

void ResizeAreaApprox(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Size dst_size);

void WarpAffineLinear(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Mat& matrix,
                      cv::Size out_size, const cv::Scalar& border_value);

}  // namespace CUDA
}  // namespace puerhlab

