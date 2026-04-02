//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace puerhlab {
namespace CUDA {
// Port of CPU::ApplyColorMatrix (camera WB path only).
// Note: "user-specified" WB is intentionally not supported on the CUDA backend.
void ApplyColorMatrix(cv::cuda::GpuMat& img, const float rgb_cam[][4], const float* pre_mul,
                      const float* cam_mul, const float cam_xyz[][3],
                      cv::cuda::Stream* stream = nullptr);

void ApplyInverseCamMul(cv::cuda::GpuMat& img, const float* cam_mul,
                        cv::cuda::Stream* stream = nullptr);
};
};  // namespace puerhlab
