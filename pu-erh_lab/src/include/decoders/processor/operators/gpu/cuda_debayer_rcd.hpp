//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <opencv2/core/cuda.hpp>

namespace puerhlab {
namespace CUDA {
void BayerRGGB2RGB_RCD(cv::cuda::GpuMat& image);
};
};  // namespace puerhlab
