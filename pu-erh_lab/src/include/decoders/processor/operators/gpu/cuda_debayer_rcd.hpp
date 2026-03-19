//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <opencv2/core/cuda.hpp>

#include "decoders/processor/raw_processor_pattern.hpp"

namespace puerhlab {
namespace CUDA {
void Bayer2x2ToRGB_RCD(cv::cuda::GpuMat& image, const BayerPattern2x2& pattern);
};
};  // namespace puerhlab
