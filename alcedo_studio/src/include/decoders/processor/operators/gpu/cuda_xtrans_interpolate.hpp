//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <opencv2/core/cuda.hpp>

#include "decoders/processor/raw_processor_pattern.hpp"

namespace alcedo {
namespace CUDA {

void XTransToRGB_Ref(cv::cuda::GpuMat& image, const XTransPattern6x6& pattern, int passes);

}  // namespace CUDA
}  // namespace alcedo
