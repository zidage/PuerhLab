//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <libraw/libraw.h>
#include <opencv2/core/cuda.hpp>

#include "decoders/processor/raw_processor_pattern.hpp"

namespace puerhlab {
namespace CUDA {
void HighlightReconstruct(cv::cuda::GpuMat& img, LibRaw& raw_processor);
void Clamp01(cv::cuda::GpuMat& img);
};  // namespace CUDA
};  // namespace puerhlab
