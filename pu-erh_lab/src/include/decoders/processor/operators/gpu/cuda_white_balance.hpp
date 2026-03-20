//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <libraw/libraw.h>

#include <opencv2/core.hpp>

#include "decoders/processor/raw_processor_pattern.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace CUDA {
void ToLinearRef(cv::cuda::GpuMat& img, LibRaw& raw_processor, const RawCfaPattern& pattern);
};
};  // namespace puerhlab
