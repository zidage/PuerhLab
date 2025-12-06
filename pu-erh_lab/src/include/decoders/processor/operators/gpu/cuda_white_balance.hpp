#pragma once

#include <libraw/libraw.h>

#include <opencv2/core.hpp>

#include "image/image_buffer.hpp"

namespace puerhlab {
namespace CUDA {
void ToLinearRef(cv::cuda::GpuMat& img, LibRaw& raw_processor);
};
};  // namespace puerhlab