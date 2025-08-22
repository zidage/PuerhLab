#pragma once

#include <opencv2/core.hpp>

#include "image/image_buffer.hpp"

namespace puerhlab {
namespace CUDA {
void BayerRGGB2RGB_AHD(cv::cuda::GpuMat& image);
};
};  // namespace puerhlab