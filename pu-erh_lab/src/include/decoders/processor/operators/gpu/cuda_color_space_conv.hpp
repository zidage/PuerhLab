#pragma once

#include <opencv2/core.hpp>

#include "image/image_buffer.hpp"

namespace puerhlab {
namespace CUDA {
void ApplyColorMatrix(cv::cuda::GpuMat& img, const float rgb_cam[][4]);
};
};  // namespace puerhlab