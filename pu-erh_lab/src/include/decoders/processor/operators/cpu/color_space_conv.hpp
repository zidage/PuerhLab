#pragma once

#include <opencv2/core.hpp>

namespace puerhlab {
namespace CPU {
void ApplyColorMatrix(cv::Mat& img, const float rgb_cam[][4]);
};
};  // namespace puerhlab