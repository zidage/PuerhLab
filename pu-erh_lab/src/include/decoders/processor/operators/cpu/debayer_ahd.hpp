#pragma once

#include <opencv2/core.hpp>

namespace puerhlab {
namespace CPU {
void BayerRGGB2RGB_AHD(cv::Mat& bayer, bool use_AHD);
};
};  // namespace puerhlab