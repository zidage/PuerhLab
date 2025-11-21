#pragma once

#include <opencv2/core.hpp>

namespace puerhlab {
namespace CPU {
void BayerRGGB2RGB_RCD(cv::Mat& bayer);
};
};  // namespace puerhlab