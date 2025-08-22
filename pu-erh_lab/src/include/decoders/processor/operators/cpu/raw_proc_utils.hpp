#pragma once

#include <opencv2/core.hpp>

#include "hwy/highway.h"

namespace puerhlab {
namespace CPU {
void boxblur2(const cv::Mat1f& src, cv::Mat1f& dst, cv::Mat1f& temp, int startY, int startX, int H,
              int W, int box);

void boxblur_resamp(const cv::Mat1f& src, cv::Mat1f& dst, cv::Mat1f& temp, int H, int W, int box,
                    int samp);
};  // namespace CPU
};  // namespace puerhlab