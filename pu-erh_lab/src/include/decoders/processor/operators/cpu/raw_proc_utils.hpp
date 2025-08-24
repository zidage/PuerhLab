#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "hwy/highway.h"

namespace puerhlab {
namespace CPU {
void boxblur2(const cv::Mat1f& src, cv::Mat1f& dst, cv::Mat1f& temp, int startY, int startX, int H,
              int W, int box);

void boxblur_resamp(const cv::Mat1f& src, cv::Mat1f& dst, cv::Mat1f& temp, int H, int W, int box,
                    int samp);

inline static void DebuggingPreview(cv::Mat& src) {
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(512, 512));
  cv::imshow("Debugging Preview", resized);
  cv::waitKey(0);
}
};  // namespace CPU
};  // namespace puerhlab