#pragma once

#include <libraw/libraw.h>
#include <opencv2/core.hpp>

namespace puerhlab {
namespace CUDA {
void HighlightReconstruct(cv::Mat& img, LibRaw& raw_processor);
};
};  // namespace puerhlab