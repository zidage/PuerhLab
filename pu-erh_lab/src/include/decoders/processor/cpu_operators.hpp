#pragma once

#include <libraw/libraw.h>
#include <libraw/libraw_types.h>

#include <opencv2/core.hpp>

#include "decoders/processor/cuda_operators.hpp"

namespace puerhlab {
namespace CPU {
void WhiteBalanceCorrectionAndHighlightRestore(cv::Mat& img, LibRaw& raw_processor,
                                               std::array<float, 4>& black_level, const float* wb);

void BayerRGGB2RGB_AHD(cv::Mat& bayer, bool use_AHD, float maximum);

void ApplyColorMatrix(cv::Mat& img, const float rgb_cam[][4]);

auto CalculateBlackLevel(const libraw_rawdata_t& raw_data) -> std::array<float, 4>;

auto GetWBCoeff(const libraw_rawdata_t& raw_data) -> const float*;
};  // namespace CPU
};  // namespace puerhlab