#pragma once
#include <opencv2/core/cuda.hpp>

namespace puerhlab {
namespace CUDA {
void ApplyColorMatrix(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Mat& matrix,
                      cv::cuda::Stream& stream);

void WhiteBalanceCorrection(cv::cuda::GpuMat& image, const std::array<float, 4>& black_level,
                            const float* wb_coeffs, float maximum, bool apply_wb_and_black_level,
                            int bayer_offset = 0);

void BayerRGGB2RGB_AHD(cv::cuda::GpuMat& image);
};  // namespace CUDA
};  // namespace puerhlab
