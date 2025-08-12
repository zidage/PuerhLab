#pragma once

#include <cstdint>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

namespace puerhlab {
class ImageBuffer {
 private:
  cv::Mat                       _cpu_data;
  // TODO: NOT USED FOR NOW: 2025-6-30
  cv::cuda::GpuMat              _gpu_data;

  bool                          _data_valid = false;

  std::vector<cv::Mat>          _channels;
  std::vector<cv::cuda::GpuMat> _gpu_channels;

  bool                          _has_channels = false;

  // TODO: NOT USED FOR NOW: 2025-6-30
  void                          SyncToGPU();
  void                          SyncToCPU();

 public:
  ImageBuffer() = default;
  ImageBuffer(cv::Mat& data);
  ImageBuffer(cv::Mat&& data);
  ImageBuffer(std::vector<uint8_t>&& buffer);
  ImageBuffer(ImageBuffer&& other) noexcept;

  ImageBuffer& operator=(ImageBuffer&& other) noexcept;

  void         ReadFromVectorBuffer(std::vector<uint8_t>&& buffer);

  auto         GetCPUData() -> cv::Mat&;
  auto         GetGPUData() -> cv::cuda::GpuMat&;

  void         ReleaseCPUData();
  void         ReleaseGPUData();
};
};  // namespace puerhlab