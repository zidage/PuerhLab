#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

namespace puerhlab {
class ImageBuffer {
 private:
  cv::Mat                               _cpu_data;
  cv::cuda::GpuMat                      _gpu_data;

  std::unique_ptr<std::vector<uint8_t>> _buffer;

 public:
  bool _cpu_data_valid       = false;
  bool _gpu_data_valid       = false;

  bool _buffer_valid         = false;
  ImageBuffer()              = default;
  ImageBuffer(cv::Mat& data);
  ImageBuffer(cv::Mat&& data);
  ImageBuffer(cv::cuda::GpuMat&& data);
  ImageBuffer(std::vector<uint8_t>&& buffer);
  ImageBuffer(ImageBuffer&& other) noexcept;

  ImageBuffer& operator=(ImageBuffer&& other) noexcept;

  void         ReadFromVectorBuffer(std::vector<uint8_t>&& buffer);

  auto         GetCPUData() -> cv::Mat&;
  auto         GetGPUData() -> cv::cuda::GpuMat&;
  auto         GetBuffer() -> std::vector<uint8_t>&;

  void         SyncToGPU();
  void         SyncToCPU();

  void         InitGPUData(int width, int height, int type);
  void         SetGPUDataValid(bool valid);

  ImageBuffer  Clone() const;

  void         ReleaseCPUData();
  void         ReleaseGPUData();
  void         ReleaseBuffer();
};
};  // namespace puerhlab