//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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
  cv::Mat                               cpu_data_;
  cv::cuda::GpuMat                      gpu_data_;

  std::unique_ptr<std::vector<uint8_t>> buffer_;

 public:
  bool cpu_data_valid_       = false;
  bool gpu_data_valid_       = false;

  bool buffer_valid_         = false;
  ImageBuffer()              = default;
  ~ImageBuffer();

  // Non-copyable
  ImageBuffer(const ImageBuffer&)            = delete;
  ImageBuffer& operator=(const ImageBuffer&) = delete;

  // Movable
  ImageBuffer(ImageBuffer&& other) noexcept;
  ImageBuffer& operator=(ImageBuffer&& other) noexcept;

  ImageBuffer(cv::Mat& data);
  ImageBuffer(cv::Mat&& data);
  ImageBuffer(cv::cuda::GpuMat&& data);
  ImageBuffer(std::vector<uint8_t>&& buffer);

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