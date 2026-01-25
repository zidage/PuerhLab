//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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