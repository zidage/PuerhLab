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

#include "image/image_buffer.hpp"

#include <opencv2/core/hal/interface.h>

#include <stdexcept>
#include <utility>

namespace puerhlab {

ImageBuffer::~ImageBuffer() {
  // RAII cleanup: release all resources
  cpu_data_.release();
  gpu_data_.release();
  buffer_.reset();
  cpu_data_valid_ = false;
  gpu_data_valid_ = false;
  buffer_valid_   = false;
}

ImageBuffer::ImageBuffer(cv::Mat& data) : cpu_data_valid_(true) { data.copyTo(cpu_data_); }

ImageBuffer::ImageBuffer(cv::Mat&& data) : cpu_data_(data), cpu_data_valid_(true) {}

ImageBuffer::ImageBuffer(std::vector<uint8_t>&& buffer) : buffer_valid_(true) {
  buffer_ = std::make_unique<std::vector<uint8_t>>(std::move(buffer));
}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept
    : cpu_data_(std::move(other.cpu_data_)),
      gpu_data_(std::move(other.gpu_data_)),
      buffer_(std::move(other.buffer_)),
      cpu_data_valid_(other.cpu_data_valid_),
      gpu_data_valid_(other.gpu_data_valid_),
      buffer_valid_(other.buffer_valid_) {}
ImageBuffer::ImageBuffer(cv::cuda::GpuMat&& data)
    : gpu_data_(std::move(data)), gpu_data_valid_(true) {}

ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
  if (this != &other) {
    cpu_data_       = std::move(other.cpu_data_);
    gpu_data_       = std::move(other.gpu_data_);
    buffer_         = std::move(other.buffer_);
    cpu_data_valid_ = other.cpu_data_valid_;
    gpu_data_valid_ = other.gpu_data_valid_;
    buffer_valid_   = other.buffer_valid_;
  }
  return *this;
}

void ImageBuffer::ReadFromVectorBuffer(std::vector<uint8_t>&& buffer) {
  buffer_       = std::make_unique<std::vector<uint8_t>>(std::move(buffer));
  buffer_valid_ = true;
}

auto ImageBuffer::GetCPUData() -> cv::Mat& {
  if (!cpu_data_valid_) {
    throw std::runtime_error("Image Buffer: No valid image data to be returned");
  }
  return cpu_data_;
}

auto ImageBuffer::GetGPUData() -> cv::cuda::GpuMat& {
  if (!gpu_data_valid_) {
    throw std::runtime_error("Image Buffer: No valid image data to be returned");
  }
  // SyncToGPU();
  return gpu_data_;
}

auto ImageBuffer::GetBuffer() -> std::vector<uint8_t>& {
  if (!buffer_valid_) {
    throw std::runtime_error("Image Buffer: No valid buffer data to be returned");
  }
  return *buffer_;
}

auto ImageBuffer::SyncToGPU() -> void {
  if (cpu_data_.empty()) {
    throw std::runtime_error("Image Buffer: No valid CPU data to sync to GPU");
  }
  gpu_data_.upload(cpu_data_);
  gpu_data_valid_ = true;
  // cpu_data_valid_ = false;
}

auto ImageBuffer::SyncToCPU() -> void {
  if (gpu_data_.empty()) {
    throw std::runtime_error("Image Buffer: No valid GPU data to sync to CPU");
  }
  gpu_data_.download(cpu_data_);
  cpu_data_valid_ = true;
  // gpu_data_valid_ = false;
}

void ImageBuffer::InitGPUData(int width, int height, int type) {
  if (gpu_data_valid_ || !gpu_data_.empty()) {
    return;
  }
  gpu_data_.create(height, width, type);
  gpu_data_valid_ = true;
}

void ImageBuffer::SetGPUDataValid(bool valid) {
  gpu_data_valid_ = valid;
}

ImageBuffer ImageBuffer::Clone() const {
  if (cpu_data_valid_) {
    return ImageBuffer{cpu_data_.clone()};
  } else if (gpu_data_valid_) {
    cv::Mat cpu_copy;
    gpu_data_.download(cpu_copy);
    return ImageBuffer{cpu_copy};
  } else if (buffer_valid_) {
    auto buffer = *buffer_;  // copy the buffer
    return ImageBuffer{std::move(buffer)};
  } else {
    throw std::runtime_error("Image Buffer: No valid data to clone");
  }
}

void ImageBuffer::ReleaseCPUData() { cpu_data_.release(); }

void ImageBuffer::ReleaseGPUData() { gpu_data_.release(); }

void ImageBuffer::ReleaseBuffer() { buffer_.reset(); }
};  // namespace puerhlab