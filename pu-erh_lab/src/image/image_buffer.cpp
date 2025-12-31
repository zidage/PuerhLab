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
ImageBuffer::ImageBuffer(cv::Mat& data) : _cpu_data_valid(true) { data.copyTo(_cpu_data); }

ImageBuffer::ImageBuffer(cv::Mat&& data) : _cpu_data(data), _cpu_data_valid(true) {}

ImageBuffer::ImageBuffer(std::vector<uint8_t>&& buffer) : _buffer_valid(true) {
  _buffer = std::make_unique<std::vector<uint8_t>>(std::move(buffer));
}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept
    : _cpu_data(std::move(other._cpu_data)),
      _gpu_data(std::move(other._gpu_data)),
      _buffer(std::move(other._buffer)),
      _cpu_data_valid(other._cpu_data_valid),
      _gpu_data_valid(other._gpu_data_valid),
      _buffer_valid(other._buffer_valid) {}

ImageBuffer::ImageBuffer(cv::cuda::GpuMat&& data)
    : _gpu_data(std::move(data)), _gpu_data_valid(true) {}

ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
  if (this != &other) {
    _cpu_data       = std::move(other._cpu_data);
    _gpu_data       = std::move(other._gpu_data);
    _buffer         = std::move(other._buffer);
    _cpu_data_valid = other._cpu_data_valid;
    _gpu_data_valid = other._gpu_data_valid;
    _buffer_valid   = other._buffer_valid;
  }
  return *this;
}

void ImageBuffer::ReadFromVectorBuffer(std::vector<uint8_t>&& buffer) {
  _buffer       = std::make_unique<std::vector<uint8_t>>(std::move(buffer));
  _buffer_valid = true;
}

auto ImageBuffer::GetCPUData() -> cv::Mat& {
  if (!_cpu_data_valid) {
    throw std::runtime_error("Image Buffer: No valid image data to be returned");
  }
  return _cpu_data;
}

auto ImageBuffer::GetGPUData() -> cv::cuda::GpuMat& {
  if (!_gpu_data_valid) {
    throw std::runtime_error("Image Buffer: No valid image data to be returned");
  }
  // SyncToGPU();
  return _gpu_data;
}

auto ImageBuffer::GetBuffer() -> std::vector<uint8_t>& {
  if (!_buffer_valid) {
    throw std::runtime_error("Image Buffer: No valid buffer data to be returned");
  }
  return *_buffer;
}

auto ImageBuffer::SyncToGPU() -> void {
  if (_cpu_data.empty()) {
    throw std::runtime_error("Image Buffer: No valid CPU data to sync to GPU");
  }
  _gpu_data.upload(_cpu_data);
  _gpu_data_valid = true;
  // _cpu_data_valid = false;
}

auto ImageBuffer::SyncToCPU() -> void {
  if (_gpu_data.empty()) {
    throw std::runtime_error("Image Buffer: No valid GPU data to sync to CPU");
  }
  _gpu_data.download(_cpu_data);
  _cpu_data_valid = true;
  // _gpu_data_valid = false;
}

void ImageBuffer::InitGPUData(int width, int height, int type) {
  if (_gpu_data_valid || !_gpu_data.empty()) {
    return;
  }
  _gpu_data.create(height, width, type);
  _gpu_data_valid = true;
}

void ImageBuffer::SetGPUDataValid(bool valid) {
  _gpu_data_valid = valid;
}

ImageBuffer ImageBuffer::Clone() const {
  if (_cpu_data_valid) {
    return ImageBuffer{_cpu_data.clone()};
  } else if (_gpu_data_valid) {
    cv::Mat cpu_copy;
    _gpu_data.download(cpu_copy);
    return ImageBuffer{cpu_copy};
  } else if (_buffer_valid) {
    auto buffer = *_buffer;  // copy the buffer
    return ImageBuffer{std::move(buffer)};
  } else {
    throw std::runtime_error("Image Buffer: No valid data to clone");
  }
}

void ImageBuffer::ReleaseCPUData() { _cpu_data.release(); }

void ImageBuffer::ReleaseGPUData() { _gpu_data.release(); }

void ImageBuffer::ReleaseBuffer() { _buffer.reset(); }
};  // namespace puerhlab