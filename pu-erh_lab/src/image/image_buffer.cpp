#include "image/image_buffer.hpp"

#include <opencv2/core/hal/interface.h>

#include <stdexcept>
#include <utility>

namespace puerhlab {
ImageBuffer::ImageBuffer(cv::Mat& data) : _data_valid(true) { data.copyTo(_cpu_data); }

ImageBuffer::ImageBuffer(cv::Mat&& data) : _cpu_data(data), _data_valid(true) {}

ImageBuffer::ImageBuffer(std::vector<uint8_t>&& buffer)
    : _cpu_data(static_cast<int>(buffer.size()), 1, CV_32FC3, buffer.data()), _data_valid(true) {}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept
    : _cpu_data(std::move(other._cpu_data)),
      _gpu_data(std::move(other._gpu_data)),
      _data_valid(other._data_valid) {}

ImageBuffer::ImageBuffer(cv::cuda::GpuMat&& data) : _gpu_data(std::move(data)), _data_valid(true) {}

ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
  if (this != &other) {
    _cpu_data   = std::move(other._cpu_data);
    _gpu_data   = std::move(other._gpu_data);
    _data_valid = other._data_valid;
  }
  return *this;
}

void ImageBuffer::ReadFromVectorBuffer(std::vector<uint8_t>&& buffer) {
  cv::Mat loaded_data{(int)buffer.size(), 1, CV_32FC3, buffer.data()};
  _data_valid = true;
}

auto ImageBuffer::GetCPUData() -> cv::Mat& {
  if (!_data_valid) {
    throw std::runtime_error("Image Buffer: No valid image data to be returned");
  }
  return _cpu_data;
}

auto ImageBuffer::GetGPUData() -> cv::cuda::GpuMat& {
  if (!_data_valid) {
    throw std::runtime_error("Image Buffer: No valid image data to be returned");
  }
  SyncToGPU();
  return _gpu_data;
}

auto ImageBuffer::SyncToGPU() -> void {
  if (_cpu_data.empty()) {
    throw std::runtime_error("Image Buffer: No valid CPU data to sync to GPU");
  }
  _gpu_data.upload(_cpu_data);
}

auto ImageBuffer::SyncToCPU() -> void {
  if (_gpu_data.empty()) {
    throw std::runtime_error("Image Buffer: No valid GPU data to sync to CPU");
  }
  _gpu_data.download(_cpu_data);
}

void ImageBuffer::ReleaseCPUData() { _cpu_data.release(); }

void ImageBuffer::ReleaseGPUData() { _gpu_data.release(); }
};  // namespace puerhlab