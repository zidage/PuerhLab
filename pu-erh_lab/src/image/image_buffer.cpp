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

void ImageBuffer::ReleaseCPUData() { _cpu_data.release(); }

void ImageBuffer::ReleaseGPUData() { _gpu_data.release(); }
};  // namespace puerhlab