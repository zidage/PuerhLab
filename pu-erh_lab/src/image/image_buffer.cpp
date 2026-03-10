//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "image/image_buffer.hpp"

#include <stdexcept>
#include <utility>

namespace puerhlab {

#if defined(HAVE_CUDA) && defined(HAVE_METAL)
#error "PuerhLab currently supports only one compiled GPU backend per build."
#endif

#ifdef HAVE_CUDA
GpuImageWrapper::GpuImageWrapper(cv::cuda::GpuMat&& image) : image_(std::move(image)) {}

auto GpuImageWrapper::GetCUDAImage() -> cv::cuda::GpuMat& { return image_; }

auto GpuImageWrapper::GetCUDAImage() const -> const cv::cuda::GpuMat& { return image_; }
#endif

#ifdef HAVE_METAL
GpuImageWrapper::GpuImageWrapper(metal::MetalImage&& image) : image_(std::move(image)) {}

auto GpuImageWrapper::GetMetalImage() -> metal::MetalImage& { return image_; }

auto GpuImageWrapper::GetMetalImage() const -> const metal::MetalImage& { return image_; }
#endif

auto GpuImageWrapper::Empty() const -> bool {
#ifdef HAVE_CUDA
  return image_.empty();
#elif defined(HAVE_METAL)
  return image_.Empty();
#else
  return true;
#endif
}

auto GpuImageWrapper::Width() const -> int {
#ifdef HAVE_CUDA
  return image_.cols;
#elif defined(HAVE_METAL)
  return static_cast<int>(image_.Width());
#else
  return 0;
#endif
}

auto GpuImageWrapper::Height() const -> int {
#ifdef HAVE_CUDA
  return image_.rows;
#elif defined(HAVE_METAL)
  return static_cast<int>(image_.Height());
#else
  return 0;
#endif
}

auto GpuImageWrapper::Type() const -> int {
#ifdef HAVE_CUDA
  return image_.type();
#elif defined(HAVE_METAL)
  return metal::MetalImage::CVTypeFromPixelFormat(image_.Format());
#else
  return -1;
#endif
}

void GpuImageWrapper::Create(int width, int height, int type) {
#ifdef HAVE_CUDA
  image_.create(height, width, type);
#elif defined(HAVE_METAL)
  image_.Create(static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                metal::MetalImage::PixelFormatFromCVType(type));
#else
  (void)width;
  (void)height;
  (void)type;
  throw std::runtime_error("GpuImageWrapper: No compiled GPU backend.");
#endif
}

void GpuImageWrapper::Upload(const cv::Mat& cpu_data) {
#ifdef HAVE_CUDA
  image_.upload(cpu_data);
#elif defined(HAVE_METAL)
  image_.Upload(cpu_data);
#else
  (void)cpu_data;
  throw std::runtime_error("GpuImageWrapper: No compiled GPU backend.");
#endif
}

void GpuImageWrapper::Download(cv::Mat& cpu_data) const {
#ifdef HAVE_CUDA
  image_.download(cpu_data);
#elif defined(HAVE_METAL)
  image_.Download(cpu_data);
#else
  (void)cpu_data;
  throw std::runtime_error("GpuImageWrapper: No compiled GPU backend.");
#endif
}

void GpuImageWrapper::CopyTo(GpuImageWrapper& dst) const {
  if (Empty()) {
    throw std::runtime_error("GpuImageWrapper: Cannot copy empty GPU data.");
  }

#ifdef HAVE_CUDA
  dst.image_.create(image_.rows, image_.cols, image_.type());
  image_.copyTo(dst.image_);
#elif defined(HAVE_METAL)
  image_.CopyTo(dst.image_);
#else
  (void)dst;
  throw std::runtime_error("GpuImageWrapper: No compiled GPU backend.");
#endif
}

void GpuImageWrapper::ConvertTo(GpuImageWrapper& dst, int type, double alpha, double beta) const {
  if (Empty()) {
    throw std::runtime_error("GpuImageWrapper: Cannot convert empty GPU data.");
  }

#ifdef HAVE_CUDA
  image_.convertTo(dst.image_, type, alpha, beta);
#elif defined(HAVE_METAL)
  image_.ConvertTo(dst.image_, metal::MetalImage::PixelFormatFromCVType(type), alpha, beta);
#else
  (void)dst;
  (void)type;
  (void)alpha;
  (void)beta;
  throw std::runtime_error("GpuImageWrapper: No compiled GPU backend.");
#endif
}

void GpuImageWrapper::Release() {
#ifdef HAVE_CUDA
  image_.release();
#elif defined(HAVE_METAL)
  image_.Release();
#endif
}

ImageBuffer::~ImageBuffer() {
  cpu_data_.release();
  gpu_data_.Release();
  buffer_.reset();
  cpu_data_valid_ = false;
  gpu_data_valid_ = false;
  buffer_valid_   = false;
}

ImageBuffer::ImageBuffer(cv::Mat& data) : cpu_data_valid_(true) { data.copyTo(cpu_data_); }

ImageBuffer::ImageBuffer(cv::Mat&& data) : cpu_data_(std::move(data)), cpu_data_valid_(true) {}

ImageBuffer::ImageBuffer(std::vector<uint8_t>&& buffer) : buffer_valid_(true) {
  buffer_ = std::make_unique<std::vector<uint8_t>>(std::move(buffer));
}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept
    : cpu_data_(std::move(other.cpu_data_)),
      gpu_data_(std::move(other.gpu_data_)),
      buffer_(std::move(other.buffer_)),
      cpu_data_valid_(other.cpu_data_valid_),
      gpu_data_valid_(other.gpu_data_valid_),
      buffer_valid_(other.buffer_valid_) {
  other.cpu_data_valid_ = false;
  other.gpu_data_valid_ = false;
  other.buffer_valid_   = false;
}

#ifdef HAVE_CUDA
ImageBuffer::ImageBuffer(cv::cuda::GpuMat&& data) : gpu_data_(std::move(data)), gpu_data_valid_(true) {}
#endif

#ifdef HAVE_METAL
ImageBuffer::ImageBuffer(metal::MetalImage&& data) : gpu_data_(std::move(data)), gpu_data_valid_(true) {}
#endif

ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
  if (this != &other) {
    cpu_data_       = std::move(other.cpu_data_);
    gpu_data_       = std::move(other.gpu_data_);
    buffer_         = std::move(other.buffer_);
    cpu_data_valid_ = other.cpu_data_valid_;
    gpu_data_valid_ = other.gpu_data_valid_;
    buffer_valid_   = other.buffer_valid_;

    other.cpu_data_valid_ = false;
    other.gpu_data_valid_ = false;
    other.buffer_valid_   = false;
  }
  return *this;
}

void ImageBuffer::ReadFromVectorBuffer(std::vector<uint8_t>&& buffer) {
  buffer_       = std::make_unique<std::vector<uint8_t>>(std::move(buffer));
  buffer_valid_ = true;
}

auto ImageBuffer::GetCPUData() -> cv::Mat& {
  if (!cpu_data_valid_) {
    throw std::runtime_error("ImageBuffer: No valid CPU image data.");
  }
  return cpu_data_;
}

auto ImageBuffer::GetBuffer() -> std::vector<uint8_t>& {
  if (!buffer_valid_) {
    throw std::runtime_error("ImageBuffer: No valid encoded buffer data.");
  }
  return *buffer_;
}

#ifdef HAVE_CUDA
auto ImageBuffer::GetCUDAImage() -> cv::cuda::GpuMat& {
  if (!gpu_data_valid_) {
    throw std::runtime_error("ImageBuffer: No valid GPU image data.");
  }
  return gpu_data_.GetCUDAImage();
}

auto ImageBuffer::GetCUDAImage() const -> const cv::cuda::GpuMat& {
  if (!gpu_data_valid_) {
    throw std::runtime_error("ImageBuffer: No valid GPU image data.");
  }
  return gpu_data_.GetCUDAImage();
}
#endif

#ifdef HAVE_METAL
auto ImageBuffer::GetMetalImage() -> metal::MetalImage& {
  if (!gpu_data_valid_) {
    throw std::runtime_error("ImageBuffer: No valid GPU image data.");
  }
  return gpu_data_.GetMetalImage();
}

auto ImageBuffer::GetMetalImage() const -> const metal::MetalImage& {
  if (!gpu_data_valid_) {
    throw std::runtime_error("ImageBuffer: No valid GPU image data.");
  }
  return gpu_data_.GetMetalImage();
}
#endif

auto ImageBuffer::GetGPUWidth() const -> int {
  if (!gpu_data_valid_) {
    throw std::runtime_error("ImageBuffer: No valid GPU image data.");
  }
  return gpu_data_.Width();
}

auto ImageBuffer::GetGPUHeight() const -> int {
  if (!gpu_data_valid_) {
    throw std::runtime_error("ImageBuffer: No valid GPU image data.");
  }
  return gpu_data_.Height();
}

auto ImageBuffer::GetGPUType() const -> int {
  if (!gpu_data_valid_) {
    throw std::runtime_error("ImageBuffer: No valid GPU image data.");
  }
  return gpu_data_.Type();
}

auto ImageBuffer::SyncToGPU() -> void {
  if (!cpu_data_valid_ || cpu_data_.empty()) {
    throw std::runtime_error("ImageBuffer: No valid CPU data to sync to GPU.");
  }
  gpu_data_.Upload(cpu_data_);
  gpu_data_valid_ = true;
}

auto ImageBuffer::SyncToCPU() -> void {
  if (!gpu_data_valid_ || gpu_data_.Empty()) {
    throw std::runtime_error("ImageBuffer: No valid GPU data to sync to CPU.");
  }
  gpu_data_.Download(cpu_data_);
  cpu_data_valid_ = true;
}

void ImageBuffer::ConvertGPUDataTo(int type, double alpha, double beta) {
  if (!gpu_data_valid_ || gpu_data_.Empty()) {
    throw std::runtime_error("ImageBuffer: No valid GPU data to convert.");
  }

  GpuImageWrapper converted;
  gpu_data_.ConvertTo(converted, type, alpha, beta);
  gpu_data_       = std::move(converted);
  gpu_data_valid_ = true;
}

void ImageBuffer::CopyGPUDataTo(ImageBuffer& dst) const {
  if (!gpu_data_valid_ || gpu_data_.Empty()) {
    throw std::runtime_error("ImageBuffer: No valid GPU data to copy.");
  }

  gpu_data_.CopyTo(dst.gpu_data_);
  dst.gpu_data_valid_ = true;
}

void ImageBuffer::InitGPUData(int width, int height, int type) {
  if (gpu_data_valid_ || !gpu_data_.Empty()) {
    return;
  }
  gpu_data_.Create(width, height, type);
  gpu_data_valid_ = true;
}

void ImageBuffer::SetGPUDataValid(bool valid) { gpu_data_valid_ = valid; }

ImageBuffer ImageBuffer::Clone() const {
  if (cpu_data_valid_) {
    return ImageBuffer{cpu_data_.clone()};
  }
  if (gpu_data_valid_) {
    cv::Mat cpu_copy;
    gpu_data_.Download(cpu_copy);
    return ImageBuffer{std::move(cpu_copy)};
  }
  if (buffer_valid_) {
    auto buffer = *buffer_;
    return ImageBuffer{std::move(buffer)};
  }
  throw std::runtime_error("ImageBuffer: No valid data to clone.");
}

void ImageBuffer::ReleaseCPUData() {
  cpu_data_.release();
  cpu_data_valid_ = false;
}

void ImageBuffer::ReleaseGPUData() {
  gpu_data_.Release();
  gpu_data_valid_ = false;
}

void ImageBuffer::ReleaseBuffer() {
  buffer_.reset();
  buffer_valid_ = false;
}

}  // namespace puerhlab
