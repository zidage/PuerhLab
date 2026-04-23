//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_WEBGPU

#include "image/webgpu_image.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "image/webgpu_context.hpp"

namespace alcedo {
namespace webgpu {
namespace {

constexpr uint32_t kCopyRowAlignmentBytes = 256;

auto               AlignUp(size_t value, size_t alignment) -> size_t {
  return ((value + alignment - 1) / alignment) * alignment;
}

auto ToWebGpuTextureFormat(PixelFormat format) -> wgpu::TextureFormat {
  switch (format) {
    case PixelFormat::R16UINT:
      return wgpu::TextureFormat::R16Uint;
    case PixelFormat::RGBA16UINT:
      return wgpu::TextureFormat::RGBA16Uint;
    case PixelFormat::R16FLOAT:
      return wgpu::TextureFormat::R16Float;
    case PixelFormat::RGBA16FLOAT:
      return wgpu::TextureFormat::RGBA16Float;
    case PixelFormat::R32FLOAT:
      return wgpu::TextureFormat::R32Float;
    case PixelFormat::RGBA32FLOAT:
      return wgpu::TextureFormat::RGBA32Float;
  }
  throw std::invalid_argument("WebGpuImage: Unsupported pixel format.");
}

auto AlignedRowBytes(uint32_t width, PixelFormat format) -> size_t {
  const auto cv_type       = WebGpuImage::CVTypeFromPixelFormat(format);
  const auto raw_row_bytes = static_cast<size_t>(width) * CV_ELEM_SIZE(cv_type);
  return AlignUp(raw_row_bytes, kCopyRowAlignmentBytes);
}

auto StagingBufferSize(uint32_t width, uint32_t height, PixelFormat format) -> size_t {
  return AlignedRowBytes(width, format) * static_cast<size_t>(height);
}

auto SupportsStorageBinding(PixelFormat format) -> bool {
  switch (format) {
    case PixelFormat::RGBA16UINT:
    case PixelFormat::RGBA16FLOAT:
    case PixelFormat::R32FLOAT:
    case PixelFormat::RGBA32FLOAT:
      return true;
    case PixelFormat::R16UINT:
    case PixelFormat::R16FLOAT:
      return false;
  }
  return false;
}

void CopyRowsToBuffer(const cv::Mat& host_image, void* destination, size_t destination_row_bytes) {
  const auto source_row_bytes = static_cast<size_t>(host_image.cols) * host_image.elemSize();
  auto*      dst_bytes        = static_cast<std::byte*>(destination);
  for (int row = 0; row < host_image.rows; ++row) {
    std::memcpy(dst_bytes + static_cast<size_t>(row) * destination_row_bytes, host_image.ptr(row),
                source_row_bytes);
  }
}

void CopyRowsFromBuffer(const void* source, size_t source_row_bytes, cv::Mat& host_image) {
  const auto destination_row_bytes = static_cast<size_t>(host_image.cols) * host_image.elemSize();
  auto*      src_bytes             = static_cast<const std::byte*>(source);
  for (int row = 0; row < host_image.rows; ++row) {
    std::memcpy(host_image.ptr(row), src_bytes + static_cast<size_t>(row) * source_row_bytes,
                destination_row_bytes);
  }
}

auto MakeBuffer(uint64_t size, wgpu::BufferUsage usage, bool mapped_at_creation = false)
    -> wgpu::Buffer {
  wgpu::BufferDescriptor descriptor{};
  descriptor.usage            = usage;
  descriptor.size             = size;
  descriptor.mappedAtCreation = mapped_at_creation;
  auto buffer                 = WebGpuContext::Instance().Device().CreateBuffer(&descriptor);
  if (!buffer.Get()) {
    throw std::runtime_error("WebGpuImage: Failed to create WebGPU buffer.");
  }
  return buffer;
}

auto MakeExtent(uint32_t width, uint32_t height) -> wgpu::Extent3D {
  return wgpu::Extent3D{width, height, 1};
}

auto MakeTextureCopy(const wgpu::Texture& texture) -> wgpu::TexelCopyTextureInfo {
  wgpu::TexelCopyTextureInfo copy{};
  copy.texture  = texture;
  copy.mipLevel = 0;
  copy.origin   = wgpu::Origin3D{0, 0, 0};
  copy.aspect   = wgpu::TextureAspect::All;
  return copy;
}

auto MakeBufferCopy(const wgpu::Buffer& buffer, uint32_t row_bytes, uint32_t rows)
    -> wgpu::TexelCopyBufferInfo {
  wgpu::TexelCopyBufferInfo copy{};
  copy.buffer              = buffer;
  copy.layout.offset       = 0;
  copy.layout.bytesPerRow  = row_bytes;
  copy.layout.rowsPerImage = rows;
  return copy;
}

void SubmitAndWait(const wgpu::CommandBuffer& command_buffer) {
  WebGpuContext::Instance().Queue().Submit(1, &command_buffer);
  WebGpuContext::Instance().WaitForSubmittedWork();
}

void WaitForBufferMap(const wgpu::Buffer& buffer, size_t size) {
  bool mapped = false;
  auto future = buffer.MapAsync(
      wgpu::MapMode::Read, 0, size, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::MapAsyncStatus status, wgpu::StringView, bool* complete) {
        if (status == wgpu::MapAsyncStatus::Success) {
          *complete = true;
        }
      },
      &mapped);
  WebGpuContext::Instance().Wait(future);
  if (!mapped) {
    throw std::runtime_error("WebGpuImage: Failed to map readback buffer.");
  }
}

}  // namespace

auto WebGpuImage::PixelFormatFromCVType(int cv_type) -> PixelFormat {
  switch (cv_type) {
    case CV_16UC1:
      return PixelFormat::R16UINT;
    case CV_16UC4:
      return PixelFormat::RGBA16UINT;
    case CV_16FC1:
      return PixelFormat::R16FLOAT;
    case CV_16FC4:
      return PixelFormat::RGBA16FLOAT;
    case CV_32FC1:
      return PixelFormat::R32FLOAT;
    case CV_32FC4:
      return PixelFormat::RGBA32FLOAT;
    default:
      throw std::invalid_argument("WebGpuImage: Unsupported OpenCV type for WebGPU texture.");
  }
}

auto WebGpuImage::CVTypeFromPixelFormat(PixelFormat format) -> int {
  switch (format) {
    case PixelFormat::R16UINT:
      return CV_16UC1;
    case PixelFormat::RGBA16UINT:
      return CV_16UC4;
    case PixelFormat::R16FLOAT:
      return CV_16FC1;
    case PixelFormat::RGBA16FLOAT:
      return CV_16FC4;
    case PixelFormat::R32FLOAT:
      return CV_32FC1;
    case PixelFormat::RGBA32FLOAT:
      return CV_32FC4;
  }
  throw std::invalid_argument("WebGpuImage: Unsupported WebGPU pixel format.");
}

void WebGpuImage::Create(uint32_t width, uint32_t height, PixelFormat format, bool texture_binding,
                         bool storage_binding) {
  if (width == 0 || height == 0) {
    throw std::invalid_argument("WebGpuImage: Texture dimensions must be non-zero.");
  }
  if (!Empty() && width_ == width && height_ == height && format_ == format) {
    return;
  }

  wgpu::TextureUsage usage = wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
  if (texture_binding) {
    usage = usage | wgpu::TextureUsage::TextureBinding;
  }
  if (storage_binding && SupportsStorageBinding(format)) {
    usage = usage | wgpu::TextureUsage::StorageBinding;
  }

  wgpu::TextureDescriptor descriptor{};
  descriptor.usage         = usage;
  descriptor.dimension     = wgpu::TextureDimension::e2D;
  descriptor.size          = MakeExtent(width, height);
  descriptor.format        = ToWebGpuTextureFormat(format);
  descriptor.mipLevelCount = 1;
  descriptor.sampleCount   = 1;

  auto texture             = WebGpuContext::Instance().Device().CreateTexture(&descriptor);
  if (!texture.Get()) {
    throw std::runtime_error("WebGpuImage: Failed to allocate WebGPU texture.");
  }

  width_   = width;
  height_  = height;
  format_  = format;
  texture_ = std::move(texture);
}

void WebGpuImage::Upload(const cv::Mat& host_image) {
  if (host_image.empty()) {
    throw std::invalid_argument("WebGpuImage: Cannot upload an empty host image.");
  }

  const auto format      = PixelFormatFromCVType(host_image.type());
  const auto width       = static_cast<uint32_t>(host_image.cols);
  const auto height      = static_cast<uint32_t>(host_image.rows);
  const auto row_bytes   = static_cast<uint32_t>(AlignedRowBytes(width, format));
  const auto buffer_size = StagingBufferSize(width, height, format);

  Create(width, height, format);

  auto staging =
      MakeBuffer(buffer_size, wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc, true);
  auto* mapped = staging.GetMappedRange(0, buffer_size);
  if (mapped == nullptr) {
    throw std::runtime_error("WebGpuImage: Failed to map upload buffer.");
  }
  std::memset(mapped, 0, buffer_size);
  CopyRowsToBuffer(host_image, mapped, row_bytes);
  staging.Unmap();

  auto encoder = WebGpuContext::Instance().Device().CreateCommandEncoder();
  auto src     = MakeBufferCopy(staging, row_bytes, height);
  auto dst     = MakeTextureCopy(texture_);
  auto extent  = MakeExtent(width, height);
  encoder.CopyBufferToTexture(&src, &dst, &extent);
  SubmitAndWait(encoder.Finish());
}

void WebGpuImage::Download(cv::Mat& host_image) const {
  if (Empty()) {
    throw std::runtime_error("WebGpuImage: Cannot download from an empty texture.");
  }

  const auto cv_type     = CVTypeFromPixelFormat(format_);
  const auto row_bytes   = static_cast<uint32_t>(AlignedRowBytes(width_, format_));
  const auto buffer_size = StagingBufferSize(width_, height_, format_);

  host_image.create(static_cast<int>(height_), static_cast<int>(width_), cv_type);
  auto staging = MakeBuffer(buffer_size, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);

  auto encoder = WebGpuContext::Instance().Device().CreateCommandEncoder();
  auto src     = MakeTextureCopy(texture_);
  auto dst     = MakeBufferCopy(staging, row_bytes, height_);
  auto extent  = MakeExtent(width_, height_);
  encoder.CopyTextureToBuffer(&src, &dst, &extent);
  SubmitAndWait(encoder.Finish());

  WaitForBufferMap(staging, buffer_size);
  const auto* mapped = staging.GetConstMappedRange(0, buffer_size);
  if (mapped == nullptr) {
    throw std::runtime_error("WebGpuImage: Failed to read mapped buffer.");
  }
  CopyRowsFromBuffer(mapped, row_bytes, host_image);
  staging.Unmap();
}

void WebGpuImage::CopyTo(WebGpuImage& dst) const {
  if (Empty()) {
    throw std::runtime_error("WebGpuImage: Cannot copy an empty texture.");
  }

  dst.Create(width_, height_, format_);
  auto encoder = WebGpuContext::Instance().Device().CreateCommandEncoder();
  auto src     = MakeTextureCopy(texture_);
  auto out     = MakeTextureCopy(dst.texture_);
  auto extent  = MakeExtent(width_, height_);
  encoder.CopyTextureToTexture(&src, &out, &extent);
  SubmitAndWait(encoder.Finish());
}

void WebGpuImage::ConvertTo(WebGpuImage& dst, PixelFormat dst_format, double alpha,
                            double beta) const {
  if (Empty()) {
    throw std::runtime_error("WebGpuImage: Cannot convert an empty texture.");
  }
  cv::Mat host;
  Download(host);
  cv::Mat converted;
  host.convertTo(converted, CVTypeFromPixelFormat(dst_format), alpha, beta);
  dst.Upload(converted);
}

void WebGpuImage::Release() noexcept {
  texture_ = nullptr;
  width_   = 0;
  height_  = 0;
  format_  = PixelFormat::RGBA32FLOAT;
}

}  // namespace webgpu
}  // namespace alcedo

#endif
