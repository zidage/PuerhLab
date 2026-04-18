//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL

#include "metal/metal_utils/metal_convert_utils.hpp"

#include "image/metal_image.hpp"
#include "metal/compute_pipeline_cache.hpp"
#include "metal/metal_context.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace alcedo::metal::utils {
namespace {

struct ConvertParams {
  float    alpha;
  float    beta;
  uint32_t width;
  uint32_t height;
  uint32_t src_stride;
  uint32_t dst_stride;
};

struct CropParams {
  uint32_t origin_x;
  uint32_t origin_y;
  uint32_t crop_width;
  uint32_t crop_height;
  uint32_t src_stride;
  uint32_t dst_stride;
};

struct ClampParams {
  float    lo;
  float    hi;
  uint32_t width;
  uint32_t height;
  uint32_t stride;
};

struct RotateParams {
  uint32_t src_width;
  uint32_t src_height;
  uint32_t dst_width;
  uint32_t dst_height;
  uint32_t src_stride;
  uint32_t dst_stride;
};

enum class RotationOp : uint32_t {
  Rotate180,
  Rotate90CW,
  Rotate90CCW,
};

constexpr uint32_t kRowAlignmentBytes = 256;

auto AlignRowBytes(size_t row_bytes) -> size_t {
  return ((row_bytes + kRowAlignmentBytes - 1) / kRowAlignmentBytes) * kRowAlignmentBytes;
}

auto RowBytesFor(uint32_t width, PixelFormat format) -> size_t {
  const auto cv_type = MetalImage::CVTypeFromPixelFormat(format);
  return AlignRowBytes(static_cast<size_t>(width) * CV_ELEM_SIZE(cv_type));
}

auto BuildPipelineKey(PixelFormat src_format, PixelFormat dst_format) -> uint32_t {
  return (static_cast<uint32_t>(src_format) << 16U) | static_cast<uint32_t>(dst_format);
}

auto KernelNameFor(PixelFormat src_format, PixelFormat dst_format) -> const char* {
  switch (BuildPipelineKey(src_format, dst_format)) {
    case (static_cast<uint32_t>(PixelFormat::R16UINT) << 16U) |
        static_cast<uint32_t>(PixelFormat::R16UINT):
      return "convert_r16u_to_r16u";
    case (static_cast<uint32_t>(PixelFormat::R16UINT) << 16U) |
        static_cast<uint32_t>(PixelFormat::R32FLOAT):
      return "convert_r16u_to_r32f";
    case (static_cast<uint32_t>(PixelFormat::R32FLOAT) << 16U) |
        static_cast<uint32_t>(PixelFormat::R16UINT):
      return "convert_r32f_to_r16u";
    case (static_cast<uint32_t>(PixelFormat::R32FLOAT) << 16U) |
        static_cast<uint32_t>(PixelFormat::R32FLOAT):
      return "convert_r32f_to_r32f";
    case (static_cast<uint32_t>(PixelFormat::RGBA16UINT) << 16U) |
        static_cast<uint32_t>(PixelFormat::RGBA16UINT):
      return "convert_rgba16u_to_rgba16u";
    case (static_cast<uint32_t>(PixelFormat::RGBA16UINT) << 16U) |
        static_cast<uint32_t>(PixelFormat::RGBA32FLOAT):
      return "convert_rgba16u_to_rgba32f";
    case (static_cast<uint32_t>(PixelFormat::RGBA32FLOAT) << 16U) |
        static_cast<uint32_t>(PixelFormat::RGBA16UINT):
      return "convert_rgba32f_to_rgba16u";
    case (static_cast<uint32_t>(PixelFormat::RGBA32FLOAT) << 16U) |
        static_cast<uint32_t>(PixelFormat::RGBA32FLOAT):
      return "convert_rgba32f_to_rgba32f";
    default:
      return nullptr;
  }
}

auto CropKernelNameFor(PixelFormat format) -> const char* {
  switch (format) {
    case PixelFormat::R32FLOAT:
      return "crop_r32f";
    case PixelFormat::RGBA32FLOAT:
      return "crop_rgba32f";
    default:
      return nullptr;
  }
}

auto ClampKernelNameFor(PixelFormat format) -> const char* {
  switch (format) {
    case PixelFormat::R32FLOAT:
      return "clamp_r32f";
    case PixelFormat::RGBA32FLOAT:
      return "clamp_rgba32f";
    default:
      return nullptr;
  }
}

auto RotateKernelNameFor(RotationOp op, PixelFormat format) -> const char* {
  switch (op) {
    case RotationOp::Rotate180:
      switch (format) {
        case PixelFormat::R32FLOAT:
          return "rotate180_r32f";
        case PixelFormat::RGBA32FLOAT:
          return "rotate180_rgba32f";
        default:
          return nullptr;
      }
    case RotationOp::Rotate90CW:
      switch (format) {
        case PixelFormat::R32FLOAT:
          return "rotate90cw_r32f";
        case PixelFormat::RGBA32FLOAT:
          return "rotate90cw_rgba32f";
        default:
          return nullptr;
      }
    case RotationOp::Rotate90CCW:
      switch (format) {
        case PixelFormat::R32FLOAT:
          return "rotate90ccw_r32f";
        case PixelFormat::RGBA32FLOAT:
          return "rotate90ccw_rgba32f";
        default:
          return nullptr;
      }
  }

  return nullptr;
}

auto GetPipelineState(PixelFormat src_format, PixelFormat dst_format)
    -> NS::SharedPtr<MTL::ComputePipelineState> {
  const auto* kernel_name = KernelNameFor(src_format, dst_format);
  if (kernel_name == nullptr) {
    throw std::runtime_error("Metal convert utils: unsupported conversion pair.");
  }

#ifndef ALCEDO_METAL_UTILS_METALLIB_PATH
  throw std::runtime_error("Metal convert utils metallib path is not configured.");
#else
  return ComputePipelineCache::Instance().GetPipelineState(
      ALCEDO_METAL_UTILS_METALLIB_PATH, kernel_name, "Metal convert utils");
#endif
}

auto GetCropPipelineState(PixelFormat format) -> NS::SharedPtr<MTL::ComputePipelineState> {
  const auto* kernel_name = CropKernelNameFor(format);
  if (kernel_name == nullptr) {
    throw std::runtime_error("Metal convert utils: unsupported crop format.");
  }

#ifndef ALCEDO_METAL_UTILS_METALLIB_PATH
  throw std::runtime_error("Metal convert utils metallib path is not configured.");
#else
  return ComputePipelineCache::Instance().GetPipelineState(
      ALCEDO_METAL_UTILS_METALLIB_PATH, kernel_name, "Metal convert utils");
#endif
}

auto GetClampPipelineState(PixelFormat format) -> NS::SharedPtr<MTL::ComputePipelineState> {
  const auto* kernel_name = ClampKernelNameFor(format);
  if (kernel_name == nullptr) {
    throw std::runtime_error("Metal convert utils: unsupported clamp format.");
  }

#ifndef ALCEDO_METAL_UTILS_METALLIB_PATH
  throw std::runtime_error("Metal convert utils metallib path is not configured.");
#else
  return ComputePipelineCache::Instance().GetPipelineState(
      ALCEDO_METAL_UTILS_METALLIB_PATH, kernel_name, "Metal convert utils");
#endif
}

auto GetRotatePipelineState(RotationOp op, PixelFormat format)
    -> NS::SharedPtr<MTL::ComputePipelineState> {
  const auto* kernel_name = RotateKernelNameFor(op, format);
  if (kernel_name == nullptr) {
    throw std::runtime_error("Metal convert utils: unsupported rotate format.");
  }

#ifndef ALCEDO_METAL_UTILS_METALLIB_PATH
  throw std::runtime_error("Metal convert utils metallib path is not configured.");
#else
  return ComputePipelineCache::Instance().GetPipelineState(
      ALCEDO_METAL_UTILS_METALLIB_PATH, kernel_name, "Metal convert utils");
#endif
}

auto MakeSharedBuffer(size_t length) -> NS::SharedPtr<MTL::Buffer> {
  auto* device = MetalContext::Instance().Device();
  if (device == nullptr) {
    throw std::runtime_error("Metal convert utils: Metal device is unavailable.");
  }

  auto buffer = NS::TransferPtr(
      device->newBuffer(static_cast<NS::UInteger>(length), MTL::ResourceStorageModeShared));
  if (!buffer) {
    throw std::runtime_error("Metal convert utils: failed to allocate staging buffer.");
  }
  return buffer;
}

void DispatchThreads(MTL::ComputeCommandEncoder* encoder, MTL::ComputePipelineState* pipeline,
                     uint32_t width, uint32_t height) {
  const auto thread_width  = std::max<NS::UInteger>(1, pipeline->threadExecutionWidth());
  const auto thread_height =
      std::max<NS::UInteger>(1, pipeline->maxTotalThreadsPerThreadgroup() / thread_width);
  const MTL::Size threads_per_group{thread_width, thread_height, 1};
  const MTL::Size threads_per_grid{width, height, 1};
  encoder->dispatchThreads(threads_per_grid, threads_per_group);
}

void DispatchConversion(const MetalImage& src, MetalImage& dst, double alpha, double beta) {
  const auto src_row_bytes = RowBytesFor(src.Width(), src.Format());
  const auto dst_row_bytes = RowBytesFor(dst.Width(), dst.Format());
  const auto src_size      = src_row_bytes * src.Height();
  const auto dst_size      = dst_row_bytes * dst.Height();

  auto src_buffer = MakeSharedBuffer(src_size);
  auto dst_buffer = MakeSharedBuffer(dst_size);

  auto* queue = MetalContext::Instance().Queue();
  if (queue == nullptr) {
    throw std::runtime_error("Metal convert utils: Metal queue is unavailable.");
  }

  auto command_buffer = NS::RetainPtr(queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("Metal convert utils: failed to create command buffer.");
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromTexture(src.Texture(), 0, 0, MTL::Origin{0, 0, 0},
                          MTL::Size{src.Width(), src.Height(), 1}, src_buffer.get(), 0,
                          src_row_bytes, src_size);
    blit->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(src.Format(), dst.Format());
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    const ConvertParams params{
        .alpha      = static_cast<float>(alpha),
        .beta       = static_cast<float>(beta),
        .width      = src.Width(),
        .height     = src.Height(),
        .src_stride = static_cast<uint32_t>(
            src_row_bytes / CV_ELEM_SIZE(MetalImage::CVTypeFromPixelFormat(src.Format()))),
        .dst_stride = static_cast<uint32_t>(
            dst_row_bytes / CV_ELEM_SIZE(MetalImage::CVTypeFromPixelFormat(dst.Format()))),
    };

    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(src_buffer.get(), 0, 0);
    compute->setBuffer(dst_buffer.get(), 0, 1);
    compute->setBytes(&params, sizeof(params), 2);
    DispatchThreads(compute.get(), pipeline.get(), src.Width(), src.Height());
    compute->endEncoding();
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromBuffer(dst_buffer.get(), 0, dst_row_bytes, dst_size,
                         MTL::Size{dst.Width(), dst.Height(), 1}, dst.Texture(), 0, 0,
                         MTL::Origin{0, 0, 0});
    blit->endEncoding();
  }

  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

void DispatchCrop(const MetalImage& src, MetalImage& dst, const cv::Rect& crop_rect) {
  const auto src_row_bytes = RowBytesFor(src.Width(), src.Format());
  const auto dst_row_bytes = RowBytesFor(dst.Width(), dst.Format());
  const auto src_size      = src_row_bytes * src.Height();
  const auto dst_size      = dst_row_bytes * dst.Height();

  auto src_buffer = MakeSharedBuffer(src_size);
  auto dst_buffer = MakeSharedBuffer(dst_size);

  auto* queue = MetalContext::Instance().Queue();
  if (queue == nullptr) {
    throw std::runtime_error("Metal convert utils: Metal queue is unavailable.");
  }

  auto command_buffer = NS::RetainPtr(queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("Metal convert utils: failed to create command buffer.");
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromTexture(src.Texture(), 0, 0, MTL::Origin{0, 0, 0},
                          MTL::Size{src.Width(), src.Height(), 1}, src_buffer.get(), 0,
                          src_row_bytes, src_size);
    blit->endEncoding();
  }

  {
    auto pipeline = GetCropPipelineState(src.Format());
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    const CropParams params{
        .origin_x    = static_cast<uint32_t>(crop_rect.x),
        .origin_y    = static_cast<uint32_t>(crop_rect.y),
        .crop_width  = dst.Width(),
        .crop_height = dst.Height(),
        .src_stride  = static_cast<uint32_t>(
            src_row_bytes / CV_ELEM_SIZE(MetalImage::CVTypeFromPixelFormat(src.Format()))),
        .dst_stride  = static_cast<uint32_t>(
            dst_row_bytes / CV_ELEM_SIZE(MetalImage::CVTypeFromPixelFormat(dst.Format()))),
    };

    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(src_buffer.get(), 0, 0);
    compute->setBuffer(dst_buffer.get(), 0, 1);
    compute->setBytes(&params, sizeof(params), 2);
    DispatchThreads(compute.get(), pipeline.get(), dst.Width(), dst.Height());
    compute->endEncoding();
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromBuffer(dst_buffer.get(), 0, dst_row_bytes, dst_size,
                         MTL::Size{dst.Width(), dst.Height(), 1}, dst.Texture(), 0, 0,
                         MTL::Origin{0, 0, 0});
    blit->endEncoding();
  }

  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

void DispatchClamp(MetalImage& image, float lo, float hi) {
  const auto row_bytes = RowBytesFor(image.Width(), image.Format());
  const auto size      = row_bytes * image.Height();

  auto buffer = MakeSharedBuffer(size);

  auto* queue = MetalContext::Instance().Queue();
  if (queue == nullptr) {
    throw std::runtime_error("Metal convert utils: Metal queue is unavailable.");
  }

  auto command_buffer = NS::RetainPtr(queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("Metal convert utils: failed to create command buffer.");
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromTexture(image.Texture(), 0, 0, MTL::Origin{0, 0, 0},
                          MTL::Size{image.Width(), image.Height(), 1}, buffer.get(), 0, row_bytes,
                          size);
    blit->endEncoding();
  }

  {
    auto pipeline = GetClampPipelineState(image.Format());
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    const ClampParams params{
        .lo     = lo,
        .hi     = hi,
        .width  = image.Width(),
        .height = image.Height(),
        .stride = static_cast<uint32_t>(
            row_bytes / CV_ELEM_SIZE(MetalImage::CVTypeFromPixelFormat(image.Format()))),
    };

    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(buffer.get(), 0, 0);
    compute->setBytes(&params, sizeof(params), 1);
    DispatchThreads(compute.get(), pipeline.get(), image.Width(), image.Height());
    compute->endEncoding();
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromBuffer(buffer.get(), 0, row_bytes, size,
                         MTL::Size{image.Width(), image.Height(), 1}, image.Texture(), 0, 0,
                         MTL::Origin{0, 0, 0});
    blit->endEncoding();
  }

  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

void DispatchRotate(const MetalImage& src, MetalImage& dst, RotationOp op) {
  const auto src_row_bytes = RowBytesFor(src.Width(), src.Format());
  const auto dst_row_bytes = RowBytesFor(dst.Width(), dst.Format());
  const auto src_size      = src_row_bytes * src.Height();
  const auto dst_size      = dst_row_bytes * dst.Height();

  auto src_buffer = MakeSharedBuffer(src_size);
  auto dst_buffer = MakeSharedBuffer(dst_size);

  auto* queue = MetalContext::Instance().Queue();
  if (queue == nullptr) {
    throw std::runtime_error("Metal convert utils: Metal queue is unavailable.");
  }

  auto command_buffer = NS::RetainPtr(queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("Metal convert utils: failed to create command buffer.");
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromTexture(src.Texture(), 0, 0, MTL::Origin{0, 0, 0},
                          MTL::Size{src.Width(), src.Height(), 1}, src_buffer.get(), 0,
                          src_row_bytes, src_size);
    blit->endEncoding();
  }

  {
    auto pipeline = GetRotatePipelineState(op, src.Format());
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    const RotateParams params{
        .src_width  = src.Width(),
        .src_height = src.Height(),
        .dst_width  = dst.Width(),
        .dst_height = dst.Height(),
        .src_stride = static_cast<uint32_t>(
            src_row_bytes / CV_ELEM_SIZE(MetalImage::CVTypeFromPixelFormat(src.Format()))),
        .dst_stride = static_cast<uint32_t>(
            dst_row_bytes / CV_ELEM_SIZE(MetalImage::CVTypeFromPixelFormat(dst.Format()))),
    };

    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(src_buffer.get(), 0, 0);
    compute->setBuffer(dst_buffer.get(), 0, 1);
    compute->setBytes(&params, sizeof(params), 2);
    DispatchThreads(compute.get(), pipeline.get(), dst.Width(), dst.Height());
    compute->endEncoding();
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromBuffer(dst_buffer.get(), 0, dst_row_bytes, dst_size,
                         MTL::Size{dst.Width(), dst.Height(), 1}, dst.Texture(), 0, 0,
                         MTL::Origin{0, 0, 0});
    blit->endEncoding();
  }

  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

}  // namespace

void ConvertTexture(const MetalImage& src, MetalImage& dst, double alpha, double beta) {
  if (src.Empty()) {
    throw std::runtime_error("Metal convert utils: source texture is empty.");
  }
  if (dst.Empty()) {
    throw std::runtime_error("Metal convert utils: destination texture is empty.");
  }
  if (src.Width() != dst.Width() || src.Height() != dst.Height()) {
    throw std::runtime_error("Metal convert utils: source and destination sizes must match.");
  }

  DispatchConversion(src, dst, alpha, beta);
}

void CropTexture(const MetalImage& src, MetalImage& dst, const cv::Rect& crop_rect) {
  if (src.Empty()) {
    throw std::runtime_error("Metal convert utils: source texture is empty.");
  }
  if (dst.Empty()) {
    throw std::runtime_error("Metal convert utils: destination texture is empty.");
  }
  if (crop_rect.width <= 0 || crop_rect.height <= 0) {
    throw std::runtime_error("Metal convert utils: crop rectangle must be non-empty.");
  }
  if (crop_rect.x < 0 || crop_rect.y < 0 ||
      crop_rect.x + crop_rect.width > static_cast<int>(src.Width()) ||
      crop_rect.y + crop_rect.height > static_cast<int>(src.Height())) {
    throw std::runtime_error("Metal convert utils: crop rectangle is out of bounds.");
  }
  if (dst.Width() != static_cast<uint32_t>(crop_rect.width) ||
      dst.Height() != static_cast<uint32_t>(crop_rect.height) || dst.Format() != src.Format()) {
    throw std::runtime_error("Metal convert utils: destination size/format does not match crop.");
  }

  DispatchCrop(src, dst, crop_rect);
}

void ClampTexture(MetalImage& image, float lo, float hi) {
  if (image.Empty()) {
    throw std::runtime_error("Metal convert utils: texture is empty.");
  }
  if (lo > hi) {
    throw std::runtime_error("Metal convert utils: clamp lower bound exceeds upper bound.");
  }
  if (image.Format() != PixelFormat::R32FLOAT && image.Format() != PixelFormat::RGBA32FLOAT) {
    throw std::runtime_error("Metal convert utils: unsupported clamp format.");
  }

  DispatchClamp(image, lo, hi);
}

void Rotate180(MetalImage& image) {
  if (image.Empty()) {
    return;
  }
  if (image.Format() != PixelFormat::R32FLOAT && image.Format() != PixelFormat::RGBA32FLOAT) {
    throw std::runtime_error("Metal convert utils: unsupported rotate format.");
  }

  MetalImage output =
      MetalImage::Create2D(image.Width(), image.Height(), image.Format(), true, true, false);
  DispatchRotate(image, output, RotationOp::Rotate180);
  image = std::move(output);
}

void Rotate90CW(MetalImage& image) {
  if (image.Empty()) {
    return;
  }
  if (image.Format() != PixelFormat::R32FLOAT && image.Format() != PixelFormat::RGBA32FLOAT) {
    throw std::runtime_error("Metal convert utils: unsupported rotate format.");
  }

  MetalImage output =
      MetalImage::Create2D(image.Height(), image.Width(), image.Format(), true, true, false);
  DispatchRotate(image, output, RotationOp::Rotate90CW);
  image = std::move(output);
}

void Rotate90CCW(MetalImage& image) {
  if (image.Empty()) {
    return;
  }
  if (image.Format() != PixelFormat::R32FLOAT && image.Format() != PixelFormat::RGBA32FLOAT) {
    throw std::runtime_error("Metal convert utils: unsupported rotate format.");
  }

  MetalImage output =
      MetalImage::Create2D(image.Height(), image.Width(), image.Format(), true, true, false);
  DispatchRotate(image, output, RotationOp::Rotate90CCW);
  image = std::move(output);
}

}  // namespace alcedo::metal::utils

#endif
