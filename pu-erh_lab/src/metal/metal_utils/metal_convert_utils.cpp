//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL

#include "metal/metal_utils/metal_convert_utils.hpp"

#include "image/metal_image.hpp"
#include "metal/metal_context.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace puerhlab::metal::utils {
namespace {

struct ConvertParams {
  float    alpha;
  float    beta;
  uint32_t width;
  uint32_t height;
  uint32_t src_stride;
  uint32_t dst_stride;
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

auto LoadLibrary() -> NS::SharedPtr<MTL::Library> {
#ifndef PUERHLAB_METAL_UTILS_METALLIB_PATH
  throw std::runtime_error("Metal convert utils metallib path is not configured.");
#else
  auto* device = MetalContext::Instance().Device();
  if (device == nullptr) {
    throw std::runtime_error("Metal convert utils: Metal device is unavailable.");
  }

  NS::Error* error = nullptr;
  auto metallib_path = NS::String::string(PUERHLAB_METAL_UTILS_METALLIB_PATH, NS::UTF8StringEncoding);
  auto library       = NS::TransferPtr(device->newLibrary(metallib_path, &error));
  if (!library) {
    std::string error_message = "Metal convert utils: failed to load metallib.";
    if (error != nullptr) {
      error_message += " ";
      error_message += error->localizedDescription()->utf8String();
    }
    throw std::runtime_error(error_message);
  }
  return library;
#endif
}

auto GetPipelineState(PixelFormat src_format, PixelFormat dst_format)
    -> NS::SharedPtr<MTL::ComputePipelineState> {
  static std::mutex pipeline_mutex;
  static auto       library = LoadLibrary();
  static std::unordered_map<uint32_t, NS::SharedPtr<MTL::ComputePipelineState>> cache;

  const auto key = BuildPipelineKey(src_format, dst_format);
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (const auto it = cache.find(key); it != cache.end()) {
      return it->second;
    }
  }

  const auto* kernel_name = KernelNameFor(src_format, dst_format);
  if (kernel_name == nullptr) {
    throw std::runtime_error("Metal convert utils: unsupported conversion pair.");
  }

  auto function_name = NS::String::string(kernel_name, NS::UTF8StringEncoding);
  auto function      = NS::TransferPtr(library->newFunction(function_name));
  if (!function) {
    throw std::runtime_error("Metal convert utils: failed to load compute function from metallib.");
  }

  NS::Error* error = nullptr;
  auto* device     = MetalContext::Instance().Device();
  auto  pipeline   = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  if (!pipeline) {
    std::string error_message = "Metal convert utils: failed to create compute pipeline.";
    if (error != nullptr) {
      error_message += " ";
      error_message += error->localizedDescription()->utf8String();
    }
    throw std::runtime_error(error_message);
  }

  std::lock_guard<std::mutex> lock(pipeline_mutex);
  cache.emplace(key, pipeline);
  return pipeline;
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

  auto command_buffer = NS::TransferPtr(queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("Metal convert utils: failed to create command buffer.");
  }

  {
    auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
    blit->copyFromTexture(src.Texture(), 0, 0, MTL::Origin{0, 0, 0},
                          MTL::Size{src.Width(), src.Height(), 1}, src_buffer.get(), 0,
                          src_row_bytes, src_size);
    blit->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(src.Format(), dst.Format());
    auto compute  = NS::TransferPtr(command_buffer->computeCommandEncoder());
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

    const auto thread_width = pipeline->threadExecutionWidth();
    const auto thread_height =
        std::max<NS::UInteger>(1, pipeline->maxTotalThreadsPerThreadgroup() / thread_width);
    const MTL::Size threads_per_group{thread_width, thread_height, 1};
    const MTL::Size threadgroups{
        (src.Width() + threads_per_group.width - 1) / threads_per_group.width,
        (src.Height() + threads_per_group.height - 1) / threads_per_group.height,
        1,
    };

    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(src_buffer.get(), 0, 0);
    compute->setBuffer(dst_buffer.get(), 0, 1);
    compute->setBytes(&params, sizeof(params), 2);
    compute->dispatchThreadgroups(threadgroups, threads_per_group);
    compute->endEncoding();
  }

  {
    auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
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

}  // namespace puerhlab::metal::utils

#endif
