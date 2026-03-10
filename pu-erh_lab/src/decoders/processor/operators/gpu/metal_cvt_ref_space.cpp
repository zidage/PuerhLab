//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL

#include "decoders/processor/operators/gpu/metal_cvt_ref_space.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "image/metal_image.hpp"
#include "metal/compute_pipeline_cache.hpp"
#include "metal/metal_context.hpp"

namespace puerhlab {
namespace metal {
namespace {

struct InverseCamMulParams {
  float    scale_r;
  float    scale_g;
  float    scale_b;
  float    scale_a;
  uint32_t width;
  uint32_t height;
  uint32_t stride;
};

constexpr float kMinGain             = 1e-6f;
constexpr uint32_t kRowAlignmentBytes = 256;

auto AlignRowBytes(size_t row_bytes) -> size_t {
  return ((row_bytes + kRowAlignmentBytes - 1) / kRowAlignmentBytes) * kRowAlignmentBytes;
}

auto GetPipelineState() -> NS::SharedPtr<MTL::ComputePipelineState> {
#ifndef PUERHLAB_METAL_CVT_REF_SPACE_METALLIB_PATH
  throw std::runtime_error("Metal ApplyInverseCamMul metallib path is not configured.");
#else
  return ComputePipelineCache::Instance().GetPipelineState(
      PUERHLAB_METAL_CVT_REF_SPACE_METALLIB_PATH, "apply_inverse_cam_mul_rgba32f",
      "Metal ApplyInverseCamMul");
#endif
}

auto MakeSharedBuffer(size_t length) -> NS::SharedPtr<MTL::Buffer> {
  auto* device = MetalContext::Instance().Device();
  if (device == nullptr) {
    throw std::runtime_error("Metal ApplyInverseCamMul: Metal device is unavailable.");
  }

  auto buffer = NS::TransferPtr(
      device->newBuffer(static_cast<NS::UInteger>(length), MTL::ResourceStorageModeShared));
  if (!buffer) {
    throw std::runtime_error("Metal ApplyInverseCamMul: failed to allocate staging buffer.");
  }
  return buffer;
}

void DispatchInverseCamMul(MetalImage& image, const float* cam_mul) {
  const auto row_bytes   = AlignRowBytes(static_cast<size_t>(image.Width()) * sizeof(float) * 4U);
  const auto buffer_size = row_bytes * image.Height();
  const auto stride      = static_cast<uint32_t>(row_bytes / (sizeof(float) * 4U));

  auto staging_buffer = MakeSharedBuffer(buffer_size);

  auto* queue = MetalContext::Instance().Queue();
  if (queue == nullptr) {
    throw std::runtime_error("Metal ApplyInverseCamMul: Metal queue is unavailable.");
  }

  auto command_buffer = NS::TransferPtr(queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("Metal ApplyInverseCamMul: failed to create command buffer.");
  }

  {
    auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
    blit->copyFromTexture(image.Texture(), 0, 0, MTL::Origin{0, 0, 0},
                          MTL::Size{image.Width(), image.Height(), 1}, staging_buffer.get(), 0,
                          row_bytes, buffer_size);
    blit->endEncoding();
  }

  {
    auto pipeline = GetPipelineState();
    auto compute  = NS::TransferPtr(command_buffer->computeCommandEncoder());
    const float g = std::max(cam_mul[1], kMinGain);
    const InverseCamMulParams params{
        .scale_r = g / std::max(cam_mul[0], kMinGain),
        .scale_g = 1.0f,
        .scale_b = g / std::max(cam_mul[2], kMinGain),
        .scale_a = 1.0f,
        .width   = image.Width(),
        .height  = image.Height(),
        .stride  = stride,
    };

    const auto thread_width  = std::max<NS::UInteger>(1, pipeline->threadExecutionWidth());
    const auto thread_height =
        std::max<NS::UInteger>(1, pipeline->maxTotalThreadsPerThreadgroup() / thread_width);
    const MTL::Size threads_per_threadgroup{thread_width, thread_height, 1};
    const MTL::Size threads_per_grid{image.Width(), image.Height(), 1};

    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(staging_buffer.get(), 0, 0);
    compute->setBytes(&params, sizeof(params), 1);
    compute->dispatchThreads(threads_per_grid, threads_per_threadgroup);
    compute->endEncoding();
  }

  {
    auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
    blit->copyFromBuffer(staging_buffer.get(), 0, row_bytes, buffer_size,
                         MTL::Size{image.Width(), image.Height(), 1}, image.Texture(), 0, 0,
                         MTL::Origin{0, 0, 0});
    blit->endEncoding();
  }

  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

}  // namespace

void ApplyInverseCamMul(MetalImage& img, const float* cam_mul) {
  if (img.Empty()) {
    return;
  }
  if (img.Format() != PixelFormat::RGBA32FLOAT) {
    throw std::runtime_error("Metal ApplyInverseCamMul: expected RGBA32FLOAT image.");
  }
  if (cam_mul == nullptr) {
    throw std::runtime_error("Metal ApplyInverseCamMul: cam_mul is null.");
  }

  DispatchInverseCamMul(img, cam_mul);
}

}  // namespace metal
}  // namespace puerhlab

#endif
