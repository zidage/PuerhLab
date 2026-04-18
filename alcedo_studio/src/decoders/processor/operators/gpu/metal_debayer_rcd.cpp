//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL

#include "decoders/processor/operators/gpu/metal_debayer_rcd.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "image/metal_image.hpp"
#include "metal/compute_pipeline_cache.hpp"
#include "metal/metal_context.hpp"

namespace alcedo {
namespace metal {
namespace {

struct SinglePlaneParams {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t rgb_fc[4];
};

struct MergeParams {
  uint32_t width;
  uint32_t height;
  uint32_t plane_stride;
  uint32_t rgba_stride;
};

enum class Kernel : uint32_t {
  InitAndVH,
  GreenAtRB,
  PQDir,
  RBAtRB,
  RBAtG,
  MergeRGBA,
};

constexpr uint32_t kRowAlignmentBytes = 256;

auto AlignRowBytes(size_t row_bytes) -> size_t {
  return ((row_bytes + kRowAlignmentBytes - 1) / kRowAlignmentBytes) * kRowAlignmentBytes;
}

auto KernelNameFor(Kernel kernel) -> const char* {
  switch (kernel) {
    case Kernel::InitAndVH:
      return "rcd_init_and_vh";
    case Kernel::GreenAtRB:
      return "rcd_green_at_rb";
    case Kernel::PQDir:
      return "rcd_pq_dir";
    case Kernel::RBAtRB:
      return "rcd_rb_at_rb";
    case Kernel::RBAtG:
      return "rcd_rb_at_g";
    case Kernel::MergeRGBA:
      return "rcd_merge_rgba";
  }

  throw std::runtime_error("Metal Debayer RCD: unknown kernel.");
}

auto GetPipelineState(Kernel kernel) -> NS::SharedPtr<MTL::ComputePipelineState> {
#ifndef ALCEDO_METAL_DEBAYER_RCD_METALLIB_PATH
  throw std::runtime_error("Metal Debayer RCD metallib path is not configured.");
#else
  return ComputePipelineCache::Instance().GetPipelineState(
      ALCEDO_METAL_DEBAYER_RCD_METALLIB_PATH, KernelNameFor(kernel), "Metal Debayer RCD");
#endif
}

auto MakeSharedBuffer(size_t length) -> NS::SharedPtr<MTL::Buffer> {
  auto* device = MetalContext::Instance().Device();
  if (device == nullptr) {
    throw std::runtime_error("Metal Debayer RCD: Metal device is unavailable.");
  }

  auto buffer = NS::TransferPtr(
      device->newBuffer(static_cast<NS::UInteger>(length), MTL::ResourceStorageModeShared));
  if (!buffer) {
    throw std::runtime_error("Metal Debayer RCD: failed to allocate staging buffer.");
  }

  return buffer;
}

void DispatchThreads(MTL::ComputeCommandEncoder* encoder, MTL::ComputePipelineState* pipeline,
                     uint32_t width, uint32_t height) {
  const auto thread_width  = std::max<NS::UInteger>(1, pipeline->threadExecutionWidth());
  const auto thread_height =
      std::max<NS::UInteger>(1, pipeline->maxTotalThreadsPerThreadgroup() / thread_width);
  const MTL::Size threads_per_threadgroup{thread_width, thread_height, 1};
  const MTL::Size threads_per_grid{width, height, 1};
  encoder->dispatchThreads(threads_per_grid, threads_per_threadgroup);
}

}  // namespace

void Bayer2x2ToRGB_RCD(MetalImage& image, const BayerPattern2x2& pattern) {
  if (image.Empty()) {
    throw std::runtime_error("Metal Debayer RCD: input image is empty.");
  }
  if (image.Format() != PixelFormat::R32FLOAT) {
    throw std::runtime_error("Metal Debayer RCD: expected R32FLOAT Bayer input.");
  }

  const uint32_t width  = image.Width();
  const uint32_t height = image.Height();
  if (width == 0 || height == 0) {
    return;
  }

  const auto plane_row_bytes = AlignRowBytes(static_cast<size_t>(width) * sizeof(float));
  const auto plane_size      = plane_row_bytes * height;
  const auto plane_stride    = static_cast<uint32_t>(plane_row_bytes / sizeof(float));

  const auto rgba_row_bytes = AlignRowBytes(static_cast<size_t>(width) * sizeof(float) * 4U);
  const auto rgba_size      = rgba_row_bytes * height;
  const auto rgba_stride    = static_cast<uint32_t>(rgba_row_bytes / (sizeof(float) * 4U));

  auto raw_buffer  = MakeSharedBuffer(plane_size);
  auto r_buffer    = MakeSharedBuffer(plane_size);
  auto g_buffer    = MakeSharedBuffer(plane_size);
  auto b_buffer    = MakeSharedBuffer(plane_size);
  auto vh_buffer   = MakeSharedBuffer(plane_size);
  auto pq_buffer   = MakeSharedBuffer(plane_size);
  auto rgba_buffer = MakeSharedBuffer(rgba_size);

  MetalImage output =
      MetalImage::Create2D(width, height, PixelFormat::RGBA32FLOAT, true, true, false);

  auto* queue = MetalContext::Instance().Queue();
  if (queue == nullptr) {
    throw std::runtime_error("Metal Debayer RCD: Metal queue is unavailable.");
  }

  auto command_buffer = NS::RetainPtr(queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("Metal Debayer RCD: failed to create command buffer.");
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromTexture(image.Texture(), 0, 0, MTL::Origin{0, 0, 0}, MTL::Size{width, height, 1},
                          raw_buffer.get(), 0, plane_row_bytes, plane_size);
    blit->endEncoding();
  }

  const SinglePlaneParams plane_params{
      .width  = width,
      .height = height,
      .stride = plane_stride,
      .rgb_fc = {static_cast<uint32_t>(pattern.rgb_fc[0]),
                 static_cast<uint32_t>(pattern.rgb_fc[1]),
                 static_cast<uint32_t>(pattern.rgb_fc[2]),
                 static_cast<uint32_t>(pattern.rgb_fc[3])},
  };
  const MergeParams merge_params{
      .width       = width,
      .height      = height,
      .plane_stride = plane_stride,
      .rgba_stride = rgba_stride,
  };

  {
    auto pipeline = GetPipelineState(Kernel::InitAndVH);
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(raw_buffer.get(), 0, 0);
    compute->setBuffer(r_buffer.get(), 0, 1);
    compute->setBuffer(g_buffer.get(), 0, 2);
    compute->setBuffer(b_buffer.get(), 0, 3);
    compute->setBuffer(vh_buffer.get(), 0, 4);
    compute->setBytes(&plane_params, sizeof(plane_params), 5);
    DispatchThreads(compute.get(), pipeline.get(), width, height);
    compute->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(Kernel::GreenAtRB);
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(raw_buffer.get(), 0, 0);
    compute->setBuffer(vh_buffer.get(), 0, 1);
    compute->setBuffer(g_buffer.get(), 0, 2);
    compute->setBytes(&plane_params, sizeof(plane_params), 3);
    DispatchThreads(compute.get(), pipeline.get(), width, height);
    compute->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(Kernel::PQDir);
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(raw_buffer.get(), 0, 0);
    compute->setBuffer(pq_buffer.get(), 0, 1);
    compute->setBytes(&plane_params, sizeof(plane_params), 2);
    DispatchThreads(compute.get(), pipeline.get(), width, height);
    compute->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(Kernel::RBAtRB);
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(pq_buffer.get(), 0, 0);
    compute->setBuffer(g_buffer.get(), 0, 1);
    compute->setBuffer(r_buffer.get(), 0, 2);
    compute->setBuffer(b_buffer.get(), 0, 3);
    compute->setBytes(&plane_params, sizeof(plane_params), 4);
    DispatchThreads(compute.get(), pipeline.get(), width, height);
    compute->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(Kernel::RBAtG);
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(vh_buffer.get(), 0, 0);
    compute->setBuffer(g_buffer.get(), 0, 1);
    compute->setBuffer(r_buffer.get(), 0, 2);
    compute->setBuffer(b_buffer.get(), 0, 3);
    compute->setBytes(&plane_params, sizeof(plane_params), 4);
    DispatchThreads(compute.get(), pipeline.get(), width, height);
    compute->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(Kernel::MergeRGBA);
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(r_buffer.get(), 0, 0);
    compute->setBuffer(g_buffer.get(), 0, 1);
    compute->setBuffer(b_buffer.get(), 0, 2);
    compute->setBuffer(rgba_buffer.get(), 0, 3);
    compute->setBytes(&merge_params, sizeof(merge_params), 4);
    DispatchThreads(compute.get(), pipeline.get(), width, height);
    compute->endEncoding();
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromBuffer(rgba_buffer.get(), 0, rgba_row_bytes, rgba_size, MTL::Size{width, height, 1},
                         output.Texture(), 0, 0, MTL::Origin{0, 0, 0});
    blit->endEncoding();
  }

  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  image = std::move(output);
}

}  // namespace metal
}  // namespace alcedo

#endif
