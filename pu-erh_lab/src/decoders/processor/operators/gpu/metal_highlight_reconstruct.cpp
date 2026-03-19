//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL

#include "decoders/processor/operators/gpu/metal_highlight_reconstruct.hpp"

#include <opencv2/core/hal/interface.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "image/metal_image.hpp"
#include "metal/compute_pipeline_cache.hpp"
#include "metal/metal_context.hpp"

namespace puerhlab {
namespace metal {
namespace {

constexpr float kHilightMagic = 0.987f;

struct HighlightParams {
  float    correction[4];
  float    chrominance[4];
  float    clip_val;
  float    lo_clip_val;
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_size;
  uint32_t rgb_fc[4];
};

enum class Kernel : uint32_t {
  BuildMask,
  DilateMask,
  ChrominanceContrib,
  Reconstruct,
};

constexpr uint32_t kRowAlignmentBytes = 256;

auto AlignRowBytes(size_t row_bytes) -> size_t {
  return ((row_bytes + kRowAlignmentBytes - 1) / kRowAlignmentBytes) * kRowAlignmentBytes;
}

auto RoundSize(int size, int alignment) -> int {
  return ((size % alignment) == 0) ? size : ((size - 1) / alignment + 1) * alignment;
}

auto FC(const BayerPattern2x2& pattern, int y, int x) -> int {
  return pattern.rgb_fc[BayerCellIndex(y, x)];
}

auto KernelNameFor(Kernel kernel) -> const char* {
  switch (kernel) {
    case Kernel::BuildMask:
      return "hlr_build_mask";
    case Kernel::DilateMask:
      return "hlr_dilate_mask";
    case Kernel::ChrominanceContrib:
      return "hlr_chrominance_contrib";
    case Kernel::Reconstruct:
      return "hlr_reconstruct";
  }

  throw std::runtime_error("Metal HighlightReconstruct: unknown kernel.");
}

auto GetPipelineState(Kernel kernel) -> NS::SharedPtr<MTL::ComputePipelineState> {
#ifndef PUERHLAB_METAL_HIGHLIGHT_RECONSTRUCT_METALLIB_PATH
  throw std::runtime_error("Metal HighlightReconstruct metallib path is not configured.");
#else
  return ComputePipelineCache::Instance().GetPipelineState(
      PUERHLAB_METAL_HIGHLIGHT_RECONSTRUCT_METALLIB_PATH, KernelNameFor(kernel),
      "Metal HighlightReconstruct");
#endif
}

auto MakeSharedBuffer(size_t length) -> NS::SharedPtr<MTL::Buffer> {
  auto* device = MetalContext::Instance().Device();
  if (device == nullptr) {
    throw std::runtime_error("Metal HighlightReconstruct: Metal device is unavailable.");
  }

  auto buffer = NS::TransferPtr(
      device->newBuffer(static_cast<NS::UInteger>(length), MTL::ResourceStorageModeShared));
  if (!buffer) {
    throw std::runtime_error("Metal HighlightReconstruct: failed to allocate shared buffer.");
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

}  // namespace

void HighlightReconstruct(MetalImage& img, LibRaw& raw_processor, const BayerPattern2x2& pattern) {
  if (img.Empty()) {
    throw std::runtime_error("Metal HighlightReconstruct: input image is empty.");
  }
  if (img.Format() != PixelFormat::R32FLOAT) {
    throw std::runtime_error("Metal HighlightReconstruct: expected R32FLOAT Bayer input.");
  }

  const uint32_t width  = img.Width();
  const uint32_t height = img.Height();
  if (width == 0 || height == 0) {
    return;
  }

  const auto row_bytes = AlignRowBytes(static_cast<size_t>(width) * sizeof(float));
  const auto img_size  = row_bytes * height;
  const auto stride    = static_cast<uint32_t>(row_bytes / CV_ELEM_SIZE(CV_32FC1));

  const auto flag_row_bytes = AlignRowBytes(static_cast<size_t>(width) * sizeof(uint32_t));
  const auto flag_size      = flag_row_bytes * height;
  const auto flag_stride    = static_cast<uint32_t>(flag_row_bytes / sizeof(uint32_t));

  HighlightParams params = {};
  const float*    cam_mul = raw_processor.imgdata.color.cam_mul;
  params.correction[0]    = cam_mul[0] / cam_mul[1];
  params.correction[1]    = 1.0f;
  params.correction[2]    = cam_mul[2] / cam_mul[1];
  params.clip_val         = kHilightMagic;
  params.lo_clip_val      = 0.99f * params.clip_val;
  params.width            = width;
  params.height           = height;
  params.stride           = stride;
  params.m_width          = width / 3;
  params.m_height         = height / 3;
  params.m_size           = static_cast<uint32_t>(
      RoundSize(static_cast<int>((params.m_width + 1) * (params.m_height + 1)), 16));
  params.rgb_fc[0]        = static_cast<uint32_t>(pattern.rgb_fc[0]);
  params.rgb_fc[1]        = static_cast<uint32_t>(pattern.rgb_fc[1]);
  params.rgb_fc[2]        = static_cast<uint32_t>(pattern.rgb_fc[2]);
  params.rgb_fc[3]        = static_cast<uint32_t>(pattern.rgb_fc[3]);

  auto input_buffer   = MakeSharedBuffer(img_size);
  auto mask_buffer    = MakeSharedBuffer(static_cast<size_t>(6) * params.m_size * sizeof(uint8_t));
  auto contrib_buffer = MakeSharedBuffer(img_size);
  auto flag_buffer    = MakeSharedBuffer(flag_size);

  std::memset(mask_buffer->contents(), 0, static_cast<size_t>(6) * params.m_size * sizeof(uint8_t));
  std::memset(contrib_buffer->contents(), 0, img_size);
  std::memset(flag_buffer->contents(), 0, flag_size);

  auto* queue = MetalContext::Instance().Queue();
  if (queue == nullptr) {
    throw std::runtime_error("Metal HighlightReconstruct: Metal queue is unavailable.");
  }

  auto mask_command_buffer = NS::RetainPtr(queue->commandBuffer());
  if (!mask_command_buffer) {
    throw std::runtime_error("Metal HighlightReconstruct: failed to create command buffer.");
  }

  {
    auto blit = NS::RetainPtr(mask_command_buffer->blitCommandEncoder());
    blit->copyFromTexture(img.Texture(), 0, 0, MTL::Origin{0, 0, 0}, MTL::Size{width, height, 1},
                          input_buffer.get(), 0, row_bytes, img_size);
    blit->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(Kernel::BuildMask);
    auto compute  = NS::RetainPtr(mask_command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(input_buffer.get(), 0, 0);
    compute->setBuffer(mask_buffer.get(), 0, 1);
    compute->setBytes(&params, sizeof(params), 2);
    DispatchThreads(compute.get(), pipeline.get(), params.m_width, params.m_height);
    compute->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(Kernel::DilateMask);
    auto compute  = NS::RetainPtr(mask_command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(mask_buffer.get(), 0, 0);
    compute->setBytes(&params, sizeof(params), 1);
    DispatchThreads(compute.get(), pipeline.get(), params.m_width, params.m_height);
    compute->endEncoding();
  }

  {
    auto pipeline = GetPipelineState(Kernel::ChrominanceContrib);
    auto compute  = NS::RetainPtr(mask_command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(input_buffer.get(), 0, 0);
    compute->setBuffer(mask_buffer.get(), 0, 1);
    compute->setBuffer(contrib_buffer.get(), 0, 2);
    compute->setBuffer(flag_buffer.get(), 0, 3);
    compute->setBytes(&params, sizeof(params), 4);
    DispatchThreads(compute.get(), pipeline.get(), width, height);
    compute->endEncoding();
  }

  mask_command_buffer->commit();
  mask_command_buffer->waitUntilCompleted();

  const auto* mask_data = static_cast<const uint8_t*>(mask_buffer->contents());
  bool        anyclipped = false;
  for (size_t i = 0; i < static_cast<size_t>(3) * params.m_size; ++i) {
    if (mask_data[i] != 0) {
      anyclipped = true;
      break;
    }
  }
  if (!anyclipped) {
    return;
  }

  std::array<float, 4> sums = {0.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 4> cnts = {0.0f, 0.0f, 0.0f, 0.0f};

  const auto* contrib_data = static_cast<const float*>(contrib_buffer->contents());
  const auto* flag_data    = static_cast<const uint32_t*>(flag_buffer->contents());
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      const size_t flag_index = static_cast<size_t>(row) * flag_stride + col;
      if (flag_data[flag_index] == 0U) {
        continue;
      }

      const int color = FC(pattern, static_cast<int>(row), static_cast<int>(col));
      const size_t contrib_index = static_cast<size_t>(row) * stride + col;
      sums[color] += contrib_data[contrib_index];
      cnts[color] += 1.0f;
    }
  }

  for (int c = 0; c < 3; ++c) {
    params.chrominance[c] = (cnts[c] > 80.0f) ? (sums[c] / cnts[c]) : 0.0f;
  }

  auto output_buffer = MakeSharedBuffer(img_size);

  auto reconstruct_command_buffer = NS::RetainPtr(queue->commandBuffer());
  if (!reconstruct_command_buffer) {
    throw std::runtime_error("Metal HighlightReconstruct: failed to create command buffer.");
  }

  {
    auto pipeline = GetPipelineState(Kernel::Reconstruct);
    auto compute  = NS::RetainPtr(reconstruct_command_buffer->computeCommandEncoder());
    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(input_buffer.get(), 0, 0);
    compute->setBuffer(output_buffer.get(), 0, 1);
    compute->setBytes(&params, sizeof(params), 2);
    DispatchThreads(compute.get(), pipeline.get(), width, height);
    compute->endEncoding();
  }

  {
    auto blit = NS::RetainPtr(reconstruct_command_buffer->blitCommandEncoder());
    blit->copyFromBuffer(output_buffer.get(), 0, row_bytes, img_size,
                         MTL::Size{width, height, 1}, img.Texture(), 0, 0,
                         MTL::Origin{0, 0, 0});
    blit->endEncoding();
  }

  reconstruct_command_buffer->commit();
  reconstruct_command_buffer->waitUntilCompleted();
}

};  // namespace metal
};  // namespace puerhlab

#endif
