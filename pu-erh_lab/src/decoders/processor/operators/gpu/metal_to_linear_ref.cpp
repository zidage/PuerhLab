//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL

#include "decoders/processor/operators/gpu/metal_to_linear_ref.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "image/metal_image.hpp"
#include "metal/compute_pipeline_cache.hpp"
#include "metal/metal_context.hpp"

namespace puerhlab {
namespace metal {
namespace {

struct WBParams {
  float black_level[4];
  float wb_scale[4];
};

struct ToLinearRefParams {
  float    white_level_scale;
  uint32_t width;
  uint32_t height;
  uint32_t stride;
};

constexpr uint32_t kRowAlignmentBytes = 256;

auto AlignRowBytes(size_t row_bytes) -> size_t {
  return ((row_bytes + kRowAlignmentBytes - 1) / kRowAlignmentBytes) * kRowAlignmentBytes;
}

auto GetPipelineState() -> NS::SharedPtr<MTL::ComputePipelineState> {
#ifndef PUERHLAB_METAL_TO_LINEAR_REF_METALLIB_PATH
  throw std::runtime_error("Metal ToLinearRef metallib path is not configured.");
#else
  return ComputePipelineCache::Instance().GetPipelineState(
      PUERHLAB_METAL_TO_LINEAR_REF_METALLIB_PATH, "to_linear_ref_r32f", "Metal ToLinearRef");
#endif
}

auto MakeSharedBuffer(size_t length) -> NS::SharedPtr<MTL::Buffer> {
  auto* device = MetalContext::Instance().Device();
  if (device == nullptr) {
    throw std::runtime_error("[FATAL] Metal ToLinearRef: Metal device is unavailable.");
  }

  auto buffer = NS::TransferPtr(
      device->newBuffer(static_cast<NS::UInteger>(length), MTL::ResourceStorageModeShared));
  if (!buffer) {
    throw std::runtime_error("[FATAL] Metal ToLinearRef: failed to allocate staging buffer.");
  }

  return buffer;
}

void DispatchToLinearRef(MetalImage& image, float white_level_scale, const WBParams& wb_params) {
  if (image.Empty()) {
    throw std::runtime_error("[ERROR] Metal ToLinearRef: image is empty.");
  }
  if (image.Format() != PixelFormat::R32FLOAT) {
    throw std::runtime_error("[ERROR] Metal ToLinearRef: expected R32FLOAT image.");
  }

  const auto row_bytes   = AlignRowBytes(static_cast<size_t>(image.Width()) * sizeof(float));
  const auto buffer_size = row_bytes * image.Height();
  const auto stride      = static_cast<uint32_t>(row_bytes / sizeof(float));

  auto staging_buffer = MakeSharedBuffer(buffer_size);

  auto* queue = MetalContext::Instance().Queue();
  if (queue == nullptr) {
    throw std::runtime_error("[ERROR] Metal ToLinearRef: Metal queue is unavailable.");
  }

  auto command_buffer = NS::RetainPtr(queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("[ERROR] Metal ToLinearRef: failed to create command buffer.");
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromTexture(image.Texture(), 0, 0, MTL::Origin{0, 0, 0},
                          MTL::Size{image.Width(), image.Height(), 1}, staging_buffer.get(), 0,
                          row_bytes, buffer_size);
    blit->endEncoding();
  }

  {
    auto pipeline = GetPipelineState();
    auto compute  = NS::RetainPtr(command_buffer->computeCommandEncoder());

    const ToLinearRefParams params{
        .white_level_scale = white_level_scale,
        .width             = image.Width(),
        .height            = image.Height(),
        .stride            = stride,
    };

    // Follow Apple's guidance for image kernels: align threadgroup width to the execution width
    // and fill the remaining threadgroup budget in Y.
    const auto thread_width  = std::max<NS::UInteger>(1, pipeline->threadExecutionWidth());
    const auto thread_height =
        std::max<NS::UInteger>(1, pipeline->maxTotalThreadsPerThreadgroup() / thread_width);
    const MTL::Size threads_per_threadgroup{thread_width, thread_height, 1};
    const MTL::Size threads_per_grid{image.Width(), image.Height(), 1};

    compute->setComputePipelineState(pipeline.get());
    compute->setBuffer(staging_buffer.get(), 0, 0);
    compute->setBytes(&params, sizeof(params), 1);
    compute->setBytes(&wb_params, sizeof(wb_params), 2);
    compute->dispatchThreads(threads_per_grid, threads_per_threadgroup);
    compute->endEncoding();
  }

  {
    auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
    blit->copyFromBuffer(staging_buffer.get(), 0, row_bytes, buffer_size,
                         MTL::Size{image.Width(), image.Height(), 1}, image.Texture(), 0, 0,
                         MTL::Origin{0, 0, 0});
    blit->endEncoding();
  }

  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

static auto CalculateBlackLevel(const libraw_rawdata_t& raw_data) -> std::array<float, 4> {
  const auto           base_black_level = static_cast<float>(raw_data.color.black);
  std::array<float, 4> black_level      = {
      base_black_level + static_cast<float>(raw_data.color.cblack[0]),
      base_black_level + static_cast<float>(raw_data.color.cblack[1]),
      base_black_level + static_cast<float>(raw_data.color.cblack[2]),
      base_black_level + static_cast<float>(raw_data.color.cblack[3])};

  if (raw_data.color.cblack[4] == 2 && raw_data.color.cblack[5] == 2) {
    for (unsigned int x = 0; x < raw_data.color.cblack[4]; ++x) {
      for (unsigned int y = 0; y < raw_data.color.cblack[5]; ++y) {
        const auto index   = y * 2 + x;
        black_level[index] = raw_data.color.cblack[6 + index];
      }
    }
  }

  return black_level;
}

static auto GetWBCoeff(const libraw_rawdata_t& raw_data) -> const float* {
  return raw_data.color.cam_mul;
}

}  // namespace

void ToLinearRef(MetalImage& img, LibRaw& raw_processor) {
  auto black_level = CalculateBlackLevel(raw_processor.imgdata.rawdata);
  auto wb          = GetWBCoeff(raw_processor.imgdata.rawdata);

  if (raw_processor.imgdata.color.as_shot_wb_applied != 1) {
    for (int c = 0; c < 4; ++c) {
      black_level[c] /= 65535.0f;
    }

    const float min     = black_level[0];
    const float maximum = raw_processor.imgdata.rawdata.color.maximum / 65535.0f - min;

    if (img.Format() != metal::PixelFormat::R16UINT) {
      throw std::runtime_error("Metal ToLinearRef: expected R16UINT Bayer input.");
    }
    if (maximum <= 0.f) {
      throw std::runtime_error("Metal ToLinearRef: invalid white level scale.");
    }

    MetalImage linearized;
    img.ConvertTo(linearized, PixelFormat::R32FLOAT, 1.0f / 65535.0f);
    img = std::move(linearized);

    WBParams wb_params = {};
    const float green_wb = wb[1];
    for (int c = 0; c < 4; ++c) {
      wb_params.black_level[c] = black_level[c];
      wb_params.wb_scale[c]    = 1.0f;
    }
    wb_params.wb_scale[0] = wb[0] / green_wb;
    wb_params.wb_scale[2] = wb[2] / green_wb;

    DispatchToLinearRef(img, maximum, wb_params);
  } else {
    MetalImage converted;
    img.ConvertTo(converted, PixelFormat::R32FLOAT);
    img = std::move(converted);
  }
}

};  // namespace metal
};  // namespace puerhlab

#endif
