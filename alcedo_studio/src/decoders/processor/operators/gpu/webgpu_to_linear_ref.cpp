//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_WEBGPU

#include "decoders/processor/operators/gpu/webgpu_to_linear_ref.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "decoders/processor/raw_normalization.hpp"
#include "image/webgpu_image.hpp"
#include "webgpu/webgpu_context.hpp"

namespace alcedo {
namespace webgpu {
namespace {

struct WBParams {
  float    black_level[4];
  float    white_level[4];
  float    wb_multipliers[4];
  uint32_t apply_white_balance;
  uint32_t padding[3];
};

struct ToLinearRefParams {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t tile_width;
  uint32_t tile_height;
  uint32_t black_tile_width;
  uint32_t black_tile_height;
  uint32_t padding;
  uint32_t raw_fc[36];
};

constexpr uint32_t kRowAlignmentBytes = 256;

auto               AlignRowBytes(size_t row_bytes) -> size_t {
  return ((row_bytes + kRowAlignmentBytes - 1) / kRowAlignmentBytes) * kRowAlignmentBytes;
}

auto ReadTextFile(const std::filesystem::path& path, const char* label) -> std::string {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error(std::string("WebGPU ToLinearRef: Failed to open ") + label + ": " +
                             path.string());
  }

  const auto size = file.tellg();
  if (size < 0) {
    throw std::runtime_error(std::string("WebGPU ToLinearRef: Failed to stat ") + label + ": " +
                             path.string());
  }

  std::string contents(static_cast<size_t>(size), '\0');
  file.seekg(0, std::ios::beg);
  if (!contents.empty() &&
      !file.read(contents.data(), static_cast<std::streamsize>(contents.size()))) {
    throw std::runtime_error(std::string("WebGPU ToLinearRef: Failed to read ") + label + ": " +
                             path.string());
  }
  return contents;
}

auto MakeBuffer(uint64_t size, wgpu::BufferUsage usage, bool mapped_at_creation = false)
    -> wgpu::Buffer {
  wgpu::BufferDescriptor descriptor{};
  descriptor.usage            = usage;
  descriptor.size             = size;
  descriptor.mappedAtCreation = mapped_at_creation;
  auto buffer                 = WebGpuContext::Instance().Device().CreateBuffer(&descriptor);
  if (!buffer.Get()) {
    throw std::runtime_error("WebGPU ToLinearRef: Failed to create buffer.");
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

auto GetOrCreatePipeline() -> wgpu::ComputePipeline {
  static wgpu::ComputePipeline pipeline = nullptr;
  if (pipeline) {
    return pipeline;
  }

#ifndef ALCEDO_WEBGPU_TO_LINEAR_REF_WGSL_PATH
#error "ALCEDO_WEBGPU_TO_LINEAR_REF_WGSL_PATH must be defined when WebGPU ToLinearRef is enabled."
#endif

  auto&      device = WebGpuContext::Instance().Device();
  const auto wgsl_source =
      ReadTextFile(ALCEDO_WEBGPU_TO_LINEAR_REF_WGSL_PATH, "to_linear_ref WGSL shader");

  wgpu::ShaderSourceWGSL       wgsl_desc{};
  wgpu::ShaderModuleDescriptor shader_desc{};
  wgsl_desc.code          = std::string_view(wgsl_source.data(), wgsl_source.size());
  shader_desc.nextInChain = &wgsl_desc;
  auto shader_module      = device.CreateShaderModule(&shader_desc);
  if (!shader_module.Get()) {
    throw std::runtime_error("WebGPU ToLinearRef: Failed to create shader module.");
  }

  std::array<wgpu::BindGroupLayoutEntry, 4> entries{};

  entries[0].binding     = 0;
  entries[0].visibility  = wgpu::ShaderStage::Compute;
  entries[0].buffer.type = wgpu::BufferBindingType::Storage;

  entries[1].binding     = 1;
  entries[1].visibility  = wgpu::ShaderStage::Compute;
  entries[1].buffer.type = wgpu::BufferBindingType::Uniform;

  entries[2].binding     = 2;
  entries[2].visibility  = wgpu::ShaderStage::Compute;
  entries[2].buffer.type = wgpu::BufferBindingType::Uniform;

  entries[3].binding     = 3;
  entries[3].visibility  = wgpu::ShaderStage::Compute;
  entries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

  wgpu::BindGroupLayoutDescriptor bgl_desc{};
  bgl_desc.entryCount    = entries.size();
  bgl_desc.entries       = entries.data();
  auto bind_group_layout = device.CreateBindGroupLayout(&bgl_desc);
  if (!bind_group_layout.Get()) {
    throw std::runtime_error("WebGPU ToLinearRef: Failed to create bind group layout.");
  }

  wgpu::PipelineLayoutDescriptor pl_desc{};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts     = &bind_group_layout;
  auto pipeline_layout         = device.CreatePipelineLayout(&pl_desc);
  if (!pipeline_layout.Get()) {
    throw std::runtime_error("WebGPU ToLinearRef: Failed to create pipeline layout.");
  }

  wgpu::ComputePipelineDescriptor cp_desc{};
  cp_desc.layout             = pipeline_layout;
  cp_desc.compute.module     = shader_module;
  cp_desc.compute.entryPoint = "main";
  pipeline                   = device.CreateComputePipeline(&cp_desc);
  if (!pipeline.Get()) {
    throw std::runtime_error("WebGPU ToLinearRef: Failed to create compute pipeline.");
  }

  return pipeline;
}

void DispatchToLinearRef(WebGpuImage& image, const WBParams& wb_params,
                         const RawCfaPattern& pattern, const std::vector<float>& black_pattern,
                         uint32_t black_tile_width, uint32_t black_tile_height) {
  if (image.Empty()) {
    throw std::runtime_error("[ERROR] WebGPU ToLinearRef: image is empty.");
  }
  if (image.Format() != PixelFormat::R32FLOAT) {
    throw std::runtime_error("[ERROR] WebGPU ToLinearRef: expected R32FLOAT image.");
  }

  const auto row_bytes   = AlignRowBytes(static_cast<size_t>(image.Width()) * sizeof(float));
  const auto buffer_size = row_bytes * image.Height();
  const auto stride      = static_cast<uint32_t>(row_bytes / sizeof(float));

  auto       staging_buffer =
      MakeBuffer(buffer_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                                  wgpu::BufferUsage::CopyDst);

  const auto black_pattern_size = std::max<size_t>(black_pattern.size(), 1) * sizeof(float);
  auto       black_pattern_buffer =
      MakeBuffer(black_pattern_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  {
    std::vector<float> padded_black(std::max<size_t>(black_pattern.size(), 1), 0.0f);
    if (!black_pattern.empty()) {
      std::copy(black_pattern.begin(), black_pattern.end(), padded_black.begin());
    }
    WebGpuContext::Instance().Queue().WriteBuffer(black_pattern_buffer, 0, padded_black.data(),
                                                  padded_black.size() * sizeof(float));
  }

  // Create uniform buffers
  ToLinearRefParams params = {};
  params.width             = image.Width();
  params.height            = image.Height();
  params.stride            = stride;
  if (pattern.kind == RawCfaKind::XTrans6x6) {
    params.tile_width  = 6;
    params.tile_height = 6;
    for (int i = 0; i < 36; ++i) {
      params.raw_fc[i] = static_cast<uint32_t>(pattern.xtrans_pattern.raw_fc[i]);
    }
  } else {
    params.tile_width  = 2;
    params.tile_height = 2;
    for (int i = 0; i < 4; ++i) {
      params.raw_fc[i] = static_cast<uint32_t>(pattern.bayer_pattern.raw_fc[i]);
    }
  }
  params.black_tile_width        = black_tile_width;
  params.black_tile_height       = black_tile_height;

  constexpr size_t kUniformAlign = 256;
  const auto       params_buffer_size =
      (sizeof(ToLinearRefParams) + kUniformAlign - 1) & ~(kUniformAlign - 1);
  auto params_buffer =
      MakeBuffer(params_buffer_size, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  WebGpuContext::Instance().Queue().WriteBuffer(params_buffer, 0, &params, sizeof(params));

  const auto wb_buffer_size = (sizeof(WBParams) + kUniformAlign - 1) & ~(kUniformAlign - 1);
  auto       wb_buffer =
      MakeBuffer(wb_buffer_size, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  WebGpuContext::Instance().Queue().WriteBuffer(wb_buffer, 0, &wb_params, sizeof(wb_params));

  auto                                pipeline = GetOrCreatePipeline();

  std::array<wgpu::BindGroupEntry, 4> bg_entries{};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer  = staging_buffer;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer  = params_buffer;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer  = wb_buffer;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer  = black_pattern_buffer;

  wgpu::BindGroupDescriptor bg_desc{};
  bg_desc.layout     = pipeline.GetBindGroupLayout(0);
  bg_desc.entryCount = bg_entries.size();
  bg_desc.entries    = bg_entries.data();
  auto bind_group    = WebGpuContext::Instance().Device().CreateBindGroup(&bg_desc);

  // Single encoder: texture→staging, compute in-place, staging→texture.
  // Within one command buffer WebGPU guarantees in-order execution, so the
  // staging buffer transitions CopyDst→Storage→CopySrc without explicit barriers.
  // Reducing to one Submit/Wait eliminates the D3D12 alertable-wait re-entrancy
  // that caused stack overflows when three fences were waited on sequentially.
  auto encoder       = WebGpuContext::Instance().Device().CreateCommandEncoder();

  {
    auto src    = MakeTextureCopy(image.Texture());
    auto dst    = MakeBufferCopy(staging_buffer, static_cast<uint32_t>(row_bytes), image.Height());
    auto extent = MakeExtent(image.Width(), image.Height());
    encoder.CopyTextureToBuffer(&src, &dst, &extent);
  }

  {
    auto compute = encoder.BeginComputePass();
    compute.SetPipeline(pipeline);
    compute.SetBindGroup(0, bind_group);
    compute.DispatchWorkgroups((image.Width() + 7) / 8, (image.Height() + 7) / 8, 1);
    compute.End();
  }

  {
    auto src    = MakeBufferCopy(staging_buffer, static_cast<uint32_t>(row_bytes), image.Height());
    auto dst    = MakeTextureCopy(image.Texture());
    auto extent = MakeExtent(image.Width(), image.Height());
    encoder.CopyBufferToTexture(&src, &dst, &extent);
  }

  SubmitAndWait(encoder.Finish());
}

static auto GetWBCoeff(const libraw_rawdata_t& raw_data) -> const float* {
  return raw_data.color.cam_mul;
}

static auto GetPatternBlackLevels(const libraw_rawdata_t& raw_data) -> std::vector<float> {
  const uint32_t tile_width  = raw_data.color.cblack[4];
  const uint32_t tile_height = raw_data.color.cblack[5];
  const uint32_t entries     = tile_width * tile_height;
  if (entries == 0U) {
    return {};
  }

  std::vector<float> pattern_black(entries, 0.0f);
  for (uint32_t i = 0; i < entries; ++i) {
    pattern_black[i] = static_cast<float>(raw_data.color.cblack[6 + i]);
  }
  return pattern_black;
}

}  // namespace

void ToLinearRef(WebGpuImage& img, LibRaw& raw_processor, const RawCfaPattern& pattern) {
  const auto     raw_curve     = raw_norm::BuildLinearizationCurve(raw_processor.imgdata.rawdata);
  const auto     wb            = GetWBCoeff(raw_processor.imgdata.rawdata);
  auto           black_pattern = GetPatternBlackLevels(raw_processor.imgdata.rawdata);
  const uint32_t black_tile_width  = raw_processor.imgdata.rawdata.color.cblack[4];
  const uint32_t black_tile_height = raw_processor.imgdata.rawdata.color.cblack[5];

  if (img.Format() != PixelFormat::R16UINT) {
    throw std::runtime_error("WebGPU ToLinearRef: expected R16UINT raw input.");
  }

  WebGpuImage linearized;
  img.ConvertTo(linearized, PixelFormat::R32FLOAT);
  img                = std::move(linearized);

  WBParams wb_params = {};
  for (int c = 0; c < 4; ++c) {
    wb_params.black_level[c]    = raw_curve.black_level[c];
    wb_params.white_level[c]    = raw_curve.white_level[c];
    wb_params.wb_multipliers[c] = wb[c];
  }
  wb_params.apply_white_balance = raw_processor.imgdata.color.as_shot_wb_applied != 1 ? 1u : 0u;

  DispatchToLinearRef(img, wb_params, pattern, black_pattern, black_tile_width, black_tile_height);
}

}  // namespace webgpu
}  // namespace alcedo

#endif
