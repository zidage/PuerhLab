//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_WEBGPU

#include "decoders/processor/operators/gpu/webgpu_debayer_rcd.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "image/webgpu_image.hpp"
#include "webgpu/webgpu_context.hpp"

namespace alcedo {
namespace webgpu {
namespace {

struct SinglePlaneParams {
  std::array<uint32_t, 4> rgb_fc;
  uint32_t                width;
  uint32_t                height;
  uint32_t                stride;
  uint32_t                padding;
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

auto               AlignRowBytes(size_t row_bytes) -> size_t {
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
  throw std::runtime_error("WebGPU Debayer RCD: unknown kernel.");
}

auto BindingCountFor(Kernel kernel) -> uint32_t {
  switch (kernel) {
    case Kernel::InitAndVH:
      return 6;
    case Kernel::GreenAtRB:
      return 4;
    case Kernel::PQDir:
      return 3;
    case Kernel::RBAtRB:
      return 5;
    case Kernel::RBAtG:
      return 5;
    case Kernel::MergeRGBA:
      return 5;
  }
  return 0;
}

auto BindingTypeFor(Kernel kernel, uint32_t binding) -> wgpu::BufferBindingType {
  switch (kernel) {
    case Kernel::InitAndVH:
      if (binding == 0) {
        return wgpu::BufferBindingType::ReadOnlyStorage;
      }
      if (binding == 5) {
        return wgpu::BufferBindingType::Uniform;
      }
      return wgpu::BufferBindingType::Storage;
    case Kernel::GreenAtRB:
      if (binding == 3) {
        return wgpu::BufferBindingType::Uniform;
      }
      if (binding <= 1) {
        return wgpu::BufferBindingType::ReadOnlyStorage;
      }
      return wgpu::BufferBindingType::Storage;
    case Kernel::PQDir:
      if (binding == 2) {
        return wgpu::BufferBindingType::Uniform;
      }
      return binding == 0 ? wgpu::BufferBindingType::ReadOnlyStorage
                          : wgpu::BufferBindingType::Storage;
    case Kernel::RBAtRB:
    case Kernel::RBAtG:
      if (binding == 4) {
        return wgpu::BufferBindingType::Uniform;
      }
      if (binding <= 1) {
        return wgpu::BufferBindingType::ReadOnlyStorage;
      }
      return wgpu::BufferBindingType::Storage;
    case Kernel::MergeRGBA:
      if (binding == 4) {
        return wgpu::BufferBindingType::Uniform;
      }
      if (binding <= 2) {
        return wgpu::BufferBindingType::ReadOnlyStorage;
      }
      return wgpu::BufferBindingType::Storage;
  }
  throw std::runtime_error("WebGPU Debayer RCD: unknown kernel binding type.");
}

auto ReadTextFile(const std::filesystem::path& path, const char* label) -> std::string {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error(std::string("WebGPU Debayer RCD: Failed to open ") + label + ": " +
                             path.string());
  }

  const auto size = file.tellg();
  if (size < 0) {
    throw std::runtime_error(std::string("WebGPU Debayer RCD: Failed to stat ") + label + ": " +
                             path.string());
  }

  std::string contents(static_cast<size_t>(size), '\0');
  file.seekg(0, std::ios::beg);
  if (!contents.empty() &&
      !file.read(contents.data(), static_cast<std::streamsize>(contents.size()))) {
    throw std::runtime_error(std::string("WebGPU Debayer RCD: Failed to read ") + label + ": " +
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
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create buffer.");
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

auto GetOrCreatePipeline(Kernel kernel) -> wgpu::ComputePipeline {
  static std::unordered_map<std::string, wgpu::ComputePipeline> cache;
  const std::string                                             key = KernelNameFor(kernel);
  auto                                                          it  = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

#ifndef ALCEDO_WEBGPU_DEBAYER_RCD_WGSL_PATH
#error "ALCEDO_WEBGPU_DEBAYER_RCD_WGSL_PATH must be defined when WebGPU Debayer RCD is enabled."
#endif

  auto&      device = WebGpuContext::Instance().Device();
  const auto wgsl_source =
      ReadTextFile(ALCEDO_WEBGPU_DEBAYER_RCD_WGSL_PATH, "debayer_rcd WGSL shader");

  wgpu::ShaderSourceWGSL       wgsl_desc{};
  wgpu::ShaderModuleDescriptor shader_desc{};
  wgsl_desc.code          = std::string_view(wgsl_source.data(), wgsl_source.size());
  shader_desc.nextInChain = &wgsl_desc;
  auto shader_module      = device.CreateShaderModule(&shader_desc);
  if (!shader_module.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create shader module.");
  }

  const auto                              binding_count = BindingCountFor(kernel);
  std::vector<wgpu::BindGroupLayoutEntry> entries;
  entries.reserve(binding_count);
  for (uint32_t i = 0; i < binding_count; ++i) {
    wgpu::BindGroupLayoutEntry e{};
    e.binding     = i;
    e.visibility  = wgpu::ShaderStage::Compute;
    e.buffer.type = BindingTypeFor(kernel, i);
    entries.push_back(e);
  }

  wgpu::BindGroupLayoutDescriptor bgl_desc{};
  bgl_desc.entryCount    = entries.size();
  bgl_desc.entries       = entries.data();
  auto bind_group_layout = device.CreateBindGroupLayout(&bgl_desc);
  if (!bind_group_layout.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create bind group layout.");
  }

  wgpu::PipelineLayoutDescriptor pl_desc{};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts     = &bind_group_layout;
  auto pipeline_layout         = device.CreatePipelineLayout(&pl_desc);
  if (!pipeline_layout.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create pipeline layout.");
  }

  wgpu::ComputePipelineDescriptor cp_desc{};
  cp_desc.layout             = pipeline_layout;
  cp_desc.compute.module     = shader_module;
  cp_desc.compute.entryPoint = key.c_str();
  auto pipeline              = device.CreateComputePipeline(&cp_desc);
  if (!pipeline.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create compute pipeline.");
  }

  cache[key] = pipeline;
  return pipeline;
}

auto CreateBindGroup(const wgpu::ComputePipeline&     pipeline,
                     const std::vector<wgpu::Buffer>& buffers) -> wgpu::BindGroup {
  std::vector<wgpu::BindGroupEntry> entries;
  entries.reserve(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    wgpu::BindGroupEntry e{};
    e.binding = static_cast<uint32_t>(i);
    e.buffer  = buffers[i];
    entries.push_back(e);
  }

  wgpu::BindGroupDescriptor desc{};
  desc.layout     = pipeline.GetBindGroupLayout(0);
  desc.entryCount = entries.size();
  desc.entries    = entries.data();
  return WebGpuContext::Instance().Device().CreateBindGroup(&desc);
}

}  // namespace

void Bayer2x2ToRGB_RCD(WebGpuImage& image, const BayerPattern2x2& pattern) {
  if (image.Empty()) {
    throw std::runtime_error("WebGPU Debayer RCD: input image is empty.");
  }
  if (image.Format() != PixelFormat::R32FLOAT) {
    throw std::runtime_error("WebGPU Debayer RCD: expected R32FLOAT Bayer input.");
  }

  const uint32_t width  = image.Width();
  const uint32_t height = image.Height();
  if (width == 0 || height == 0) {
    return;
  }

  const auto plane_row_bytes = AlignRowBytes(static_cast<size_t>(width) * sizeof(float));
  const auto plane_size      = plane_row_bytes * height;
  const auto plane_stride    = static_cast<uint32_t>(plane_row_bytes / sizeof(float));

  const auto rgba_row_bytes  = AlignRowBytes(static_cast<size_t>(width) * sizeof(float) * 4U);
  const auto rgba_size       = rgba_row_bytes * height;
  const auto rgba_stride     = static_cast<uint32_t>(rgba_row_bytes / (sizeof(float) * 4U));

  auto raw_buffer = MakeBuffer(plane_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  auto r_buffer   = MakeBuffer(plane_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  auto g_buffer   = MakeBuffer(plane_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  auto b_buffer   = MakeBuffer(plane_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  auto vh_buffer  = MakeBuffer(plane_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  auto pq_buffer  = MakeBuffer(plane_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  auto rgba_buffer = MakeBuffer(rgba_size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);

  WebGpuImage output;
  output.Create(width, height, PixelFormat::RGBA32FLOAT);

  const SinglePlaneParams plane_params{
      .rgb_fc = {static_cast<uint32_t>(pattern.rgb_fc[0]), static_cast<uint32_t>(pattern.rgb_fc[1]),
                 static_cast<uint32_t>(pattern.rgb_fc[2]),
                 static_cast<uint32_t>(pattern.rgb_fc[3])},
      .width  = width,
      .height = height,
      .stride = plane_stride,
      .padding = 0,
  };
  const MergeParams merge_params{
      .width        = width,
      .height       = height,
      .plane_stride = plane_stride,
      .rgba_stride  = rgba_stride,
  };

  auto params_buffer = MakeBuffer(sizeof(SinglePlaneParams),
                                  wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  WebGpuContext::Instance().Queue().WriteBuffer(params_buffer, 0, &plane_params,
                                                sizeof(plane_params));

  auto merge_params_buffer =
      MakeBuffer(sizeof(MergeParams), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  WebGpuContext::Instance().Queue().WriteBuffer(merge_params_buffer, 0, &merge_params,
                                                sizeof(merge_params));

  auto encoder = WebGpuContext::Instance().Device().CreateCommandEncoder();

  // Copy input texture → raw buffer
  {
    auto src    = MakeTextureCopy(image.Texture());
    auto dst    = MakeBufferCopy(raw_buffer, static_cast<uint32_t>(plane_row_bytes), height);
    auto extent = MakeExtent(width, height);
    encoder.CopyTextureToBuffer(&src, &dst, &extent);
  }

  // Single compute pass with all 6 kernels
  {
    auto compute = encoder.BeginComputePass();

    // Kernel 1: InitAndVH
    {
      auto pipeline   = GetOrCreatePipeline(Kernel::InitAndVH);
      auto bind_group = CreateBindGroup(
          pipeline, {raw_buffer, r_buffer, g_buffer, b_buffer, vh_buffer, params_buffer});
      compute.SetPipeline(pipeline);
      compute.SetBindGroup(0, bind_group);
      compute.DispatchWorkgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    // Kernel 2: GreenAtRB
    {
      auto pipeline   = GetOrCreatePipeline(Kernel::GreenAtRB);
      auto bind_group = CreateBindGroup(pipeline, {raw_buffer, vh_buffer, g_buffer, params_buffer});
      compute.SetPipeline(pipeline);
      compute.SetBindGroup(0, bind_group);
      compute.DispatchWorkgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    // Kernel 3: PQDir
    {
      auto pipeline   = GetOrCreatePipeline(Kernel::PQDir);
      auto bind_group = CreateBindGroup(pipeline, {raw_buffer, pq_buffer, params_buffer});
      compute.SetPipeline(pipeline);
      compute.SetBindGroup(0, bind_group);
      compute.DispatchWorkgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    // Kernel 4: RBAtRB
    {
      auto pipeline = GetOrCreatePipeline(Kernel::RBAtRB);
      auto bind_group =
          CreateBindGroup(pipeline, {pq_buffer, g_buffer, r_buffer, b_buffer, params_buffer});
      compute.SetPipeline(pipeline);
      compute.SetBindGroup(0, bind_group);
      compute.DispatchWorkgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    // Kernel 5: RBAtG
    {
      auto pipeline = GetOrCreatePipeline(Kernel::RBAtG);
      auto bind_group =
          CreateBindGroup(pipeline, {vh_buffer, g_buffer, r_buffer, b_buffer, params_buffer});
      compute.SetPipeline(pipeline);
      compute.SetBindGroup(0, bind_group);
      compute.DispatchWorkgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    // Kernel 6: MergeRGBA
    {
      auto pipeline   = GetOrCreatePipeline(Kernel::MergeRGBA);
      auto bind_group = CreateBindGroup(
          pipeline, {r_buffer, g_buffer, b_buffer, rgba_buffer, merge_params_buffer});
      compute.SetPipeline(pipeline);
      compute.SetBindGroup(0, bind_group);
      compute.DispatchWorkgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    compute.End();
  }

  // Copy rgba buffer → output texture
  {
    auto src    = MakeBufferCopy(rgba_buffer, static_cast<uint32_t>(rgba_row_bytes), height);
    auto dst    = MakeTextureCopy(output.Texture());
    auto extent = MakeExtent(width, height);
    encoder.CopyBufferToTexture(&src, &dst, &extent);
  }

  SubmitAndWait(encoder.Finish());

  image = std::move(output);
}

}  // namespace webgpu
}  // namespace alcedo

#endif
