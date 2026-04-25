//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_WEBGPU

#include "decoders/processor/operators/gpu/webgpu_debayer_rcd.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
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

enum class BindingKind {
  ReadTextureR32F,
  WriteTextureR32F,
  WriteTextureRGBA32F,
  UniformBuffer,
};

enum class Kernel : uint32_t {
  InitAndVH,
  GreenAtRB,
  PQDir,
  RBAtRB,
  RBAtG,
  MergeRGBA,
};

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

auto BindingKindsFor(Kernel kernel) -> std::vector<BindingKind> {
  switch (kernel) {
    case Kernel::InitAndVH:
      return {BindingKind::ReadTextureR32F,  BindingKind::WriteTextureR32F,
              BindingKind::WriteTextureR32F, BindingKind::WriteTextureR32F,
              BindingKind::WriteTextureR32F, BindingKind::UniformBuffer};
    case Kernel::GreenAtRB:
      return {BindingKind::ReadTextureR32F, BindingKind::ReadTextureR32F,
              BindingKind::ReadTextureR32F, BindingKind::WriteTextureR32F,
              BindingKind::UniformBuffer};
    case Kernel::PQDir:
      return {BindingKind::ReadTextureR32F, BindingKind::WriteTextureR32F,
              BindingKind::UniformBuffer};
    case Kernel::RBAtRB:
    case Kernel::RBAtG:
      return {BindingKind::ReadTextureR32F,  BindingKind::ReadTextureR32F,
              BindingKind::ReadTextureR32F,  BindingKind::ReadTextureR32F,
              BindingKind::WriteTextureR32F, BindingKind::WriteTextureR32F,
              BindingKind::UniformBuffer};
    case Kernel::MergeRGBA:
      return {BindingKind::ReadTextureR32F, BindingKind::ReadTextureR32F,
              BindingKind::ReadTextureR32F, BindingKind::WriteTextureRGBA32F,
              BindingKind::UniformBuffer};
  }
  return {};
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

auto MakeBuffer(uint64_t size, wgpu::BufferUsage usage) -> wgpu::Buffer {
  wgpu::BufferDescriptor descriptor{};
  descriptor.usage = usage;
  descriptor.size  = size;
  auto buffer      = WebGpuContext::Instance().Device().CreateBuffer(&descriptor);
  if (!buffer.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create buffer.");
  }
  return buffer;
}

void SubmitAndWait(const wgpu::CommandBuffer& command_buffer) {
  WebGpuContext::Instance().Queue().Submit(1, &command_buffer);
  WebGpuContext::Instance().WaitForSubmittedWork();
}

auto MakeTextureView(const WebGpuImage& image) -> wgpu::TextureView {
  wgpu::TextureViewDescriptor descriptor{};
  descriptor.dimension = wgpu::TextureViewDimension::e2D;
  auto view            = image.Texture().CreateView(&descriptor);
  if (!view.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create texture view.");
  }
  return view;
}

void ConfigureLayoutEntry(wgpu::BindGroupLayoutEntry& entry, BindingKind kind) {
  entry.visibility = wgpu::ShaderStage::Compute;
  switch (kind) {
    case BindingKind::ReadTextureR32F:
      entry.texture.sampleType    = wgpu::TextureSampleType::UnfilterableFloat;
      entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
      break;
    case BindingKind::WriteTextureR32F:
      entry.storageTexture.access        = wgpu::StorageTextureAccess::WriteOnly;
      entry.storageTexture.format        = wgpu::TextureFormat::R32Float;
      entry.storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;
      break;
    case BindingKind::WriteTextureRGBA32F:
      entry.storageTexture.access        = wgpu::StorageTextureAccess::WriteOnly;
      entry.storageTexture.format        = wgpu::TextureFormat::RGBA32Float;
      entry.storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;
      break;
    case BindingKind::UniformBuffer:
      entry.buffer.type = wgpu::BufferBindingType::Uniform;
      break;
  }
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

  const auto                              kinds = BindingKindsFor(kernel);
  std::vector<wgpu::BindGroupLayoutEntry> entries;
  entries.reserve(kinds.size());
  for (uint32_t i = 0; i < kinds.size(); ++i) {
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = i;
    ConfigureLayoutEntry(entry, kinds[i]);
    entries.push_back(entry);
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

struct BindingResource {
  const WebGpuImage* texture = nullptr;
  wgpu::Buffer       buffer  = nullptr;
};

auto CreateBindGroup(const wgpu::ComputePipeline&        pipeline,
                     const std::vector<BindingResource>& resources) -> wgpu::BindGroup {
  std::vector<wgpu::TextureView> views;
  views.reserve(resources.size());
  std::vector<wgpu::BindGroupEntry> entries;
  entries.reserve(resources.size());

  for (uint32_t i = 0; i < resources.size(); ++i) {
    wgpu::BindGroupEntry entry{};
    entry.binding = i;
    if (resources[i].texture != nullptr) {
      views.push_back(MakeTextureView(*resources[i].texture));
      entry.textureView = views.back();
    } else {
      entry.buffer = resources[i].buffer;
    }
    entries.push_back(entry);
  }

  wgpu::BindGroupDescriptor desc{};
  desc.layout     = pipeline.GetBindGroupLayout(0);
  desc.entryCount = entries.size();
  desc.entries    = entries.data();
  auto bind_group = WebGpuContext::Instance().Device().CreateBindGroup(&desc);
  if (!bind_group.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create bind group.");
  }
  return bind_group;
}

void Dispatch(wgpu::ComputePassEncoder& compute, Kernel kernel,
              const std::vector<BindingResource>& resources, uint32_t width, uint32_t height) {
  auto pipeline   = GetOrCreatePipeline(kernel);
  auto bind_group = CreateBindGroup(pipeline, resources);
  compute.SetPipeline(pipeline);
  compute.SetBindGroup(0, bind_group);
  compute.DispatchWorkgroups((width + 7) / 8, (height + 7) / 8, 1);
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

  WebGpuImage r0;
  WebGpuImage g0;
  WebGpuImage g1;
  WebGpuImage b0;
  WebGpuImage vh;
  WebGpuImage pq;
  WebGpuImage r1;
  WebGpuImage b1;
  WebGpuImage r2;
  WebGpuImage b2;
  WebGpuImage output;
  r0.Create(width, height, PixelFormat::R32FLOAT);
  g0.Create(width, height, PixelFormat::R32FLOAT);
  g1.Create(width, height, PixelFormat::R32FLOAT);
  b0.Create(width, height, PixelFormat::R32FLOAT);
  vh.Create(width, height, PixelFormat::R32FLOAT);
  pq.Create(width, height, PixelFormat::R32FLOAT);
  r1.Create(width, height, PixelFormat::R32FLOAT);
  b1.Create(width, height, PixelFormat::R32FLOAT);
  r2.Create(width, height, PixelFormat::R32FLOAT);
  b2.Create(width, height, PixelFormat::R32FLOAT);
  output.Create(width, height, PixelFormat::RGBA32FLOAT);

  const SinglePlaneParams plane_params{
      .rgb_fc = {static_cast<uint32_t>(pattern.rgb_fc[0]), static_cast<uint32_t>(pattern.rgb_fc[1]),
                 static_cast<uint32_t>(pattern.rgb_fc[2]),
                 static_cast<uint32_t>(pattern.rgb_fc[3])},
      .width  = width,
      .height = height,
      .stride = width,
      .padding = 0,
  };
  const MergeParams merge_params{
      .width        = width,
      .height       = height,
      .plane_stride = width,
      .rgba_stride  = width,
  };

  auto plane_params_buffer = MakeBuffer(sizeof(SinglePlaneParams),
                                        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  WebGpuContext::Instance().Queue().WriteBuffer(plane_params_buffer, 0, &plane_params,
                                                sizeof(plane_params));

  auto merge_params_buffer =
      MakeBuffer(sizeof(MergeParams), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  WebGpuContext::Instance().Queue().WriteBuffer(merge_params_buffer, 0, &merge_params,
                                                sizeof(merge_params));

  auto encoder = WebGpuContext::Instance().Device().CreateCommandEncoder();
  {
    auto compute = encoder.BeginComputePass();

    Dispatch(compute, Kernel::InitAndVH,
             {{&image}, {&r0}, {&g0}, {&b0}, {&vh}, {nullptr, plane_params_buffer}}, width, height);
    Dispatch(compute, Kernel::GreenAtRB,
             {{&image}, {&vh}, {&g0}, {&g1}, {nullptr, plane_params_buffer}}, width, height);
    Dispatch(compute, Kernel::PQDir, {{&image}, {&pq}, {nullptr, plane_params_buffer}}, width,
             height);
    Dispatch(compute, Kernel::RBAtRB,
             {{&pq}, {&g1}, {&r0}, {&b0}, {&r1}, {&b1}, {nullptr, plane_params_buffer}}, width,
             height);
    Dispatch(compute, Kernel::RBAtG,
             {{&vh}, {&g1}, {&r1}, {&b1}, {&r2}, {&b2}, {nullptr, plane_params_buffer}}, width,
             height);
    Dispatch(compute, Kernel::MergeRGBA,
             {{&r2}, {&g1}, {&b2}, {&output}, {nullptr, merge_params_buffer}}, width, height);

    compute.End();
  }
  SubmitAndWait(encoder.Finish());

  image = std::move(output);
}

}  // namespace webgpu
}  // namespace alcedo

#endif
