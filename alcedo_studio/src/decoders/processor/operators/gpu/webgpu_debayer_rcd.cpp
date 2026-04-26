//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_WEBGPU

#include "decoders/processor/operators/gpu/webgpu_debayer_rcd.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
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

using Clock = std::chrono::steady_clock;

struct SinglePlaneParams {
  std::array<uint32_t, 4> rgb_fc;
  uint32_t                width;
  uint32_t                height;
  uint32_t                stride;
  uint32_t                padding;
};

auto MsSince(Clock::time_point start) -> double {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

enum class BindingKind {
  ReadTextureR32F,
  WriteTextureR32F,
  WriteTextureRGBA16F,
  WriteTextureRGBA32F,
  UniformBuffer,
};

enum class Kernel : uint32_t {
  InitAndVH,
  GreenAtRB,
  FinalRGBA,
};

auto KernelNameFor(Kernel kernel) -> const char* {
  switch (kernel) {
    case Kernel::InitAndVH:
      return "rcd_init_and_vh";
    case Kernel::GreenAtRB:
      return "rcd_green_at_rb";
    case Kernel::FinalRGBA:
      return "rcd_final_rgba";
  }
  throw std::runtime_error("WebGPU Debayer RCD: unknown kernel.");
}

auto BindingKindsFor(Kernel kernel) -> std::vector<BindingKind> {
  switch (kernel) {
    case Kernel::InitAndVH:
      return {BindingKind::ReadTextureR32F, BindingKind::WriteTextureR32F,
              BindingKind::WriteTextureRGBA16F, BindingKind::UniformBuffer};
    case Kernel::GreenAtRB:
      return {BindingKind::ReadTextureR32F, BindingKind::ReadTextureR32F,
              BindingKind::ReadTextureR32F, BindingKind::WriteTextureR32F,
              BindingKind::UniformBuffer};
    case Kernel::FinalRGBA:
      return {BindingKind::ReadTextureR32F,  BindingKind::ReadTextureR32F,
              BindingKind::ReadTextureR32F,
              BindingKind::WriteTextureRGBA32F,
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
  const auto submit_start = Clock::now();
  WebGpuContext::Instance().Queue().Submit(1, &command_buffer);
  const auto submit_ms = MsSince(submit_start);
  const auto wait_start = Clock::now();
  WebGpuContext::Instance().WaitForSubmittedWork();
  const auto wait_ms = MsSince(wait_start);
  std::cout << "[WebGPU RCD timing] queue_submit=" << submit_ms << " ms"
            << " wait_submitted_work=" << wait_ms << " ms"
            << " submit_wait_total=" << (submit_ms + wait_ms) << " ms\n";
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
    case BindingKind::WriteTextureRGBA16F:
      entry.storageTexture.access        = wgpu::StorageTextureAccess::WriteOnly;
      entry.storageTexture.format        = wgpu::TextureFormat::RGBA16Float;
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

auto GetOrCreatePipeline(Kernel kernel, bool* cache_hit, double* lookup_ms)
    -> wgpu::ComputePipeline {
  const auto start = Clock::now();
  static std::unordered_map<std::string, wgpu::ComputePipeline> cache;
  const std::string                                             key = KernelNameFor(kernel);
  auto                                                          it  = cache.find(key);
  if (it != cache.end()) {
    if (cache_hit != nullptr) {
      *cache_hit = true;
    }
    if (lookup_ms != nullptr) {
      *lookup_ms = MsSince(start);
    }
    return it->second;
  }
  if (cache_hit != nullptr) {
    *cache_hit = false;
  }

#ifndef ALCEDO_WEBGPU_DEBAYER_RCD_WGSL_PATH
#error "ALCEDO_WEBGPU_DEBAYER_RCD_WGSL_PATH must be defined when WebGPU Debayer RCD is enabled."
#endif

  auto&      device = WebGpuContext::Instance().Device();
  auto       stage_start = Clock::now();
  const auto wgsl_source =
      ReadTextFile(ALCEDO_WEBGPU_DEBAYER_RCD_WGSL_PATH, "debayer_rcd WGSL shader");
  const auto read_wgsl_ms = MsSince(stage_start);

  wgpu::ShaderSourceWGSL       wgsl_desc{};
  wgpu::ShaderModuleDescriptor shader_desc{};
  wgsl_desc.code          = std::string_view(wgsl_source.data(), wgsl_source.size());
  shader_desc.nextInChain = &wgsl_desc;
  stage_start             = Clock::now();
  auto shader_module      = device.CreateShaderModule(&shader_desc);
  const auto shader_module_ms = MsSince(stage_start);
  if (!shader_module.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create shader module.");
  }

  stage_start                              = Clock::now();
  const auto                              kinds = BindingKindsFor(kernel);
  std::vector<wgpu::BindGroupLayoutEntry> entries;
  entries.reserve(kinds.size());
  for (uint32_t i = 0; i < kinds.size(); ++i) {
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = i;
    ConfigureLayoutEntry(entry, kinds[i]);
    entries.push_back(entry);
  }
  const auto layout_entries_ms = MsSince(stage_start);

  wgpu::BindGroupLayoutDescriptor bgl_desc{};
  bgl_desc.entryCount    = entries.size();
  bgl_desc.entries       = entries.data();
  stage_start            = Clock::now();
  auto bind_group_layout = device.CreateBindGroupLayout(&bgl_desc);
  const auto bind_group_layout_ms = MsSince(stage_start);
  if (!bind_group_layout.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create bind group layout.");
  }

  wgpu::PipelineLayoutDescriptor pl_desc{};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts     = &bind_group_layout;
  stage_start                  = Clock::now();
  auto pipeline_layout         = device.CreatePipelineLayout(&pl_desc);
  const auto pipeline_layout_ms = MsSince(stage_start);
  if (!pipeline_layout.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create pipeline layout.");
  }

  wgpu::ComputePipelineDescriptor cp_desc{};
  cp_desc.layout             = pipeline_layout;
  cp_desc.compute.module     = shader_module;
  cp_desc.compute.entryPoint = key.c_str();
  stage_start                = Clock::now();
  auto pipeline              = device.CreateComputePipeline(&cp_desc);
  const auto compute_pipeline_ms = MsSince(stage_start);
  if (!pipeline.Get()) {
    throw std::runtime_error("WebGPU Debayer RCD: Failed to create compute pipeline.");
  }

  cache[key] = pipeline;
  const auto total_ms = MsSince(start);
  if (lookup_ms != nullptr) {
    *lookup_ms = total_ms;
  }
  std::cout << "[WebGPU RCD timing] pipeline kernel=" << key << " cache=miss"
            << " read_wgsl=" << read_wgsl_ms << " ms"
            << " shader_module=" << shader_module_ms << " ms"
            << " layout_entries=" << layout_entries_ms << " ms"
            << " bind_group_layout=" << bind_group_layout_ms << " ms"
            << " pipeline_layout=" << pipeline_layout_ms << " ms"
            << " compute_pipeline=" << compute_pipeline_ms << " ms"
            << " total=" << total_ms << " ms\n";
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
  constexpr uint32_t kWorkgroupWidth  = 32;
  constexpr uint32_t kWorkgroupHeight = 8;
  bool              pipeline_cache_hit = false;
  double            pipeline_lookup_ms = 0.0;
  auto pipeline = GetOrCreatePipeline(kernel, &pipeline_cache_hit, &pipeline_lookup_ms);
  auto stage_start = Clock::now();
  auto bind_group = CreateBindGroup(pipeline, resources);
  const auto bind_group_ms = MsSince(stage_start);
  stage_start              = Clock::now();
  compute.SetPipeline(pipeline);
  compute.SetBindGroup(0, bind_group);
  const auto groups_x = (width + kWorkgroupWidth - 1) / kWorkgroupWidth;
  const auto groups_y = (height + kWorkgroupHeight - 1) / kWorkgroupHeight;
  compute.DispatchWorkgroups(groups_x, groups_y, 1);
  const auto encode_ms = MsSince(stage_start);
  std::cout << "[WebGPU RCD timing] dispatch kernel=" << KernelNameFor(kernel)
            << " pipeline_cache=" << (pipeline_cache_hit ? "hit" : "miss")
            << " pipeline_lookup=" << pipeline_lookup_ms << " ms"
            << " bind_group=" << bind_group_ms << " ms"
            << " encode=" << encode_ms << " ms"
            << " workgroups=" << groups_x << 'x' << groups_y << "\n";
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

  WebGpuImage g0;
  WebGpuImage g1;
  WebGpuImage dir;
  WebGpuImage output;
  auto        stage_start = Clock::now();
  g0.Create(width, height, PixelFormat::R32FLOAT);
  g1.Create(width, height, PixelFormat::R32FLOAT);
  dir.Create(width, height, PixelFormat::RGBA16FLOAT);
  output.Create(width, height, PixelFormat::RGBA32FLOAT);
  const auto texture_create_ms = MsSince(stage_start);

  const SinglePlaneParams plane_params{
      .rgb_fc = {static_cast<uint32_t>(pattern.rgb_fc[0]), static_cast<uint32_t>(pattern.rgb_fc[1]),
                 static_cast<uint32_t>(pattern.rgb_fc[2]),
                 static_cast<uint32_t>(pattern.rgb_fc[3])},
      .width  = width,
      .height = height,
      .stride = width,
      .padding = 0,
  };
  stage_start = Clock::now();
  auto plane_params_buffer = MakeBuffer(sizeof(SinglePlaneParams),
                                        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  const auto uniform_buffer_ms = MsSince(stage_start);
  stage_start                  = Clock::now();
  WebGpuContext::Instance().Queue().WriteBuffer(plane_params_buffer, 0, &plane_params,
                                                sizeof(plane_params));
  const auto uniform_write_ms = MsSince(stage_start);

  stage_start  = Clock::now();
  auto encoder = WebGpuContext::Instance().Device().CreateCommandEncoder();
  const auto encoder_create_ms = MsSince(stage_start);
  {
    stage_start  = Clock::now();
    auto compute = encoder.BeginComputePass();
    const auto begin_compute_ms = MsSince(stage_start);
    std::cout << "[WebGPU RCD timing] setup"
              << " texture_create=" << texture_create_ms << " ms"
              << " uniform_buffer=" << uniform_buffer_ms << " ms"
              << " uniform_write=" << uniform_write_ms << " ms"
              << " encoder_create=" << encoder_create_ms << " ms"
              << " begin_compute=" << begin_compute_ms << " ms\n";

    Dispatch(compute, Kernel::InitAndVH,
             {{&image}, {&g0}, {&dir}, {nullptr, plane_params_buffer}}, width, height);
    Dispatch(compute, Kernel::GreenAtRB,
             {{&image}, {&dir}, {&g0}, {&g1}, {nullptr, plane_params_buffer}}, width, height);
    Dispatch(compute, Kernel::FinalRGBA,
             {{&dir}, {&g1}, {&image}, {&output}, {nullptr, plane_params_buffer}}, width,
             height);

    stage_start = Clock::now();
    compute.End();
    std::cout << "[WebGPU RCD timing] compute_end=" << MsSince(stage_start) << " ms\n";
  }
  stage_start = Clock::now();
  auto command_buffer = encoder.Finish();
  std::cout << "[WebGPU RCD timing] encoder_finish=" << MsSince(stage_start) << " ms\n";
  SubmitAndWait(command_buffer);

  image = std::move(output);
}

}  // namespace webgpu
}  // namespace alcedo

#endif
