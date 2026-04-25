//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_WEBGPU

#include "decoders/processor/operators/gpu/webgpu_cvt_ref_space.hpp"

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

struct ImageParams {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t channels;
};

struct OrientParams {
  uint32_t src_width;
  uint32_t src_height;
  uint32_t dst_width;
  uint32_t dst_height;
  uint32_t src_stride;
  uint32_t dst_stride;
  uint32_t flip;
  uint32_t padding;
  float    gain[4];
};

constexpr float    kMinGain           = 1e-6f;
constexpr uint32_t kRowAlignmentBytes = 256;

auto               AlignRowBytes(size_t row_bytes) -> size_t {
  return ((row_bytes + kRowAlignmentBytes - 1) / kRowAlignmentBytes) * kRowAlignmentBytes;
}

auto ChannelCount(PixelFormat format) -> uint32_t {
  switch (format) {
    case PixelFormat::R32FLOAT:
      return 1;
    case PixelFormat::RGBA32FLOAT:
      return 4;
    default:
      throw std::runtime_error("WebGPU CvtRefSpace: expected R32FLOAT or RGBA32FLOAT image.");
  }
}

auto ReadTextFile(const std::filesystem::path& path, const char* label) -> std::string {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error(std::string("WebGPU CvtRefSpace: Failed to open ") + label + ": " +
                             path.string());
  }

  const auto size = file.tellg();
  if (size < 0) {
    throw std::runtime_error(std::string("WebGPU CvtRefSpace: Failed to stat ") + label + ": " +
                             path.string());
  }

  std::string contents(static_cast<size_t>(size), '\0');
  file.seekg(0, std::ios::beg);
  if (!contents.empty() &&
      !file.read(contents.data(), static_cast<std::streamsize>(contents.size()))) {
    throw std::runtime_error(std::string("WebGPU CvtRefSpace: Failed to read ") + label + ": " +
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
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create buffer.");
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

auto MakeTextureView(const WebGpuImage& image) -> wgpu::TextureView {
  wgpu::TextureViewDescriptor descriptor{};
  descriptor.dimension = wgpu::TextureViewDimension::e2D;
  auto view            = image.Texture().CreateView(&descriptor);
  if (!view.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create texture view.");
  }
  return view;
}

void SubmitAndWait(const wgpu::CommandBuffer& command_buffer) {
  WebGpuContext::Instance().Queue().Submit(1, &command_buffer);
  WebGpuContext::Instance().WaitForSubmittedWork();
}

auto GetOrCreatePipeline(const char* entry_point, const std::vector<wgpu::BufferBindingType>& types)
    -> wgpu::ComputePipeline {
  static std::unordered_map<std::string, wgpu::ComputePipeline> cache;
  auto                                                          it = cache.find(entry_point);
  if (it != cache.end()) {
    return it->second;
  }

#ifndef ALCEDO_WEBGPU_CVT_REF_SPACE_WGSL_PATH
#error "ALCEDO_WEBGPU_CVT_REF_SPACE_WGSL_PATH must be defined when WebGPU CvtRefSpace is enabled."
#endif

  auto&      device = WebGpuContext::Instance().Device();
  const auto wgsl_source =
      ReadTextFile(ALCEDO_WEBGPU_CVT_REF_SPACE_WGSL_PATH, "cvt_ref_space WGSL shader");

  wgpu::ShaderSourceWGSL       wgsl_desc{};
  wgpu::ShaderModuleDescriptor shader_desc{};
  wgsl_desc.code          = std::string_view(wgsl_source.data(), wgsl_source.size());
  shader_desc.nextInChain = &wgsl_desc;
  auto shader_module      = device.CreateShaderModule(&shader_desc);
  if (!shader_module.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create shader module.");
  }

  std::vector<wgpu::BindGroupLayoutEntry> entries;
  entries.reserve(types.size());
  for (uint32_t i = 0; i < types.size(); ++i) {
    wgpu::BindGroupLayoutEntry e{};
    e.binding     = i;
    e.visibility  = wgpu::ShaderStage::Compute;
    e.buffer.type = types[i];
    entries.push_back(e);
  }

  wgpu::BindGroupLayoutDescriptor bgl_desc{};
  bgl_desc.entryCount    = entries.size();
  bgl_desc.entries       = entries.data();
  auto bind_group_layout = device.CreateBindGroupLayout(&bgl_desc);
  if (!bind_group_layout.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create bind group layout.");
  }

  wgpu::PipelineLayoutDescriptor pl_desc{};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts     = &bind_group_layout;
  auto pipeline_layout         = device.CreatePipelineLayout(&pl_desc);
  if (!pipeline_layout.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create pipeline layout.");
  }

  wgpu::ComputePipelineDescriptor cp_desc{};
  cp_desc.layout             = pipeline_layout;
  cp_desc.compute.module     = shader_module;
  cp_desc.compute.entryPoint = entry_point;
  auto pipeline              = device.CreateComputePipeline(&cp_desc);
  if (!pipeline.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create compute pipeline.");
  }

  cache[entry_point] = pipeline;
  return pipeline;
}

auto CreateBindGroup(const wgpu::ComputePipeline&     pipeline,
                     const std::vector<wgpu::Buffer>& buffers) -> wgpu::BindGroup {
  std::vector<wgpu::BindGroupEntry> entries;
  entries.reserve(buffers.size());
  for (uint32_t i = 0; i < buffers.size(); ++i) {
    wgpu::BindGroupEntry e{};
    e.binding = i;
    e.buffer  = buffers[i];
    entries.push_back(e);
  }

  wgpu::BindGroupDescriptor desc{};
  desc.layout     = pipeline.GetBindGroupLayout(0);
  desc.entryCount = entries.size();
  desc.entries    = entries.data();
  return WebGpuContext::Instance().Device().CreateBindGroup(&desc);
}

auto GetOrCreateOrientTexturePipeline() -> wgpu::ComputePipeline {
  static wgpu::ComputePipeline pipeline = nullptr;
  if (pipeline.Get()) {
    return pipeline;
  }

#ifndef ALCEDO_WEBGPU_CVT_REF_SPACE_WGSL_PATH
#error "ALCEDO_WEBGPU_CVT_REF_SPACE_WGSL_PATH must be defined when WebGPU CvtRefSpace is enabled."
#endif

  auto&      device = WebGpuContext::Instance().Device();
  const auto wgsl_source =
      ReadTextFile(ALCEDO_WEBGPU_CVT_REF_SPACE_WGSL_PATH, "cvt_ref_space WGSL shader");

  wgpu::ShaderSourceWGSL       wgsl_desc{};
  wgpu::ShaderModuleDescriptor shader_desc{};
  wgsl_desc.code          = std::string_view(wgsl_source.data(), wgsl_source.size());
  shader_desc.nextInChain = &wgsl_desc;
  auto shader_module      = device.CreateShaderModule(&shader_desc);
  if (!shader_module.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create shader module.");
  }

  std::array<wgpu::BindGroupLayoutEntry, 3> entries{};
  entries[0].binding                      = 0;
  entries[0].visibility                   = wgpu::ShaderStage::Compute;
  entries[0].texture.sampleType           = wgpu::TextureSampleType::UnfilterableFloat;
  entries[0].texture.viewDimension        = wgpu::TextureViewDimension::e2D;

  entries[1].binding                      = 1;
  entries[1].visibility                   = wgpu::ShaderStage::Compute;
  entries[1].storageTexture.access        = wgpu::StorageTextureAccess::WriteOnly;
  entries[1].storageTexture.format        = wgpu::TextureFormat::RGBA32Float;
  entries[1].storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;

  entries[2].binding                      = 2;
  entries[2].visibility                   = wgpu::ShaderStage::Compute;
  entries[2].buffer.type                  = wgpu::BufferBindingType::Uniform;

  wgpu::BindGroupLayoutDescriptor bgl_desc{};
  bgl_desc.entryCount    = entries.size();
  bgl_desc.entries       = entries.data();
  auto bind_group_layout = device.CreateBindGroupLayout(&bgl_desc);
  if (!bind_group_layout.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create bind group layout.");
  }

  wgpu::PipelineLayoutDescriptor pl_desc{};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts     = &bind_group_layout;
  auto pipeline_layout         = device.CreatePipelineLayout(&pl_desc);
  if (!pipeline_layout.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create pipeline layout.");
  }

  wgpu::ComputePipelineDescriptor cp_desc{};
  cp_desc.layout             = pipeline_layout;
  cp_desc.compute.module     = shader_module;
  cp_desc.compute.entryPoint = "apply_inverse_cam_mul_oriented_rgba";
  pipeline                   = device.CreateComputePipeline(&cp_desc);
  if (!pipeline.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create compute pipeline.");
  }
  return pipeline;
}

auto CreateOrientTextureBindGroup(const wgpu::ComputePipeline& pipeline, const WebGpuImage& src,
                                  const WebGpuImage& dst, const wgpu::Buffer& params_buffer)
    -> wgpu::BindGroup {
  auto                                src_view = MakeTextureView(src);
  auto                                dst_view = MakeTextureView(dst);

  std::array<wgpu::BindGroupEntry, 3> entries{};
  entries[0].binding     = 0;
  entries[0].textureView = src_view;
  entries[1].binding     = 1;
  entries[1].textureView = dst_view;
  entries[2].binding     = 2;
  entries[2].buffer      = params_buffer;

  wgpu::BindGroupDescriptor desc{};
  desc.layout     = pipeline.GetBindGroupLayout(0);
  desc.entryCount = entries.size();
  desc.entries    = entries.data();
  auto bind_group = WebGpuContext::Instance().Device().CreateBindGroup(&desc);
  if (!bind_group.Get()) {
    throw std::runtime_error("WebGPU CvtRefSpace: Failed to create texture bind group.");
  }
  return bind_group;
}

auto OrientedWidth(uint32_t width, uint32_t height, int flip) -> uint32_t {
  return (flip == 5 || flip == 6) ? height : width;
}

auto OrientedHeight(uint32_t width, uint32_t height, int flip) -> uint32_t {
  return (flip == 5 || flip == 6) ? width : height;
}

}  // namespace

void Clamp01(WebGpuImage& image) {
  if (image.Empty()) {
    return;
  }

  const uint32_t channels   = ChannelCount(image.Format());
  const uint32_t elem_bytes = static_cast<uint32_t>(sizeof(float) * channels);
  const uint32_t row_bytes =
      static_cast<uint32_t>(AlignRowBytes(static_cast<size_t>(image.Width()) * elem_bytes));
  const uint64_t buffer_size = static_cast<uint64_t>(row_bytes) * image.Height();
  const uint32_t stride      = row_bytes / elem_bytes;
  auto           image_buffer =
      MakeBuffer(buffer_size, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc |
                                  wgpu::BufferUsage::Storage);
  auto params_buffer =
      MakeBuffer(sizeof(ImageParams), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  const ImageParams params{
      .width    = image.Width(),
      .height   = image.Height(),
      .stride   = stride,
      .channels = channels,
  };
  WebGpuContext::Instance().Queue().WriteBuffer(params_buffer, 0, &params, sizeof(params));

  auto pipeline = GetOrCreatePipeline(
      "clamp01", {wgpu::BufferBindingType::Storage, wgpu::BufferBindingType::Uniform});
  auto bind_group = CreateBindGroup(pipeline, {image_buffer, params_buffer});

  auto encoder    = WebGpuContext::Instance().Device().CreateCommandEncoder();
  {
    auto src = MakeTextureCopy(image.Texture());
    auto dst = MakeBufferCopy(image_buffer, row_bytes, image.Height());
    auto ext = MakeExtent(image.Width(), image.Height());
    encoder.CopyTextureToBuffer(&src, &dst, &ext);
  }
  {
    auto compute = encoder.BeginComputePass();
    compute.SetPipeline(pipeline);
    compute.SetBindGroup(0, bind_group);
    compute.DispatchWorkgroups((image.Width() + 7) / 8, (image.Height() + 7) / 8, 1);
    compute.End();
  }
  {
    auto src = MakeBufferCopy(image_buffer, row_bytes, image.Height());
    auto dst = MakeTextureCopy(image.Texture());
    auto ext = MakeExtent(image.Width(), image.Height());
    encoder.CopyBufferToTexture(&src, &dst, &ext);
  }
  SubmitAndWait(encoder.Finish());
}

void ApplyInverseCamMulAndOrientRGBA(WebGpuImage& image, const float* cam_mul, int flip) {
  if (image.Empty()) {
    return;
  }
  if (image.Format() != PixelFormat::RGBA32FLOAT) {
    throw std::runtime_error("WebGPU CvtRefSpace: expected RGBA32FLOAT image.");
  }
  if (cam_mul == nullptr) {
    throw std::runtime_error("WebGPU CvtRefSpace: cam_mul is null.");
  }

  const uint32_t src_width  = image.Width();
  const uint32_t src_height = image.Height();
  const uint32_t dst_width  = OrientedWidth(src_width, src_height, flip);
  const uint32_t dst_height = OrientedHeight(src_width, src_height, flip);
  auto           params_buffer =
      MakeBuffer(sizeof(OrientParams), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);

  const float        g = std::max(cam_mul[1], kMinGain);
  const OrientParams params{
      .src_width  = src_width,
      .src_height = src_height,
      .dst_width  = dst_width,
      .dst_height = dst_height,
      .src_stride = src_width,
      .dst_stride = dst_width,
      .flip       = static_cast<uint32_t>(flip),
      .padding    = 0,
      .gain = {g / std::max(cam_mul[0], kMinGain), 1.0f, g / std::max(cam_mul[2], kMinGain), 1.0f},
  };
  WebGpuContext::Instance().Queue().WriteBuffer(params_buffer, 0, &params, sizeof(params));

  WebGpuImage output;
  output.Create(dst_width, dst_height, PixelFormat::RGBA32FLOAT);
  auto pipeline   = GetOrCreateOrientTexturePipeline();
  auto bind_group = CreateOrientTextureBindGroup(pipeline, image, output, params_buffer);

  auto encoder    = WebGpuContext::Instance().Device().CreateCommandEncoder();
  {
    auto compute = encoder.BeginComputePass();
    compute.SetPipeline(pipeline);
    compute.SetBindGroup(0, bind_group);
    compute.DispatchWorkgroups((src_width + 7) / 8, (src_height + 7) / 8, 1);
    compute.End();
  }
  SubmitAndWait(encoder.Finish());

  image = std::move(output);
}

}  // namespace webgpu
}  // namespace alcedo

#endif
