//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_WEBGPU

#include "webgpu/webgpu_geometry_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "image/webgpu_image.hpp"
#include "webgpu/webgpu_context.hpp"

namespace alcedo {
namespace webgpu {
namespace utils {
namespace {

struct GeoParams {
  uint32_t src_width;
  uint32_t src_height;
  uint32_t dst_width;
  uint32_t dst_height;
  uint32_t src_stride;
  uint32_t dst_stride;
  uint32_t channels;
  uint32_t padding;
};

constexpr uint32_t kRowAlignmentBytes = 256;

auto               AlignRowBytes(size_t row_bytes) -> size_t {
  return ((row_bytes + kRowAlignmentBytes - 1) / kRowAlignmentBytes) * kRowAlignmentBytes;
}

auto ChannelsForFormat(PixelFormat format) -> uint32_t {
  switch (format) {
    case PixelFormat::R16UINT:
    case PixelFormat::R16FLOAT:
    case PixelFormat::R32FLOAT:
      return 1;
    case PixelFormat::RGBA16UINT:
    case PixelFormat::RGBA16FLOAT:
    case PixelFormat::RGBA32FLOAT:
      return 4;
  }
  throw std::runtime_error("WebGPU Geometry Utils: unsupported pixel format.");
}

auto ReadTextFile(const std::filesystem::path& path, const char* label) -> std::string {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error(std::string("WebGPU Geometry Utils: Failed to open ") + label + ": " +
                             path.string());
  }

  const auto size = file.tellg();
  if (size < 0) {
    throw std::runtime_error(std::string("WebGPU Geometry Utils: Failed to stat ") + label + ": " +
                             path.string());
  }

  std::string contents(static_cast<size_t>(size), '\0');
  file.seekg(0, std::ios::beg);
  if (!contents.empty() &&
      !file.read(contents.data(), static_cast<std::streamsize>(contents.size()))) {
    throw std::runtime_error(std::string("WebGPU Geometry Utils: Failed to read ") + label + ": " +
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
    throw std::runtime_error("WebGPU Geometry Utils: Failed to create buffer.");
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

auto GetOrCreatePipeline(const char* entry_point) -> wgpu::ComputePipeline {
  static std::unordered_map<std::string, wgpu::ComputePipeline> cache;
  auto                                                          it = cache.find(entry_point);
  if (it != cache.end()) {
    return it->second;
  }

#ifndef ALCEDO_WEBGPU_GEOMETRY_UTILS_WGSL_PATH
#error \
    "ALCEDO_WEBGPU_GEOMETRY_UTILS_WGSL_PATH must be defined when WebGPU geometry utils is enabled."
#endif

  auto&      device = WebGpuContext::Instance().Device();
  const auto wgsl_source =
      ReadTextFile(ALCEDO_WEBGPU_GEOMETRY_UTILS_WGSL_PATH, "geometry_utils WGSL shader");

  wgpu::ShaderSourceWGSL       wgsl_desc{};
  wgpu::ShaderModuleDescriptor shader_desc{};
  wgsl_desc.code          = std::string_view(wgsl_source.data(), wgsl_source.size());
  shader_desc.nextInChain = &wgsl_desc;
  auto shader_module      = device.CreateShaderModule(&shader_desc);
  if (!shader_module.Get()) {
    throw std::runtime_error("WebGPU Geometry Utils: Failed to create shader module.");
  }

  std::array<wgpu::BindGroupLayoutEntry, 3> entries{};
  entries[0].binding     = 0;
  entries[0].visibility  = wgpu::ShaderStage::Compute;
  entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

  entries[1].binding     = 1;
  entries[1].visibility  = wgpu::ShaderStage::Compute;
  entries[1].buffer.type = wgpu::BufferBindingType::Storage;

  entries[2].binding     = 2;
  entries[2].visibility  = wgpu::ShaderStage::Compute;
  entries[2].buffer.type = wgpu::BufferBindingType::Uniform;

  wgpu::BindGroupLayoutDescriptor bgl_desc{};
  bgl_desc.entryCount    = entries.size();
  bgl_desc.entries       = entries.data();
  auto bind_group_layout = device.CreateBindGroupLayout(&bgl_desc);
  if (!bind_group_layout.Get()) {
    throw std::runtime_error("WebGPU Geometry Utils: Failed to create bind group layout.");
  }

  wgpu::PipelineLayoutDescriptor pl_desc{};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts     = &bind_group_layout;
  auto pipeline_layout         = device.CreatePipelineLayout(&pl_desc);
  if (!pipeline_layout.Get()) {
    throw std::runtime_error("WebGPU Geometry Utils: Failed to create pipeline layout.");
  }

  wgpu::ComputePipelineDescriptor cp_desc{};
  cp_desc.layout             = pipeline_layout;
  cp_desc.compute.module     = shader_module;
  cp_desc.compute.entryPoint = entry_point;
  auto pipeline              = device.CreateComputePipeline(&cp_desc);
  if (!pipeline.Get()) {
    throw std::runtime_error("WebGPU Geometry Utils: Failed to create compute pipeline.");
  }

  cache[entry_point] = pipeline;
  return pipeline;
}

void DispatchGeometry(const WebGpuImage& src, WebGpuImage& dst, const char* entry_point) {
  if (src.Empty()) {
    throw std::runtime_error("WebGPU Geometry Utils: source image is empty.");
  }

  const auto channels  = ChannelsForFormat(src.Format());
  const auto elem_size = (channels == 1) ? sizeof(float) : sizeof(float) * 4;
  const auto src_row_bytes =
      static_cast<uint32_t>(AlignRowBytes(static_cast<size_t>(src.Width()) * elem_size));
  const auto dst_row_bytes =
      static_cast<uint32_t>(AlignRowBytes(static_cast<size_t>(dst.Width()) * elem_size));
  const auto src_buffer_size = static_cast<uint64_t>(src_row_bytes) * src.Height();
  const auto dst_buffer_size = static_cast<uint64_t>(dst_row_bytes) * dst.Height();

  auto       src_buffer =
      MakeBuffer(src_buffer_size, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage);
  auto dst_buffer =
      MakeBuffer(dst_buffer_size, wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage);

  GeoParams params  = {};
  params.src_width  = src.Width();
  params.src_height = src.Height();
  params.dst_width  = dst.Width();
  params.dst_height = dst.Height();
  params.src_stride = src_row_bytes / static_cast<uint32_t>(elem_size);
  params.dst_stride = dst_row_bytes / static_cast<uint32_t>(elem_size);
  params.channels   = channels;

  auto params_buffer =
      MakeBuffer(sizeof(GeoParams), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  WebGpuContext::Instance().Queue().WriteBuffer(params_buffer, 0, &params, sizeof(params));

  auto                                pipeline = GetOrCreatePipeline(entry_point);

  std::array<wgpu::BindGroupEntry, 3> bg_entries{};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer  = src_buffer;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer  = dst_buffer;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer  = params_buffer;

  wgpu::BindGroupDescriptor bg_desc{};
  bg_desc.layout     = pipeline.GetBindGroupLayout(0);
  bg_desc.entryCount = bg_entries.size();
  bg_desc.entries    = bg_entries.data();
  auto bind_group    = WebGpuContext::Instance().Device().CreateBindGroup(&bg_desc);

  auto encoder       = WebGpuContext::Instance().Device().CreateCommandEncoder();

  // src texture → src buffer
  {
    auto s   = MakeTextureCopy(src.Texture());
    auto d   = MakeBufferCopy(src_buffer, src_row_bytes, src.Height());
    auto ext = MakeExtent(src.Width(), src.Height());
    encoder.CopyTextureToBuffer(&s, &d, &ext);
  }

  // compute
  {
    auto compute = encoder.BeginComputePass();
    compute.SetPipeline(pipeline);
    compute.SetBindGroup(0, bind_group);
    compute.DispatchWorkgroups((dst.Width() + 7) / 8, (dst.Height() + 7) / 8, 1);
    compute.End();
  }

  // dst buffer → dst texture
  {
    auto s   = MakeBufferCopy(dst_buffer, dst_row_bytes, dst.Height());
    auto d   = MakeTextureCopy(dst.Texture());
    auto ext = MakeExtent(dst.Width(), dst.Height());
    encoder.CopyBufferToTexture(&s, &d, &ext);
  }

  SubmitAndWait(encoder.Finish());
}

}  // namespace

void Rotate180(WebGpuImage& image) {
  if (image.Empty()) {
    return;
  }
  WebGpuImage output;
  output.Create(image.Width(), image.Height(), image.Format());
  DispatchGeometry(image, output, "rotate_180");
  image = std::move(output);
}

void Rotate90CW(WebGpuImage& image) {
  if (image.Empty()) {
    return;
  }
  WebGpuImage output;
  output.Create(image.Height(), image.Width(), image.Format());
  DispatchGeometry(image, output, "rotate_90_cw");
  image = std::move(output);
}

void Rotate90CCW(WebGpuImage& image) {
  if (image.Empty()) {
    return;
  }
  WebGpuImage output;
  output.Create(image.Height(), image.Width(), image.Format());
  DispatchGeometry(image, output, "rotate_90_ccw");
  image = std::move(output);
}

}  // namespace utils
}  // namespace webgpu
}  // namespace alcedo

#endif
