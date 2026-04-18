//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/edit_viewer/rhi_edit_viewer_surface.hpp"

#include "ui/edit_viewer/color_manager.hpp"

#include <QtGui/rhi/qrhi.h>
#include <QtGui/rhi/qrhi_platform.h>

#include <QFile>
#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QWindow>

#include <array>
#include <atomic>
#include <limits>
#include <mutex>

#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#endif

namespace alcedo {
namespace {

constexpr const char* kVertexShaderResource = ":/shaders/edit_viewer/rhi_image.vert.qsb";
constexpr const char* kFragmentShaderResource = ":/shaders/edit_viewer/rhi_image.frag.qsb";

#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
using Microsoft::WRL::ComPtr;

constexpr size_t kRgba32fPixelBytes = sizeof(float) * 4U;

auto IsValidSlotIndex(int slot_index, size_t slot_count) -> bool {
  return slot_index >= 0 && slot_index < static_cast<int>(slot_count);
}

auto DescribeDxgiAdapter(IDXGIAdapter* adapter) -> QString {
  if (!adapter) {
    return QStringLiteral("<null>");
  }

  DXGI_ADAPTER_DESC desc{};
  if (FAILED(adapter->GetDesc(&desc))) {
    return QStringLiteral("<unknown>");
  }
  return QString::fromWCharArray(desc.Description);
}

auto GetDxgiAdapterFromDevice(ID3D11Device* device) -> ComPtr<IDXGIAdapter> {
  if (!device) {
    return {};
  }

  ComPtr<IDXGIDevice> dxgi_device;
  if (FAILED(device->QueryInterface(IID_PPV_ARGS(dxgi_device.GetAddressOf()))) || !dxgi_device) {
    return {};
  }

  ComPtr<IDXGIAdapter> adapter;
  if (FAILED(dxgi_device->GetAdapter(adapter.GetAddressOf()))) {
    return {};
  }
  return adapter;
}

auto DescribeCudaDeviceList(const int* devices, unsigned int count) -> QString {
  QStringList entries;
  for (unsigned int i = 0; i < count; ++i) {
    entries.push_back(QString::number(devices[i]));
  }
  return entries.join(QStringLiteral(", "));
}

auto BindCudaDeviceOnCurrentThread(int cuda_device, const char* context) -> bool {
  if (cuda_device < 0) {
    return false;
  }

  const cudaError_t set_err = cudaSetDevice(cuda_device);
  if (set_err != cudaSuccess) {
    qWarning("RhiEditViewerSurface: cudaSetDevice(%d) failed in %s: %s", cuda_device, context,
             cudaGetErrorString(set_err));
    return false;
  }

  const cudaError_t init_err = cudaFree(nullptr);
  if (init_err != cudaSuccess) {
    qWarning("RhiEditViewerSurface: cudaFree(0) failed in %s for device %d: %s", context,
             cuda_device, cudaGetErrorString(init_err));
    return false;
  }

  const cudaError_t stale_err = cudaGetLastError();
  if (stale_err != cudaSuccess) {
    qWarning("RhiEditViewerSurface: cleared stale CUDA error before %s on device %d: %s", context,
             cuda_device, cudaGetErrorString(stale_err));
  }
  return true;
}

auto ResolveCudaDeviceForD3D11Device(ID3D11Device* device) -> int {
  if (!device) {
    return -1;
  }

  int current_cuda_device = -1;
  const cudaError_t current_device_err = cudaGetDevice(&current_cuda_device);
  if (current_device_err != cudaSuccess || current_cuda_device < 0) {
    qWarning("RhiEditViewerSurface: cudaGetDevice failed while validating D3D11 interop: %s",
             cudaGetErrorString(current_device_err));
    return -1;
  }

  unsigned int cuda_device_count = 0;
  std::array<int, 8> cuda_devices{};
  const cudaError_t query_err =
      cudaD3D11GetDevices(&cuda_device_count, cuda_devices.data(),
                          static_cast<unsigned int>(cuda_devices.size()), device,
                          cudaD3D11DeviceListAll);
  if (query_err != cudaSuccess || cuda_device_count == 0) {
    const auto adapter = GetDxgiAdapterFromDevice(device);
    qWarning("RhiEditViewerSurface: D3D11 device adapter '%s' is not CUDA-interoperable for "
             "current preview device %d (%s).",
             qPrintable(DescribeDxgiAdapter(adapter.Get())), current_cuda_device,
             cudaGetErrorString(query_err));
    return -1;
  }

  for (unsigned int i = 0; i < cuda_device_count; ++i) {
    if (cuda_devices[i] == current_cuda_device) {
      return current_cuda_device;
    }
  }

  const auto adapter = GetDxgiAdapterFromDevice(device);
  qWarning("RhiEditViewerSurface: D3D11 adapter '%s' maps to CUDA device(s) [%s], but the "
           "pipeline is running on CUDA device %d.",
           qPrintable(DescribeDxgiAdapter(adapter.Get())),
           qPrintable(DescribeCudaDeviceList(cuda_devices.data(), cuda_device_count)),
           current_cuda_device);
  return -1;
}
#endif

}  // namespace

struct RhiImageRenderer::UniformData {
  float scale_zoom[4] = {1.0f, 1.0f, 1.0f, 0.0f};
  float pan_mode[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct RhiImageRenderer::VertexData {
  float position[2];
  float uv[2];
};

struct RhiEditViewerSurface::PlatformState {
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  struct DirectPresentSlot {
    ComPtr<ID3D11Texture2D> texture;
    HANDLE                  shared_handle    = nullptr;
    cudaExternalMemory_t    external_memory  = nullptr;
    cudaMipmappedArray_t    mipmapped_array  = nullptr;
    cudaArray_t             image_array      = nullptr;
    int                     width            = 0;
    int                     height           = 0;
    std::uintptr_t          texture_handle   = 0;
  };

  std::array<DirectPresentSlot, 2> targets{};
  ID3D11Device*                    device = nullptr;
  int                              cuda_device = -1;
  mutable std::mutex               mutex{};
  std::atomic<int>                 pending_frame_idx{-1};
  std::atomic<FramePresentationMode> active_presentation_mode{
      FramePresentationMode::FullFrame};
  std::atomic<FramePresentationMode> pending_presentation_mode{
      FramePresentationMode::FullFrame};
  std::atomic<bool>                pending_presentation_mode_valid{false};
  int                              active_idx        = 0;
  int                              write_idx         = 1;
  int                              render_target_idx = 0;
  int                              mapped_slot_idx   = -1;
  int                              ready_slot_idx    = -1;
  bool                             supports_direct_present = true;
#else
  bool supports_direct_present = false;
#endif
};

#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
void ReleaseDirectPresentSlot(RhiEditViewerSurface::PlatformState::DirectPresentSlot& slot) {
  if (slot.mipmapped_array) {
    cudaFreeMipmappedArray(slot.mipmapped_array);
    slot.mipmapped_array = nullptr;
  }
  slot.image_array = nullptr;
  if (slot.external_memory) {
    cudaDestroyExternalMemory(slot.external_memory);
    slot.external_memory = nullptr;
  }
  if (slot.shared_handle) {
    CloseHandle(slot.shared_handle);
    slot.shared_handle = nullptr;
  }
  slot.texture.Reset();
  slot.width          = 0;
  slot.height         = 0;
  slot.texture_handle = 0;
}
#endif

RhiImageRenderer::~RhiImageRenderer() { releaseResources(); }

void RhiImageRenderer::initialize(QRhi* rhi, QRhiRenderTarget* render_target,
                                  QRhiCommandBuffer* command_buffer) {
  if (rhi_ != rhi) {
    releaseResources();
    rhi_ = rhi;
  }

  ensureStaticResources(render_target, command_buffer);
}

void RhiImageRenderer::releaseResources() {
  destroyResource(pipeline_);
  destroyResource(shader_resource_bindings_);
  destroyResource(vertex_buffer_);
  destroyResource(uniform_buffer_);
  destroyResource(sampler_);
  destroyResource(source_texture_);
  bound_render_target_          = nullptr;
  source_texture_width_         = 0;
  source_texture_height_        = 0;
  source_texture_native_object_ = 0;
  source_texture_is_imported_   = false;
  static_upload_pending_        = false;
  rhi_                          = nullptr;
}

void RhiImageRenderer::render(QRhiCommandBuffer* command_buffer, QRhiRenderTarget* render_target,
                              const ViewerViewState& view_state,
                              const ViewportWidgetInfo& widget_info, const ViewerFrame& frame,
                              const ImportedTextureFrame& imported_frame,
                              ViewerGpuFrameUpload* pending_upload) {
  if (!command_buffer || !render_target || !rhi_) {
    return;
  }

  ensureStaticResources(render_target, command_buffer);

  QRhiResourceUpdateBatch* resource_updates = rhi_->nextResourceUpdateBatch();

  if (pending_upload && *pending_upload && frame) {
    ensureSourceTexture(frame, command_buffer);

    const size_t upload_bytes =
        pending_upload->row_bytes * static_cast<size_t>(pending_upload->height);
    if (upload_bytes <= static_cast<size_t>((std::numeric_limits<int>::max)())) {
      QByteArray upload_data = QByteArray::fromRawData(
          static_cast<const char*>(pending_upload->pixels.get()), static_cast<int>(upload_bytes));
      QRhiTextureSubresourceUploadDescription upload_desc(upload_data);
      upload_desc.setDataStride(static_cast<quint32>(pending_upload->row_bytes));
      upload_desc.setSourceSize(QSize(pending_upload->width, pending_upload->height));
      resource_updates->uploadTexture(
          source_texture_,
          QRhiTextureUploadDescription(QRhiTextureUploadEntry(0, 0, upload_desc)));
    }
  }

  if (imported_frame) {
    ensureImportedSourceTexture(imported_frame, command_buffer);
  }

  UniformData uniform_data;
  const bool has_frame = frame || imported_frame;
  const int  frame_width = imported_frame ? imported_frame.width : frame.width;
  const int  frame_height = imported_frame ? imported_frame.height : frame.height;
  const auto presentation_mode =
      imported_frame ? imported_frame.presentation_mode : frame.presentation_mode;

  if (has_frame) {
    const auto scale = ViewportMapper::ComputeLetterboxScale(
        widget_info, ViewportImageInfo{frame_width, frame_height});
    float zoom  = view_state.snapshot.view_transform.zoom;
    float pan_x = view_state.snapshot.view_transform.pan.x();
    float pan_y = view_state.snapshot.view_transform.pan.y();
    if (presentation_mode == FramePresentationMode::RoiFrame) {
      zoom  = 1.0f;
      pan_x = 0.0f;
      pan_y = 0.0f;
    }

    uniform_data.scale_zoom[0] = scale.x;
    uniform_data.scale_zoom[1] = scale.y;
    uniform_data.scale_zoom[2] = zoom;
    uniform_data.pan_mode[0]   = pan_x;
    uniform_data.pan_mode[1]   = pan_y;
    uniform_data.pan_mode[2] =
        presentation_mode == FramePresentationMode::RoiFrame ? 1.0f : 0.0f;
  }
  resource_updates->updateDynamicBuffer(uniform_buffer_, 0, sizeof(UniformData), &uniform_data);

  command_buffer->beginPass(render_target, Qt::black, {1.0f, 0}, resource_updates);
  if (has_frame && pipeline_ && shader_resource_bindings_ && vertex_buffer_) {
    const QSize rt_size = render_target->pixelSize();
    const QRhiCommandBuffer::VertexInput vertex_input[] = {{vertex_buffer_, 0}};
    command_buffer->setGraphicsPipeline(pipeline_);
    command_buffer->setViewport(QRhiViewport(0, 0, rt_size.width(), rt_size.height()));
    command_buffer->setShaderResources(shader_resource_bindings_);
    command_buffer->setVertexInput(0, 1, vertex_input);
    command_buffer->draw(4);
  }
  command_buffer->endPass();
}

auto RhiImageRenderer::loadShader(const char* resource_path) const -> QShader {
  QFile shader_file(QString::fromUtf8(resource_path));
  if (!shader_file.open(QIODevice::ReadOnly)) {
    return {};
  }
  return QShader::fromSerialized(shader_file.readAll());
}

void RhiImageRenderer::destroyResource(QRhiTexture*& resource) {
  if (!resource) {
    return;
  }
  resource->destroy();
  delete resource;
  resource = nullptr;
}

void RhiImageRenderer::destroyResource(QRhiSampler*& resource) {
  if (!resource) {
    return;
  }
  resource->destroy();
  delete resource;
  resource = nullptr;
}

void RhiImageRenderer::destroyResource(QRhiBuffer*& resource) {
  if (!resource) {
    return;
  }
  resource->destroy();
  delete resource;
  resource = nullptr;
}

void RhiImageRenderer::destroyResource(QRhiShaderResourceBindings*& resource) {
  if (!resource) {
    return;
  }
  resource->destroy();
  delete resource;
  resource = nullptr;
}

void RhiImageRenderer::destroyResource(QRhiGraphicsPipeline*& resource) {
  if (!resource) {
    return;
  }
  resource->destroy();
  delete resource;
  resource = nullptr;
}

void RhiImageRenderer::ensureStaticResources(QRhiRenderTarget* render_target,
                                             QRhiCommandBuffer* command_buffer) {
  if (!rhi_ || !render_target) {
    return;
  }

  if (bound_render_target_ != render_target) {
    destroyResource(pipeline_);
    bound_render_target_ = render_target;
  }

  if (!source_texture_) {
    source_texture_ = rhi_->newTexture(QRhiTexture::RGBA32F, QSize(1, 1), 1);
    source_texture_->create();
    source_texture_width_         = 1;
    source_texture_height_        = 1;
    source_texture_native_object_ = 0;
    source_texture_is_imported_   = false;
    static_upload_pending_        = true;
  }

  if (!sampler_) {
    sampler_ = rhi_->newSampler(QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::None,
                                QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge);
    sampler_->create();
  }

  if (!uniform_buffer_) {
    uniform_buffer_ =
        rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(UniformData));
    uniform_buffer_->create();
  }

  if (!vertex_buffer_) {
    vertex_buffer_ =
        rhi_->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, 4 * sizeof(VertexData));
    vertex_buffer_->create();
    static_upload_pending_ = true;
  }

  if (!shader_resource_bindings_) {
    recreateShaderResources();
  }

  if (!pipeline_ && shader_resource_bindings_) {
    pipeline_ = rhi_->newGraphicsPipeline();
    QRhiVertexInputLayout input_layout;
    input_layout.setBindings({QRhiVertexInputBinding(sizeof(VertexData))});
    input_layout.setAttributes(
        {QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float2, 0),
         QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float2, 2 * sizeof(float))});
    pipeline_->setTopology(QRhiGraphicsPipeline::TriangleStrip);
    pipeline_->setCullMode(QRhiGraphicsPipeline::None);
    pipeline_->setSampleCount(render_target->sampleCount());
    pipeline_->setShaderStages(
        {QRhiShaderStage(QRhiShaderStage::Vertex, loadShader(kVertexShaderResource)),
         QRhiShaderStage(QRhiShaderStage::Fragment, loadShader(kFragmentShaderResource))});
    pipeline_->setVertexInputLayout(input_layout);
    pipeline_->setShaderResourceBindings(shader_resource_bindings_);
    pipeline_->setRenderPassDescriptor(render_target->renderPassDescriptor());
    pipeline_->create();
  }

  if (command_buffer && vertex_buffer_ && static_upload_pending_) {
    static constexpr std::array<VertexData, 4> kVertices = {
        VertexData{{-1.0f, -1.0f}, {0.0f, 1.0f}},
        VertexData{{1.0f, -1.0f}, {1.0f, 1.0f}},
        VertexData{{-1.0f, 1.0f}, {0.0f, 0.0f}},
        VertexData{{1.0f, 1.0f}, {1.0f, 0.0f}},
    };
    static constexpr std::array<float, 4> kBlackPixel = {0.0f, 0.0f, 0.0f, 1.0f};

    QRhiResourceUpdateBatch* updates = rhi_->nextResourceUpdateBatch();
    updates->uploadStaticBuffer(vertex_buffer_, kVertices.data());

    QByteArray black_upload(reinterpret_cast<const char*>(kBlackPixel.data()),
                            static_cast<int>(sizeof(kBlackPixel)));
    QRhiTextureSubresourceUploadDescription black_desc(black_upload);
    black_desc.setDataStride(static_cast<quint32>(sizeof(kBlackPixel)));
    black_desc.setSourceSize(QSize(1, 1));
    updates->uploadTexture(source_texture_,
                           QRhiTextureUploadDescription(QRhiTextureUploadEntry(0, 0, black_desc)));
    command_buffer->resourceUpdate(updates);
    static_upload_pending_ = false;
  }
}

void RhiImageRenderer::ensureSourceTexture(const ViewerFrame& frame,
                                           QRhiCommandBuffer* command_buffer) {
  if (!rhi_ || !frame) {
    return;
  }

  if (frame.width == source_texture_width_ && frame.height == source_texture_height_ &&
      source_texture_ && !source_texture_is_imported_) {
    return;
  }

  destroyResource(pipeline_);
  destroyResource(shader_resource_bindings_);
  destroyResource(source_texture_);

  source_texture_ = rhi_->newTexture(QRhiTexture::RGBA32F, QSize(frame.width, frame.height), 1);
  source_texture_->create();
  source_texture_width_         = frame.width;
  source_texture_height_        = frame.height;
  source_texture_native_object_ = 0;
  source_texture_is_imported_   = false;
  static_upload_pending_        = true;

  recreateShaderResources();
  ensureStaticResources(bound_render_target_, command_buffer);
}

void RhiImageRenderer::ensureImportedSourceTexture(const ImportedTextureFrame& frame,
                                                   QRhiCommandBuffer* command_buffer) {
  if (!rhi_ || !frame) {
    return;
  }

  const quint64 native_object = static_cast<quint64>(frame.texture_handle);
  if (source_texture_ && source_texture_is_imported_ &&
      source_texture_width_ == frame.width && source_texture_height_ == frame.height &&
      source_texture_native_object_ == native_object) {
    return;
  }

  destroyResource(pipeline_);
  destroyResource(shader_resource_bindings_);
  destroyResource(source_texture_);

  source_texture_ = rhi_->newTexture(QRhiTexture::RGBA32F, QSize(frame.width, frame.height), 1);
  if (!source_texture_->createFrom({native_object, 0})) {
    destroyResource(source_texture_);
    source_texture_width_         = 0;
    source_texture_height_        = 0;
    source_texture_native_object_ = 0;
    source_texture_is_imported_   = false;
    return;
  }

  source_texture_width_         = frame.width;
  source_texture_height_        = frame.height;
  source_texture_native_object_ = native_object;
  source_texture_is_imported_   = true;
  static_upload_pending_        = false;

  recreateShaderResources();
  ensureStaticResources(bound_render_target_, command_buffer);
}

void RhiImageRenderer::recreateShaderResources() {
  destroyResource(shader_resource_bindings_);
  if (!rhi_ || !uniform_buffer_ || !source_texture_ || !sampler_) {
    return;
  }

  shader_resource_bindings_ = rhi_->newShaderResourceBindings();
  shader_resource_bindings_->setBindings(
      {QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::VertexStage,
                                                uniform_buffer_),
       QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage,
                                                 source_texture_, sampler_)});
  shader_resource_bindings_->create();
}

RhiEditViewerSurface::RhiEditViewerSurface(QWidget* parent)
    : QRhiWidget(parent), platform_state_(std::make_unique<PlatformState>()) {
  setAutoFillBackground(false);
  setMouseTracking(false);
#if defined(Q_OS_WIN)
  setApi(QRhiWidget::Api::Direct3D11);
#elif defined(HAVE_METAL)
  setApi(QRhiWidget::Api::Metal);
#endif
  setColorBufferFormat(QRhiWidget::TextureFormat::RGBA32F);
}

RhiEditViewerSurface::~RhiEditViewerSurface() { releasePlatformTargets(); }

auto RhiEditViewerSurface::widget() -> QWidget* { return this; }

void RhiEditViewerSurface::submitFrame(const ViewerFrame& frame) {
  latest_frame_ = frame;
  resetImportedFrameState();
  if (frame) {
    pending_upload_ = std::make_unique<ViewerGpuFrameUpload>(
        ViewerGpuFrameUpload{frame.width, frame.height, frame.row_bytes, frame.pixels,
                             frame.display_config, frame.presentation_mode});
  } else {
    pending_upload_.reset();
  }
}

#ifdef HAVE_METAL
void RhiEditViewerSurface::submitMetalFrame(const ViewerMetalFrame& frame) {
  latest_frame_ = {};
  pending_upload_.reset();
  resetImportedFrameState();
  if (frame) {
    latest_imported_frame_.width             = frame.width;
    latest_imported_frame_.height            = frame.height;
    latest_imported_frame_.texture_handle    = frame.texture_handle;
    latest_imported_frame_.presentation_mode = frame.presentation_mode;
    latest_imported_owner_ = frame.owner;
  }
}
#endif

void RhiEditViewerSurface::setDisplayConfig(const ViewerDisplayConfig& config) {
  if (config == display_config_ && !display_config_dirty_) {
    return;
  }
  display_config_       = config;
  display_config_dirty_ = true;
  applyDisplayConfig();
}

void RhiEditViewerSurface::setViewState(const ViewerViewState& state) { view_state_ = state; }

void RhiEditViewerSurface::requestRedraw() { update(); }

auto RhiEditViewerSurface::supportsDirectCudaPresent() const -> bool {
  return platform_state_ && platform_state_->supports_direct_present;
}

auto RhiEditViewerSurface::prepareRenderTarget(int width, int height)
    -> EditViewerRenderTargetResizeDecision {
  EditViewerRenderTargetResizeDecision decision{};
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (!supportsDirectCudaPresent()) {
    return decision;
  }

  std::lock_guard<std::mutex> lock(platform_state_->mutex);
  const auto& target_slot = platform_state_->targets[platform_state_->render_target_idx];
  if (target_slot.width != width || target_slot.height != height || !target_slot.texture ||
      !target_slot.image_array) {
    platform_state_->render_target_idx = platform_state_->write_idx;
    const auto& write_slot = platform_state_->targets[platform_state_->render_target_idx];
    decision.need_resize = write_slot.width != width || write_slot.height != height ||
                           !write_slot.texture || !write_slot.image_array;
  }
  decision.slot_index = platform_state_->render_target_idx;
#else
  (void)width;
  (void)height;
#endif
  return decision;
}

void RhiEditViewerSurface::commitRenderTargetResize(int slot_index, int width, int height) {
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (!supportsDirectCudaPresent() ||
      !IsValidSlotIndex(slot_index, platform_state_->targets.size())) {
    return;
  }

  std::lock_guard<std::mutex> lock(platform_state_->mutex);
  platform_state_->targets[slot_index].width  = width;
  platform_state_->targets[slot_index].height = height;
  platform_state_->render_target_idx        = slot_index;
#else
  (void)slot_index;
  (void)width;
  (void)height;
#endif
}

auto RhiEditViewerSurface::mapResourceForWrite() -> FrameWriteMapping {
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (!supportsDirectCudaPresent()) {
    return {};
  }

  if (!BindCudaDeviceOnCurrentThread(platform_state_->cuda_device, "mapResourceForWrite")) {
    return {};
  }

  platform_state_->mutex.lock();
  auto& slot = platform_state_->targets[platform_state_->render_target_idx];
  if (!slot.image_array || slot.width <= 0 || slot.height <= 0) {
    qWarning("RhiEditViewerSurface: no valid D3D11/CUDA present target is available.");
    platform_state_->mutex.unlock();
    return {};
  }

  platform_state_->mapped_slot_idx = platform_state_->render_target_idx;
  FrameWriteMapping mapping{};
  mapping.image_array   = slot.image_array;
  mapping.row_bytes     = static_cast<size_t>(slot.width) * kRgba32fPixelBytes;
  mapping.pixel_format  = FramePixelFormat::RGBA32F;
  mapping.memory_domain = FrameMemoryDomain::CudaDevice;
  mapping.target_type   = FrameWriteTargetType::CudaArray;
  mapping.native_object = slot.texture_handle;
  return mapping;
#else
  return {};
#endif
}

void RhiEditViewerSurface::unmapResource() {
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (!supportsDirectCudaPresent()) {
    return;
  }

  if (platform_state_->mapped_slot_idx >= 0 &&
      IsValidSlotIndex(platform_state_->mapped_slot_idx, platform_state_->targets.size())) {
    platform_state_->ready_slot_idx = platform_state_->mapped_slot_idx;
  }
  platform_state_->mapped_slot_idx = -1;
  platform_state_->mutex.unlock();
#endif
}

void RhiEditViewerSurface::notifyFrameReady() {
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (!supportsDirectCudaPresent()) {
    return;
  }

  std::lock_guard<std::mutex> lock(platform_state_->mutex);
  if (platform_state_->ready_slot_idx >= 0) {
    platform_state_->pending_frame_idx.store(platform_state_->ready_slot_idx,
                                             std::memory_order_release);
    platform_state_->ready_slot_idx = -1;
  }
#endif
}

void RhiEditViewerSurface::setNextFramePresentationMode(FramePresentationMode mode) {
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (supportsDirectCudaPresent()) {
    platform_state_->pending_presentation_mode.store(mode, std::memory_order_release);
    platform_state_->pending_presentation_mode_valid.store(true, std::memory_order_release);
    return;
  }
#endif
  latest_frame_.presentation_mode = mode;
}

auto RhiEditViewerSurface::activeRenderTargetState() const -> EditViewerRenderTargetState {
  EditViewerRenderTargetState state{};
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (!supportsDirectCudaPresent()) {
    return state;
  }

  std::lock_guard<std::mutex> lock(platform_state_->mutex);
  const auto& slot = platform_state_->targets[platform_state_->active_idx];
  state.slot_index = platform_state_->active_idx;
  state.width      = slot.width;
  state.height     = slot.height;
  state.presentation_mode =
      platform_state_->active_presentation_mode.load(std::memory_order_acquire);
#endif
  return state;
}

auto RhiEditViewerSurface::hasRenderTarget(int slot_index, int width, int height) const -> bool {
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (!supportsDirectCudaPresent() ||
      !IsValidSlotIndex(slot_index, platform_state_->targets.size())) {
    return false;
  }

  std::lock_guard<std::mutex> lock(platform_state_->mutex);
  const auto& slot = platform_state_->targets[slot_index];
  return slot.texture && slot.image_array && slot.width == width && slot.height == height;
#else
  (void)slot_index;
  (void)width;
  (void)height;
  return false;
#endif
}

auto RhiEditViewerSurface::ensureRenderTarget(int slot_index, int width, int height) -> bool {
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (!supportsDirectCudaPresent() ||
      !IsValidSlotIndex(slot_index, platform_state_->targets.size()) ||
      !platform_state_->device || width <= 0 || height <= 0) {
    return false;
  }

  if (!BindCudaDeviceOnCurrentThread(platform_state_->cuda_device, "ensureRenderTarget")) {
    return false;
  }

  std::lock_guard<std::mutex> lock(platform_state_->mutex);
  auto& slot = platform_state_->targets[slot_index];
  if (slot.texture && slot.image_array && slot.width == width && slot.height == height) {
    return true;
  }

  ReleaseDirectPresentSlot(slot);

  D3D11_TEXTURE2D_DESC desc{};
  desc.Width              = static_cast<UINT>(width);
  desc.Height             = static_cast<UINT>(height);
  desc.MipLevels          = 1;
  desc.ArraySize          = 1;
  desc.Format             = DXGI_FORMAT_R32G32B32A32_FLOAT;
  desc.SampleDesc.Count   = 1;
  desc.Usage              = D3D11_USAGE_DEFAULT;
  desc.BindFlags          = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
  desc.CPUAccessFlags     = 0;
  desc.MiscFlags          = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

  if (FAILED(platform_state_->device->CreateTexture2D(&desc, nullptr,
                                                       slot.texture.GetAddressOf()))) {
    qWarning("RhiEditViewerSurface: failed to create D3D11 present target %dx%d.", width, height);
    ReleaseDirectPresentSlot(slot);
    return false;
  }

  ComPtr<IDXGIResource1> dxgi_resource;
  if (FAILED(slot.texture.As(&dxgi_resource)) || !dxgi_resource) {
    qWarning("RhiEditViewerSurface: failed to query IDXGIResource1 for D3D11 present target.");
    ReleaseDirectPresentSlot(slot);
    return false;
  }

  if (FAILED(dxgi_resource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &slot.shared_handle)) ||
      !slot.shared_handle) {
    qWarning("RhiEditViewerSurface: CreateSharedHandle failed for D3D11 present target.");
    ReleaseDirectPresentSlot(slot);
    return false;
  }

  cudaExternalMemoryHandleDesc handle_desc{};
  handle_desc.type                = cudaExternalMemoryHandleTypeD3D11Resource;
  handle_desc.handle.win32.handle = slot.shared_handle;
  handle_desc.size                = static_cast<unsigned long long>(width) *
                     static_cast<unsigned long long>(height) *
                     static_cast<unsigned long long>(kRgba32fPixelBytes);
  handle_desc.flags               = cudaExternalMemoryDedicated;
  const cudaError_t import_err =
      cudaImportExternalMemory(&slot.external_memory, &handle_desc);
  if (import_err != cudaSuccess) {
    qWarning("RhiEditViewerSurface: cudaImportExternalMemory failed: %s",
             cudaGetErrorString(import_err));
    ReleaseDirectPresentSlot(slot);
    return false;
  }

  cudaExternalMemoryMipmappedArrayDesc array_desc{};
  array_desc.offset     = 0;
  array_desc.formatDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  array_desc.extent     =
      cudaExtent{static_cast<size_t>(width), static_cast<size_t>(height), 0};
  array_desc.flags      = cudaArrayColorAttachment;
  array_desc.numLevels  = 1;
  const cudaError_t map_err = cudaExternalMemoryGetMappedMipmappedArray(
      &slot.mipmapped_array, slot.external_memory, &array_desc);
  if (map_err != cudaSuccess) {
    qWarning("RhiEditViewerSurface: cudaExternalMemoryGetMappedMipmappedArray failed: %s",
             cudaGetErrorString(map_err));
    ReleaseDirectPresentSlot(slot);
    return false;
  }

  const cudaError_t level_err =
      cudaGetMipmappedArrayLevel(&slot.image_array, slot.mipmapped_array, 0);
  if (level_err != cudaSuccess) {
    qWarning("RhiEditViewerSurface: cudaGetMipmappedArrayLevel failed: %s",
             cudaGetErrorString(level_err));
    ReleaseDirectPresentSlot(slot);
    return false;
  }

  slot.width          = width;
  slot.height         = height;
  slot.texture_handle = reinterpret_cast<std::uintptr_t>(slot.texture.Get());
  {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
      const size_t used_bytes = total_bytes - free_bytes;
      const size_t slot_bytes =
          static_cast<size_t>(width) * static_cast<size_t>(height) * kRgba32fPixelBytes;
      qInfo("[VRAM] D3D11 present slot[%d] allocated (%dx%d, %zu MB): free=%zu MB / total=%zu MB "
            "(used=%zu MB)",
            slot_index, width, height, slot_bytes >> 20, free_bytes >> 20, total_bytes >> 20,
            used_bytes >> 20);
    }
  }
  return true;
#else
  (void)slot_index;
  (void)width;
  (void)height;
  return false;
#endif
}

void RhiEditViewerSurface::initialize(QRhiCommandBuffer* command_buffer) {
  applyDisplayConfig();
  renderer_.initialize(rhi(), renderTarget(), command_buffer);

#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  platform_state_->supports_direct_present = false;
  platform_state_->cuda_device             = -1;
  const char* backend_name = "<none>";
  if (rhi()) {
    switch (rhi()->backend()) {
      case QRhi::D3D11:    backend_name = "D3D11"; break;
      case QRhi::D3D12:    backend_name = "D3D12"; break;
      case QRhi::Vulkan:   backend_name = "Vulkan"; break;
      case QRhi::OpenGLES2:backend_name = "OpenGLES2"; break;
      case QRhi::Metal:    backend_name = "Metal"; break;
      case QRhi::Null:     backend_name = "Null"; break;
      default:             backend_name = "<other>"; break;
    }
  }
  bool bind_ok = false;
  if (rhi() && rhi()->backend() == QRhi::D3D11) {
    const auto* native_handles =
        static_cast<const QRhiD3D11NativeHandles*>(rhi()->nativeHandles());
    platform_state_->device =
        native_handles ? static_cast<ID3D11Device*>(native_handles->dev) : nullptr;
    platform_state_->cuda_device =
        platform_state_->device ? ResolveCudaDeviceForD3D11Device(platform_state_->device) : -1;
    bind_ok = platform_state_->device && platform_state_->cuda_device >= 0 &&
              BindCudaDeviceOnCurrentThread(platform_state_->cuda_device, "initialize");
    platform_state_->supports_direct_present = bind_ok;
  }
  qInfo("[DirectPresent] RHI backend=%s, d3d11_device=%p, cuda_device=%d, bind_ok=%d, "
        "supports_direct_present=%d",
        backend_name, static_cast<void*>(platform_state_->device), platform_state_->cuda_device,
        bind_ok ? 1 : 0, platform_state_->supports_direct_present ? 1 : 0);
#else
  qInfo("[DirectPresent] CUDA/D3D11 interop unavailable at compile time (HAVE_CUDA/Q_OS_WIN not set).");
#endif

  if (latest_frame_) {
    pending_upload_ = std::make_unique<ViewerGpuFrameUpload>(
        ViewerGpuFrameUpload{latest_frame_.width, latest_frame_.height, latest_frame_.row_bytes,
                             latest_frame_.pixels, latest_frame_.display_config,
                             latest_frame_.presentation_mode});
  }
}

void RhiEditViewerSurface::render(QRhiCommandBuffer* command_buffer) {
  if (display_config_dirty_) {
    applyDisplayConfig();
  }

  bool has_pending_explicit_frame = pending_upload_ && latest_frame_;

  ImportedTextureFrame direct_present_frame{};
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (supportsDirectCudaPresent() && !has_pending_explicit_frame) {
    const int pending_slot = platform_state_->pending_frame_idx.exchange(-1, std::memory_order_acq_rel);
    std::lock_guard<std::mutex> lock(platform_state_->mutex);
    int slot_to_show = platform_state_->active_idx;
    if (IsValidSlotIndex(pending_slot, platform_state_->targets.size())) {
      slot_to_show                = pending_slot;
      platform_state_->active_idx = pending_slot;
      platform_state_->write_idx  = 1 - pending_slot;
      if (platform_state_->pending_presentation_mode_valid.exchange(false,
                                                                    std::memory_order_acq_rel)) {
        platform_state_->active_presentation_mode.store(
            platform_state_->pending_presentation_mode.load(std::memory_order_acquire),
            std::memory_order_release);
      }
    }

    const auto& slot = platform_state_->targets[slot_to_show];
    if (slot.texture && slot.width > 0 && slot.height > 0) {
      direct_present_frame.width = slot.width;
      direct_present_frame.height = slot.height;
      direct_present_frame.texture_handle = slot.texture_handle;
      direct_present_frame.presentation_mode =
          platform_state_->active_presentation_mode.load(std::memory_order_acquire);
    }
  }
#endif

  if (direct_present_frame) {
    renderer_.render(
        command_buffer, renderTarget(), view_state_,
        ViewportWidgetInfo{width(), height(), static_cast<float>(devicePixelRatioF())}, {},
        direct_present_frame, nullptr);
    return;
  }

  renderer_.render(
      command_buffer, renderTarget(), view_state_,
      ViewportWidgetInfo{width(), height(), static_cast<float>(devicePixelRatioF())}, latest_frame_,
      latest_imported_frame_, pending_upload_.get());
  pending_upload_.reset();
}

void RhiEditViewerSurface::releaseResources() {
  renderer_.releaseResources();
  releasePlatformTargets();
}

void RhiEditViewerSurface::applyDisplayConfig() {
  QWidget* host_window = window();
  if (!host_window) {
    return;
  }

  const auto native_handle = reinterpret_cast<void*>(host_window->effectiveWinId());
  if (!native_handle) {
    return;
  }

  if (ColorManager::ApplyWindowColorSpace(native_handle, display_config_)) {
    display_config_dirty_ = false;
  }
}

void RhiEditViewerSurface::releasePlatformTargets() {
#if defined(Q_OS_WIN) && defined(HAVE_CUDA)
  if (!platform_state_) {
    return;
  }

  if (platform_state_->cuda_device >= 0) {
    (void)BindCudaDeviceOnCurrentThread(platform_state_->cuda_device, "releasePlatformTargets");
  }

  std::lock_guard<std::mutex> lock(platform_state_->mutex);
  for (auto& slot : platform_state_->targets) {
    ReleaseDirectPresentSlot(slot);
  }
  platform_state_->device = nullptr;
  platform_state_->cuda_device = -1;
  platform_state_->mapped_slot_idx = -1;
  platform_state_->ready_slot_idx  = -1;
#endif
}

void RhiEditViewerSurface::resetImportedFrameState() {
  latest_imported_frame_ = {};
  latest_imported_owner_.reset();
}

}  // namespace alcedo
