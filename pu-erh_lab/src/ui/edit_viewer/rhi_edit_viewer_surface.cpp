//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/edit_viewer/rhi_edit_viewer_surface.hpp"

#include <QtGui/rhi/qshader.h>
#include <QtGui/rhi/qrhi.h>

#include <QFile>

#include <array>
#include <limits>

#include "ui/edit_viewer/viewport_mapper.hpp"

namespace puerhlab {
namespace {

constexpr const char* kVertexShaderResource = ":/shaders/edit_viewer/rhi_image.vert.qsb";
constexpr const char* kFragmentShaderResource = ":/shaders/edit_viewer/rhi_image.frag.qsb";

}  // namespace

struct RhiImageRenderer::UniformData {
  float scale_zoom[4] = {1.0f, 1.0f, 1.0f, 0.0f};
  float pan_mode[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct RhiImageRenderer::VertexData {
  float position[2];
  float uv[2];
};

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
  bound_render_target_   = nullptr;
  source_texture_width_  = 0;
  source_texture_height_ = 0;
  static_upload_pending_ = false;
  rhi_                   = nullptr;
}

void RhiImageRenderer::render(QRhiCommandBuffer* command_buffer, QRhiRenderTarget* render_target,
                              const ViewerViewState& view_state,
                              const ViewportWidgetInfo& widget_info, const ViewerFrame& frame,
                              ViewerGpuFrameUpload* pending_upload) {
  if (!command_buffer || !render_target || !rhi_) {
    return;
  }

  ensureStaticResources(render_target, command_buffer);

  QRhiResourceUpdateBatch* resource_updates = rhi_->nextResourceUpdateBatch();

  if (pending_upload && *pending_upload && frame) {
    ensureSourceTexture(frame, command_buffer);

    const size_t upload_bytes = pending_upload->row_bytes * static_cast<size_t>(pending_upload->height);
    if (upload_bytes <= static_cast<size_t>(std::numeric_limits<int>::max())) {
      QByteArray upload_data = QByteArray::fromRawData(
          static_cast<const char*>(pending_upload->pixels.get()), static_cast<int>(upload_bytes));
      QRhiTextureSubresourceUploadDescription upload_desc(upload_data);
      upload_desc.setDataStride(static_cast<quint32>(pending_upload->row_bytes));
      upload_desc.setSourceSize(QSize(pending_upload->width, pending_upload->height));
      resource_updates->uploadTexture(
          source_texture_, QRhiTextureUploadDescription(
                               QRhiTextureUploadEntry(0, 0, upload_desc)));
    }
  }

  UniformData uniform_data;
  if (frame) {
    const auto scale = ViewportMapper::ComputeLetterboxScale(
        widget_info, ViewportImageInfo{frame.width, frame.height});
    float zoom = view_state.snapshot.view_transform.zoom;
    float pan_x = view_state.snapshot.view_transform.pan.x();
    float pan_y = view_state.snapshot.view_transform.pan.y();
    if (frame.presentation_mode == FramePresentationMode::RoiFrame) {
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
        frame.presentation_mode == FramePresentationMode::RoiFrame ? 1.0f : 0.0f;
  }
  resource_updates->updateDynamicBuffer(uniform_buffer_, 0, sizeof(UniformData), &uniform_data);

  command_buffer->beginPass(render_target, Qt::black, {1.0f, 0}, resource_updates);
  if (frame && pipeline_ && shader_resource_bindings_ && vertex_buffer_) {
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

  const bool render_target_changed = bound_render_target_ != render_target;
  if (render_target_changed) {
    destroyResource(pipeline_);
    bound_render_target_ = render_target;
  }

  if (!source_texture_) {
    source_texture_ = rhi_->newTexture(QRhiTexture::RGBA32F, QSize(1, 1), 1);
    source_texture_->create();
    source_texture_width_  = 1;
    source_texture_height_ = 1;
    static_upload_pending_ = true;
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
      source_texture_) {
    return;
  }

  destroyResource(pipeline_);
  destroyResource(shader_resource_bindings_);
  destroyResource(source_texture_);

  source_texture_ = rhi_->newTexture(QRhiTexture::RGBA32F, QSize(frame.width, frame.height), 1);
  source_texture_->create();
  source_texture_width_  = frame.width;
  source_texture_height_ = frame.height;
  static_upload_pending_ = true;

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

RhiEditViewerSurface::RhiEditViewerSurface(QWidget* parent) : QRhiWidget(parent) {
  setAutoFillBackground(false);
  setMouseTracking(false);
  setApi(QRhiWidget::Api::Metal);
  setColorBufferFormat(QRhiWidget::TextureFormat::RGBA32F);
}

RhiEditViewerSurface::~RhiEditViewerSurface() = default;

auto RhiEditViewerSurface::widget() -> QWidget* { return this; }

void RhiEditViewerSurface::submitFrame(const ViewerFrame& frame) {
  latest_frame_ = frame;
  if (frame) {
    pending_upload_ = std::make_unique<ViewerGpuFrameUpload>(
        ViewerGpuFrameUpload{frame.width, frame.height, frame.row_bytes, frame.pixels,
                             frame.presentation_mode});
  } else {
    pending_upload_.reset();
  }
}

void RhiEditViewerSurface::setViewState(const ViewerViewState& state) { view_state_ = state; }

void RhiEditViewerSurface::requestRedraw() { update(); }

void RhiEditViewerSurface::initialize(QRhiCommandBuffer* command_buffer) {
  renderer_.initialize(rhi(), renderTarget(), command_buffer);
  if (latest_frame_) {
    pending_upload_ = std::make_unique<ViewerGpuFrameUpload>(
        ViewerGpuFrameUpload{latest_frame_.width, latest_frame_.height, latest_frame_.row_bytes,
                             latest_frame_.pixels, latest_frame_.presentation_mode});
  }
}

void RhiEditViewerSurface::render(QRhiCommandBuffer* command_buffer) {
  renderer_.render(
      command_buffer, renderTarget(), view_state_,
      ViewportWidgetInfo{width(), height(), static_cast<float>(devicePixelRatioF())}, latest_frame_,
      pending_upload_.get());
  pending_upload_.reset();
}

void RhiEditViewerSurface::releaseResources() { renderer_.releaseResources(); }

}  // namespace puerhlab
