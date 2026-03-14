//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstdint>
#include <memory>

#include <QtGui/rhi/qshader.h>
#include <QtWidgets/qrhiwidget.h>

#include "ui/edit_viewer/edit_viewer_surface.hpp"
#include "ui/edit_viewer/viewport_mapper.hpp"

class QRhi;
class QRhiBuffer;
class QRhiCommandBuffer;
class QRhiGraphicsPipeline;
class QRhiRenderTarget;
class QRhiSampler;
class QRhiShaderResourceBindings;
class QRhiTexture;

namespace puerhlab {

struct ImportedTextureFrame {
  int                   width              = 0;
  int                   height             = 0;
  std::uintptr_t        texture_handle     = 0;
  FramePresentationMode presentation_mode  = FramePresentationMode::FullFrame;

  explicit operator bool() const {
    return width > 0 && height > 0 && texture_handle != 0;
  }
};

class RhiImageRenderer {
 public:
  RhiImageRenderer() = default;
  ~RhiImageRenderer();

  void initialize(QRhi* rhi, QRhiRenderTarget* render_target, QRhiCommandBuffer* command_buffer);
  void releaseResources();
  void render(QRhiCommandBuffer* command_buffer, QRhiRenderTarget* render_target,
              const ViewerViewState& view_state, const ViewportWidgetInfo& widget_info,
              const ViewerFrame& frame, const ImportedTextureFrame& imported_frame,
              ViewerGpuFrameUpload* pending_upload);

 private:
  struct UniformData;
  struct VertexData;

  auto loadShader(const char* resource_path) const -> QShader;
  void destroyResource(QRhiTexture*& resource);
  void destroyResource(QRhiSampler*& resource);
  void destroyResource(QRhiBuffer*& resource);
  void destroyResource(QRhiShaderResourceBindings*& resource);
  void destroyResource(QRhiGraphicsPipeline*& resource);
  void ensureStaticResources(QRhiRenderTarget* render_target, QRhiCommandBuffer* command_buffer);
  void ensureSourceTexture(const ViewerFrame& frame, QRhiCommandBuffer* command_buffer);
  void ensureImportedSourceTexture(const ImportedTextureFrame& frame,
                                   QRhiCommandBuffer* command_buffer);
  void recreateShaderResources();

  QRhi*                         rhi_                          = nullptr;
  QRhiTexture*                  source_texture_               = nullptr;
  QRhiSampler*                  sampler_                      = nullptr;
  QRhiBuffer*                   uniform_buffer_               = nullptr;
  QRhiBuffer*                   vertex_buffer_                = nullptr;
  QRhiShaderResourceBindings*   shader_resource_bindings_     = nullptr;
  QRhiGraphicsPipeline*         pipeline_                     = nullptr;
  QRhiRenderTarget*             bound_render_target_          = nullptr;
  int                           source_texture_width_         = 0;
  int                           source_texture_height_        = 0;
  quint64                       source_texture_native_object_ = 0;
  bool                          source_texture_is_imported_   = false;
  bool                          static_upload_pending_        = false;
};

class RhiEditViewerSurface final : public QRhiWidget,
                                   public IEditViewerSurface,
                                   public IEditViewerRenderTargetSurface {
 public:
  explicit RhiEditViewerSurface(QWidget* parent = nullptr);
  ~RhiEditViewerSurface() override;

  auto widget() -> QWidget* override;
  void submitFrame(const ViewerFrame& frame) override;
#ifdef HAVE_METAL
  void submitMetalFrame(const ViewerMetalFrame& frame) override;
#endif
  void setDisplayConfig(const ViewerDisplayConfig& config) override;
  void setViewState(const ViewerViewState& state) override;
  void requestRedraw() override;

  auto supportsDirectCudaPresent() const -> bool override;
  auto prepareRenderTarget(int width, int height) -> EditViewerRenderTargetResizeDecision override;
  void commitRenderTargetResize(int slot_index, int width, int height) override;
  auto mapResourceForWrite() -> FrameWriteMapping override;
  void unmapResource() override;
  void notifyFrameReady() override;
  void setNextFramePresentationMode(FramePresentationMode mode) override;
  auto activeRenderTargetState() const -> EditViewerRenderTargetState override;
  auto hasRenderTarget(int slot_index, int width, int height) const -> bool override;
  auto ensureRenderTarget(int slot_index, int width, int height) -> bool override;

 protected:
  void initialize(QRhiCommandBuffer* command_buffer) override;
  void render(QRhiCommandBuffer* command_buffer) override;
 void releaseResources() override;

 private:
 public:
  struct PlatformState;

 private:
  void applyDisplayConfig();
  void releasePlatformTargets();
  void resetImportedFrameState();

  RhiImageRenderer               renderer_{};
  ViewerViewState                view_state_{};
  ViewerFrame                    latest_frame_{};
  ImportedTextureFrame           latest_imported_frame_{};
  std::shared_ptr<const void>    latest_imported_owner_{};
  std::unique_ptr<ViewerGpuFrameUpload> pending_upload_{};
  ViewerDisplayConfig            display_config_{};
  bool                           display_config_dirty_ = true;
  std::unique_ptr<PlatformState> platform_state_{};
};

}  // namespace puerhlab
