//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <functional>
#include <optional>

#include <QOpenGLContext>

#include "ui/edit_viewer/edit_viewer_surface.hpp"
#include "ui/edit_viewer/frame_mailbox.hpp"

class QWidget;

namespace puerhlab {

class OpenGLViewerRenderer;

class IOpenGLEditViewerSurface : public virtual IEditViewerSurface {
 public:
  ~IOpenGLEditViewerSurface() override = default;

  virtual auto context() const -> QOpenGLContext* = 0;
  virtual void makeCurrent()                      = 0;
  virtual void doneCurrent()                      = 0;

  virtual void setHistogramUpdateIntervalMs(int interval_ms) = 0;
  virtual auto histogramBufferId() const -> unsigned int      = 0;
  virtual auto histogramBinCount() const -> int               = 0;
  virtual auto hasHistogramData() const -> bool               = 0;
};

class GlEditViewerSurface final : public IOpenGLEditViewerSurface,
                                  public IEditViewerRenderTargetSurface {
 public:
  struct Callbacks {
    std::function<std::optional<FrameMailbox::PendingFrame>()> consume_pending_frame;
    std::function<bool()> consume_histogram_request;
    std::function<void(int slot_index, bool apply_presentation_mode)> frame_presented;
    std::function<void()> histogram_data_updated;
  };

  explicit GlEditViewerSurface(const Callbacks& callbacks, QWidget* parent = nullptr);
  ~GlEditViewerSurface() override;

 auto widget() -> QWidget* override;
  void submitFrame(const ViewerFrame& frame) override;
  void setViewState(const ViewerViewState& state) override;
  void requestRedraw() override;

  auto hasRenderTarget(int slot_index, int width, int height) const -> bool override;
  auto ensureRenderTarget(int slot_index, int width, int height) -> bool override;

  auto context() const -> QOpenGLContext* override;
  void makeCurrent() override;
  void doneCurrent() override;

  void setHistogramUpdateIntervalMs(int interval_ms) override;
  auto histogramBufferId() const -> unsigned int override;
  auto histogramBinCount() const -> int override;
  auto hasHistogramData() const -> bool override;

 private:
  class SurfaceWidget;

  auto ensureRenderTargetCurrentContext(int slot_index, int width, int height) -> bool;
  void initialize();
  void resize(int w, int h);
  void paint();

  Callbacks             callbacks_{};
  SurfaceWidget*        widget_   = nullptr;
  OpenGLViewerRenderer* renderer_ = nullptr;
  FrameMailbox::ActiveFrame active_frame_{};
  ViewerViewState       view_state_{};
};

}  // namespace puerhlab
