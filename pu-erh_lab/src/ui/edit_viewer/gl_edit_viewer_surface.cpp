//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/edit_viewer/gl_edit_viewer_surface.hpp"

#include <QOpenGLWidget>

#include <algorithm>

#include "opengl_viewer_renderer.hpp"

namespace puerhlab {

class GlEditViewerSurface::SurfaceWidget final : public QOpenGLWidget {
 public:
  explicit SurfaceWidget(GlEditViewerSurface& owner, QWidget* parent = nullptr)
      : QOpenGLWidget(parent), owner_(owner) {
    setAutoFillBackground(false);
    setMouseTracking(false);
  }

 protected:
  void initializeGL() override { owner_.initialize(); }
  void resizeGL(int w, int h) override { owner_.resize(w, h); }
  void paintGL() override { owner_.paint(); }

 private:
  GlEditViewerSurface& owner_;
};

GlEditViewerSurface::GlEditViewerSurface(const Callbacks& callbacks, QWidget* parent)
    : callbacks_(callbacks), widget_(new SurfaceWidget(*this, parent)),
      renderer_(new OpenGLViewerRenderer()) {}

GlEditViewerSurface::~GlEditViewerSurface() {
  if (!widget_ || !renderer_ || !widget_->context()) {
    delete renderer_;
    renderer_ = nullptr;
    return;
  }

  widget_->makeCurrent();
  renderer_->Shutdown();
  delete renderer_;
  renderer_ = nullptr;
  widget_->doneCurrent();
}

auto GlEditViewerSurface::widget() -> QWidget* { return widget_; }

void GlEditViewerSurface::submitFrame(const ViewerFrame& frame) { (void)frame; }

void GlEditViewerSurface::setViewState(const ViewerViewState& state) { view_state_ = state; }

void GlEditViewerSurface::requestRedraw() {
  if (widget_) {
    widget_->update();
  }
}

auto GlEditViewerSurface::hasRenderTarget(int slot_index, int width, int height) const -> bool {
  return renderer_ && renderer_->HasSlot(slot_index, width, height);
}

auto GlEditViewerSurface::ensureRenderTarget(int slot_index, int width, int height) -> bool {
  if (!renderer_ || !widget_ || !widget_->context()) {
    return false;
  }

  widget_->makeCurrent();
  const bool ensured = ensureRenderTargetCurrentContext(slot_index, width, height);
  widget_->doneCurrent();
  return ensured;
}

auto GlEditViewerSurface::context() const -> QOpenGLContext* {
  return widget_ ? widget_->context() : nullptr;
}

void GlEditViewerSurface::makeCurrent() {
  if (widget_ && widget_->context()) {
    widget_->makeCurrent();
  }
}

void GlEditViewerSurface::doneCurrent() {
  if (widget_ && widget_->context()) {
    widget_->doneCurrent();
  }
}

void GlEditViewerSurface::setHistogramUpdateIntervalMs(int interval_ms) {
  if (renderer_) {
    renderer_->SetHistogramUpdateIntervalMs(interval_ms);
  }
}

auto GlEditViewerSurface::histogramBufferId() const -> unsigned int {
  return renderer_ ? renderer_->GetHistogramBufferId() : 0;
}

auto GlEditViewerSurface::histogramBinCount() const -> int {
  return renderer_ ? renderer_->GetHistogramBinCount() : 0;
}

auto GlEditViewerSurface::hasHistogramData() const -> bool {
  return renderer_ ? renderer_->HasHistogramData() : false;
}

auto GlEditViewerSurface::ensureRenderTargetCurrentContext(int slot_index, int width, int height)
    -> bool {
  return renderer_ && renderer_->EnsureSlot(slot_index, width, height);
}

void GlEditViewerSurface::initialize() {
  if (!renderer_) {
    renderer_ = new OpenGLViewerRenderer();
  }
  renderer_->Initialize();
}

void GlEditViewerSurface::resize(int w, int h) {
  if (w <= 0 || h <= 0) {
    return;
  }
}

void GlEditViewerSurface::paint() {
  if (!renderer_ || !widget_) {
    return;
  }

  auto active_frame = active_frame_;
  if (callbacks_.consume_pending_frame) {
    if (const auto pending_frame = callbacks_.consume_pending_frame(); pending_frame.has_value()) {
      if (ensureRenderTargetCurrentContext(pending_frame->slot_index, pending_frame->width,
                                           pending_frame->height) &&
          renderer_->UploadPendingFrame(*pending_frame)) {
        if (callbacks_.frame_presented) {
          callbacks_.frame_presented(pending_frame->slot_index,
                                     pending_frame->apply_presentation_mode);
        }
        active_frame.slot_index = pending_frame->slot_index;
        active_frame.width      = pending_frame->width;
        active_frame.height     = pending_frame->height;
        if (pending_frame->apply_presentation_mode) {
          active_frame.presentation_mode = pending_frame->presentation_mode;
        }
        active_frame_ = active_frame;
      }
    }
  }

  if (active_frame.width > 0 && active_frame.height > 0 &&
      !renderer_->HasSlot(active_frame.slot_index, active_frame.width, active_frame.height)) {
    (void)ensureRenderTargetCurrentContext(active_frame.slot_index, active_frame.width,
                                           active_frame.height);
  }

  const bool histogram_requested =
      callbacks_.consume_histogram_request ? callbacks_.consume_histogram_request() : false;
  const auto render_result =
      renderer_->Render(*widget_, active_frame, view_state_.snapshot, histogram_requested);
  if (render_result.histogram_data_updated && callbacks_.histogram_data_updated) {
    callbacks_.histogram_data_updated();
  }
}

}  // namespace puerhlab
