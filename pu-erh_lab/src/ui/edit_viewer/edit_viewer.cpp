//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/edit_viewer/edit_viewer.hpp"

#include "ui/edit_viewer/opengl_viewer_renderer.hpp"

#include <qoverload.h>

#include <QApplication>
#include <QEasingCurve>

#include <algorithm>
#include <cmath>

namespace puerhlab {

QtEditViewer::QtEditViewer(QWidget* parent) : QOpenGLWidget(parent), renderer_(new OpenGLViewerRenderer()) {
  connect(this, &QtEditViewer::RequestUpdate, this, QOverload<>::of(&QtEditViewer::update),
          Qt::QueuedConnection);
  connect(this, &QtEditViewer::RequestResize, this, &QtEditViewer::OnResizeGL,
          Qt::BlockingQueuedConnection);

  zoom_animation_ = new QVariantAnimation(this);
  zoom_animation_->setDuration(kZoomAnimationDurationMs);
  zoom_animation_->setEasingCurve(QEasingCurve::InOutCubic);
  zoom_animation_->setStartValue(0.0);
  zoom_animation_->setEndValue(1.0);
  connect(zoom_animation_, &QVariantAnimation::valueChanged, this,
          [this](const QVariant& value) {
            const auto result = view_transform_controller_.ApplyAnimationProgress(
                viewer_state_, CurrentWidgetInfo(), CurrentImageInfo(),
                std::clamp(static_cast<float>(value.toDouble()), 0.0f, 1.0f));
            ApplyViewTransformResult(result);
          });
  connect(zoom_animation_, &QVariantAnimation::finished, this, [this]() {
    const auto result = view_transform_controller_.ApplyAnimationFinished(
        viewer_state_, CurrentWidgetInfo(), CurrentImageInfo());
    ApplyViewTransformResult(result);
  });

  click_toggle_timer_ = new QTimer(this);
  click_toggle_timer_->setSingleShot(true);
  connect(click_toggle_timer_, &QTimer::timeout, this, [this]() {
    const auto result = view_transform_controller_.HandleClickToggleTimeout(
        viewer_state_, CurrentWidgetInfo(), CurrentImageInfo());
    ApplyViewTransformResult(result);
  });
}

QtEditViewer::~QtEditViewer() {
  makeCurrent();
  if (renderer_) {
    renderer_->Shutdown();
    delete renderer_;
    renderer_ = nullptr;
  }
  doneCurrent();
}

void QtEditViewer::ResetView() {
  StopZoomAnimation();
  const auto result = view_transform_controller_.ResetView(viewer_state_);
  ApplyViewTransformResult(result);
}

void QtEditViewer::SetCropToolEnabled(bool enabled) {
  if (enabled) {
    StopZoomAnimation();
  }
  viewer_state_.SetCropToolEnabled(enabled);
  if (!enabled) {
    crop_interaction_controller_.Cancel();
  }
  const auto result = view_transform_controller_.HandleCropToolEnabledChanged(viewer_state_, enabled);
  ApplyViewTransformResult(result);
  update();
}

void QtEditViewer::SetCropOverlayVisible(bool visible) {
  viewer_state_.SetCropOverlayVisible(visible);
  update();
}

void QtEditViewer::SetCropOverlayRectNormalized(float x, float y, float w, float h) {
  auto crop_state = viewer_state_.GetCropOverlay();
  const QRectF clamped_rect = CropGeometry::ClampCropRect(QRectF(x, y, w, h));
  const QRectF adjusted_rect = CropGeometry::ClampCropRectForRotation(
      clamped_rect, crop_state.rotation_degrees, crop_state.metric_aspect);
  const bool rect_changed =
      std::abs(static_cast<float>(adjusted_rect.x() - crop_state.rect.x())) > 1e-6f ||
      std::abs(static_cast<float>(adjusted_rect.y() - crop_state.rect.y())) > 1e-6f ||
      std::abs(static_cast<float>(adjusted_rect.width() - crop_state.rect.width())) > 1e-6f ||
      std::abs(static_cast<float>(adjusted_rect.height() - crop_state.rect.height())) > 1e-6f;
  crop_state.rect = adjusted_rect;
  viewer_state_.SetCropOverlayState(crop_state);
  if (rect_changed) {
    emit CropOverlayRectChanged(static_cast<float>(adjusted_rect.x()),
                                static_cast<float>(adjusted_rect.y()),
                                static_cast<float>(adjusted_rect.width()),
                                static_cast<float>(adjusted_rect.height()), false);
  }
  update();
}

void QtEditViewer::SetCropOverlayRotationDegrees(float angle_degrees) {
  auto crop_state = viewer_state_.GetCropOverlay();
  crop_state.rotation_degrees = CropGeometry::NormalizeAngleDegrees(angle_degrees);
  const QRectF adjusted_rect = CropGeometry::ClampCropRectForRotation(
      crop_state.rect, crop_state.rotation_degrees, crop_state.metric_aspect);
  const bool rect_changed =
      std::abs(static_cast<float>(adjusted_rect.x() - crop_state.rect.x())) > 1e-6f ||
      std::abs(static_cast<float>(adjusted_rect.y() - crop_state.rect.y())) > 1e-6f ||
      std::abs(static_cast<float>(adjusted_rect.width() - crop_state.rect.width())) > 1e-6f ||
      std::abs(static_cast<float>(adjusted_rect.height() - crop_state.rect.height())) > 1e-6f;
  crop_state.rect = adjusted_rect;
  viewer_state_.SetCropOverlayState(crop_state);
  if (rect_changed) {
    emit CropOverlayRectChanged(static_cast<float>(adjusted_rect.x()),
                                static_cast<float>(adjusted_rect.y()),
                                static_cast<float>(adjusted_rect.width()),
                                static_cast<float>(adjusted_rect.height()), false);
  }
  update();
}

void QtEditViewer::SetCropOverlayAspectLock(bool enabled, float aspect_ratio) {
  auto crop_state = viewer_state_.GetCropOverlay();
  crop_state.aspect_locked = enabled;
  crop_state.aspect_ratio  = CropGeometry::ClampAspectRatio(aspect_ratio);
  viewer_state_.SetCropOverlayState(crop_state);
  update();
}

void QtEditViewer::ResetCropOverlayRectToFull() { SetCropOverlayRectNormalized(0.0f, 0.0f, 1.0f, 1.0f); }

auto QtEditViewer::GetViewZoom() const -> float { return viewer_state_.GetViewZoom(); }

void QtEditViewer::EnsureSize(int width, int height) {
  const auto decision = frame_mailbox_.EnsureSize(width, height);
  if (decision.need_resize) {
    emit RequestResize(width, height);
  }
}

auto QtEditViewer::MapResourceForWrite() -> FrameWriteMapping { return frame_mailbox_.MapResourceForWrite(); }

void QtEditViewer::UnmapResource() { frame_mailbox_.UnmapResource(); }

void QtEditViewer::NotifyFrameReady() {
  frame_mailbox_.NotifyFrameReady();
  emit RequestUpdate();
}

auto QtEditViewer::GetWidth() const -> int { return frame_mailbox_.GetWidth(); }

auto QtEditViewer::GetHeight() const -> int { return frame_mailbox_.GetHeight(); }

auto QtEditViewer::GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> {
  return viewer_state_.GetViewportRenderRegion();
}

void QtEditViewer::SetNextFramePresentationMode(FramePresentationMode mode) {
  frame_mailbox_.SetNextFramePresentationMode(mode);
}

void QtEditViewer::SetHistogramFrameExpected(bool expected_fast_preview) {
  frame_mailbox_.SetHistogramFrameExpected(expected_fast_preview);
}

void QtEditViewer::SetHistogramUpdateIntervalMs(int interval_ms) {
  if (renderer_) {
    renderer_->SetHistogramUpdateIntervalMs(interval_ms);
  }
}

auto QtEditViewer::GetHistogramBufferId() const -> GLuint {
  return renderer_ ? renderer_->GetHistogramBufferId() : 0;
}

auto QtEditViewer::GetHistogramBinCount() const -> int {
  return renderer_ ? renderer_->GetHistogramBinCount() : 0;
}

auto QtEditViewer::HasHistogramData() const -> bool {
  return renderer_ ? renderer_->HasHistogramData() : false;
}

void QtEditViewer::initializeGL() {
  if (!renderer_) {
    renderer_ = new OpenGLViewerRenderer();
  }
  renderer_->Initialize();
  frame_mailbox_.InitializeDefaultSize(std::max(1, width()), std::max(1, height()));
  const auto active_frame = frame_mailbox_.GetActiveFrame();
  renderer_->EnsureSlot(active_frame.slot_index, std::max(1, active_frame.width),
                        std::max(1, active_frame.height));
  UpdateViewportRenderRegionCache();
}

void QtEditViewer::resizeGL(int w, int h) {
  if (w <= 0 || h <= 0) {
    return;
  }
  UpdateViewportRenderRegionCache();
}

void QtEditViewer::paintGL() {
  if (!renderer_) {
    return;
  }

  const auto pending_frame = frame_mailbox_.ConsumePendingFrame();
  if (pending_frame.has_value() && renderer_->UploadPendingFrame(*pending_frame)) {
    frame_mailbox_.MarkFramePresented(pending_frame->slot_index, pending_frame->apply_presentation_mode);
  }

  const auto active_frame = frame_mailbox_.GetActiveFrame();
  if (active_frame.presentation_mode != FramePresentationMode::RoiFrame && active_frame.width > 0 &&
      active_frame.height > 0) {
    viewer_state_.SetRenderReferenceSize(active_frame.width, active_frame.height);
  }
  viewer_state_.SetCropOverlayMetricAspect(
      CropGeometry::SafeAspect(active_frame.width, active_frame.height));
  UpdateViewportRenderRegionCache();

  const auto render_result = renderer_->Render(*this, active_frame, viewer_state_.Snapshot(),
                                               frame_mailbox_.ConsumeHistogramPendingFrame());
  if (render_result.histogram_data_updated) {
    emit HistogramDataUpdated();
  }
}

void QtEditViewer::OnResizeGL(int w, int h) {
  if (!renderer_) {
    return;
  }

  const int target_slot = frame_mailbox_.GetRenderTargetSlotIndex();
  if (renderer_->HasSlot(target_slot, w, h)) {
    frame_mailbox_.CommitResize(target_slot, w, h);
    UpdateViewportRenderRegionCache();
    update();
    return;
  }

  makeCurrent();
  const bool resized = renderer_->EnsureSlot(target_slot, w, h);
  doneCurrent();
  if (resized) {
    frame_mailbox_.CommitResize(target_slot, w, h);
  }
  UpdateViewportRenderRegionCache();
  update();
}

void QtEditViewer::wheelEvent(QWheelEvent* event) {
  if ((event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier) {
    StopZoomAnimation();
    const auto result = view_transform_controller_.HandleCtrlWheel(
        viewer_state_, CurrentWidgetInfo(), CurrentImageInfo(), event->angleDelta().y(),
        event->position());
    ApplyViewTransformResult(result);
    event->accept();
    return;
  }

  event->accept();
}

void QtEditViewer::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    const auto crop_result =
        crop_interaction_controller_.HandlePress(viewer_state_, CurrentWidgetInfo(), CurrentImageInfo(),
                                                 event->position());
    if (crop_result.consumed) {
      ApplyCropInteractionResult(crop_result);
      event->accept();
      return;
    }
  }

  if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
    const auto crop_state = viewer_state_.GetCropOverlay();
    StopZoomAnimation();
    const auto result = view_transform_controller_.HandlePanPress(
        crop_state.tool_enabled && crop_state.overlay_visible, event->pos());
    if (result.consumed) {
      ApplyViewTransformResult(result);
      event->accept();
      return;
    }
  }

  QOpenGLWidget::mousePressEvent(event);
}

void QtEditViewer::mouseMoveEvent(QMouseEvent* event) {
  const auto crop_result = crop_interaction_controller_.HandleMove(
      viewer_state_, CurrentWidgetInfo(), CurrentImageInfo(), event->buttons(), event->position());
  if (crop_result.consumed) {
    ApplyCropInteractionResult(crop_result);
    event->accept();
    return;
  }

  const auto result = view_transform_controller_.HandlePanMove(
      viewer_state_, CurrentWidgetInfo(), CurrentImageInfo(), event->pos());
  if (result.consumed) {
    ApplyViewTransformResult(result);
    event->accept();
    return;
  }

  QOpenGLWidget::mouseMoveEvent(event);
}

void QtEditViewer::mouseReleaseEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    const auto crop_result = crop_interaction_controller_.HandleRelease(viewer_state_);
    if (crop_result.consumed) {
      ApplyCropInteractionResult(crop_result);
      event->accept();
      return;
    }
  }

  const auto crop_state = viewer_state_.GetCropOverlay();
  const auto result = view_transform_controller_.HandlePanRelease(
      viewer_state_, crop_state.tool_enabled && crop_state.overlay_visible, event->button(),
      event->position());
  if (result.consumed) {
    ApplyViewTransformResult(result);
    event->accept();
    return;
  }

  QOpenGLWidget::mouseReleaseEvent(event);
}

void QtEditViewer::mouseDoubleClickEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    const auto crop_result = crop_interaction_controller_.HandleDoubleClick(viewer_state_);
    if (crop_result.consumed) {
      ApplyCropInteractionResult(crop_result);
      event->accept();
      return;
    }

    const auto result = view_transform_controller_.HandleDoubleClick(
        viewer_state_, CurrentWidgetInfo(), CurrentImageInfo(), event->position());
    ApplyViewTransformResult(result);
    event->accept();
    return;
  }

  QOpenGLWidget::mouseDoubleClickEvent(event);
}

void QtEditViewer::UpdateViewportRenderRegionCache() {
  const auto snapshot = viewer_state_.Snapshot();
  if (snapshot.render_reference_width <= 0 || snapshot.render_reference_height <= 0) {
    viewer_state_.SetViewportRenderRegion(std::nullopt);
    return;
  }

  viewer_state_.SetViewportRenderRegion(ViewportMapper::ComputeViewportRenderRegion(
      CurrentWidgetInfo(), snapshot.view_transform.zoom, snapshot.view_transform.pan,
      snapshot.render_reference_width, snapshot.render_reference_height));
}

void QtEditViewer::StopZoomAnimation() {
  if (zoom_animation_ && zoom_animation_->state() == QAbstractAnimation::Running) {
    zoom_animation_->stop();
  }
}

auto QtEditViewer::CurrentWidgetInfo() const -> ViewportWidgetInfo {
  return {width(), height(), devicePixelRatioF()};
}

auto QtEditViewer::CurrentImageInfo() const -> ViewportImageInfo {
  const auto active_frame = frame_mailbox_.GetActiveFrame();
  return {active_frame.width, active_frame.height};
}

void QtEditViewer::ApplyViewTransformResult(const ViewTransformResult& result) {
  if (result.stop_click_toggle_timer && click_toggle_timer_ && click_toggle_timer_->isActive()) {
    click_toggle_timer_->stop();
  }
  if (result.start_click_toggle_timer && click_toggle_timer_) {
    click_toggle_timer_->start(QApplication::doubleClickInterval());
  }
  if (result.unset_cursor) {
    unsetCursor();
  } else if (result.cursor.has_value()) {
    setCursor(*result.cursor);
  }
  if (result.start_animation && zoom_animation_) {
    zoom_animation_->setDuration(kZoomAnimationDurationMs);
    zoom_animation_->setStartValue(0.0);
    zoom_animation_->setEndValue(1.0);
    zoom_animation_->start();
  }
  if (result.request_repaint) {
    UpdateViewportRenderRegionCache();
    update();
  }
  if (result.emitted_zoom.has_value()) {
    emit ViewZoomChanged(*result.emitted_zoom);
  }
}

void QtEditViewer::ApplyCropInteractionResult(const CropInteractionResult& result) {
  if (result.unset_cursor) {
    unsetCursor();
  } else if (result.cursor.has_value()) {
    setCursor(*result.cursor);
  }
  if (result.rect_changed.has_value()) {
    emit CropOverlayRectChanged(static_cast<float>(result.rect_changed->x()),
                                static_cast<float>(result.rect_changed->y()),
                                static_cast<float>(result.rect_changed->width()),
                                static_cast<float>(result.rect_changed->height()),
                                result.rect_is_final);
  }
  if (result.rotation_changed.has_value()) {
    emit CropOverlayRotationChanged(*result.rotation_changed, result.rotation_is_final);
  }
  if (result.request_repaint) {
    update();
  }
}

}  // namespace puerhlab
