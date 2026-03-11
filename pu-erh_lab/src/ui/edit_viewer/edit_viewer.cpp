//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/edit_viewer/edit_viewer.hpp"

#ifdef HAVE_CUDA
#include "ui/edit_viewer/gl_edit_viewer_surface.hpp"
#endif
#ifdef HAVE_METAL
#include "ui/edit_viewer/rhi_edit_viewer_surface.hpp"
#endif

#include <QApplication>
#include <QEasingCurve>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPolygonF>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

namespace puerhlab {

class EditViewerOverlayWidget final : public QWidget {
 public:
  explicit EditViewerOverlayWidget(QtEditViewer& owner, QWidget* parent = nullptr)
      : QWidget(parent), owner_(owner) {
    setAutoFillBackground(false);
    setMouseTracking(true);
    setAttribute(Qt::WA_NoSystemBackground);
    setFocusPolicy(Qt::NoFocus);
  }

 protected:
  void paintEvent(QPaintEvent*) override { owner_.PaintOverlay(*this); }
  void wheelEvent(QWheelEvent* event) override { owner_.HandleOverlayWheel(event); }
  void mousePressEvent(QMouseEvent* event) override { owner_.HandleOverlayMousePress(event); }
  void mouseMoveEvent(QMouseEvent* event) override { owner_.HandleOverlayMouseMove(event); }
  void mouseReleaseEvent(QMouseEvent* event) override { owner_.HandleOverlayMouseRelease(event); }
  void mouseDoubleClickEvent(QMouseEvent* event) override {
    owner_.HandleOverlayMouseDoubleClick(event);
  }
  void leaveEvent(QEvent*) override { owner_.HandleOverlayLeave(); }

 private:
  QtEditViewer& owner_;
};

QtEditViewer::QtEditViewer(QWidget* parent) : QWidget(parent) {
  setContentsMargins(0, 0, 0, 0);
  setAutoFillBackground(false);

#ifdef HAVE_CUDA
  GlEditViewerSurface::Callbacks surface_callbacks;
  surface_callbacks.consume_pending_frame = [this]() { return frame_mailbox_.ConsumePendingFrame(); };
  surface_callbacks.consume_histogram_request = [this]() {
    return frame_mailbox_.ConsumeHistogramPendingFrame();
  };
  surface_callbacks.frame_presented = [this](int slot_index, bool apply_presentation_mode) {
    frame_mailbox_.MarkFramePresented(slot_index, apply_presentation_mode);
    RefreshFrameDerivedState();
    UpdateOverlay();
  };
  surface_callbacks.histogram_data_updated = [this]() { emit HistogramDataUpdated(); };

  surface_ = std::make_unique<GlEditViewerSurface>(surface_callbacks, this);
  render_target_surface_ = dynamic_cast<IEditViewerRenderTargetSurface*>(surface_.get());
#elif defined(HAVE_METAL)
  surface_ = std::make_unique<RhiEditViewerSurface>(this);
#endif

  overlay_ = new EditViewerOverlayWidget(*this, this);
#ifdef HAVE_CUDA
  frame_mailbox_.InitializeDefaultSize(std::max(1, width()), std::max(1, height()));
#endif
  RefreshFrameDerivedState();
  ResizeChildWidgets();
  overlay_->raise();

  connect(this, &QtEditViewer::RequestUpdate, this, &QtEditViewer::HandleQueuedUpdate,
          Qt::QueuedConnection);
  connect(this, &QtEditViewer::RequestResize, this, &QtEditViewer::OnResizeSurface,
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

QtEditViewer::~QtEditViewer() = default;

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
  UpdateOverlay();
}

void QtEditViewer::SetCropOverlayVisible(bool visible) {
  viewer_state_.SetCropOverlayVisible(visible);
  if (!visible) {
    ApplyOverlayCursor(std::nullopt, true);
  }
  UpdateOverlay();
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
  UpdateOverlay();
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
  UpdateOverlay();
}

void QtEditViewer::SetCropOverlayAspectLock(bool enabled, float aspect_ratio) {
  auto crop_state = viewer_state_.GetCropOverlay();
  crop_state.aspect_locked = enabled;
  crop_state.aspect_ratio  = CropGeometry::ClampAspectRatio(aspect_ratio);
  viewer_state_.SetCropOverlayState(crop_state);
  UpdateOverlay();
}

void QtEditViewer::ResetCropOverlayRectToFull() { SetCropOverlayRectNormalized(0.0f, 0.0f, 1.0f, 1.0f); }

auto QtEditViewer::GetViewZoom() const -> float { return viewer_state_.GetViewZoom(); }

void QtEditViewer::EnsureSize(int width, int height) {
#ifdef HAVE_CUDA
  const auto decision = frame_mailbox_.EnsureSize(width, height);
  if (decision.need_resize) {
    emit RequestResize(width, height);
  }
#else
  (void)width;
  (void)height;
#endif
}

auto QtEditViewer::MapResourceForWrite() -> FrameWriteMapping {
#ifdef HAVE_CUDA
  return frame_mailbox_.MapResourceForWrite();
#else
  return {};
#endif
}

void QtEditViewer::UnmapResource() {
#ifdef HAVE_CUDA
  frame_mailbox_.UnmapResource();
#endif
}

void QtEditViewer::NotifyFrameReady() {
#ifdef HAVE_CUDA
  frame_mailbox_.NotifyFrameReady();
  emit RequestUpdate();
#endif
}

void QtEditViewer::SubmitHostFrame(const ViewerFrame& frame) {
#ifdef HAVE_METAL
  {
    std::lock_guard<std::mutex> lock(host_frame_mutex_);
    ViewerFrame submitted_frame = frame;
    if (pending_presentation_mode_valid_) {
      submitted_frame.presentation_mode = pending_presentation_mode_;
      pending_presentation_mode_valid_ = false;
    }
    pending_host_frame_ = std::move(submitted_frame);
  }
  emit RequestUpdate();
#else
  (void)frame;
#endif
}

auto QtEditViewer::GetWidth() const -> int {
#ifdef HAVE_CUDA
  return frame_mailbox_.GetWidth();
#else
  std::lock_guard<std::mutex> lock(host_frame_mutex_);
  return active_host_frame_.width;
#endif
}

auto QtEditViewer::GetHeight() const -> int {
#ifdef HAVE_CUDA
  return frame_mailbox_.GetHeight();
#else
  std::lock_guard<std::mutex> lock(host_frame_mutex_);
  return active_host_frame_.height;
#endif
}

auto QtEditViewer::GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> {
  return viewer_state_.GetViewportRenderRegion();
}

void QtEditViewer::SetNextFramePresentationMode(FramePresentationMode mode) {
#ifdef HAVE_CUDA
  frame_mailbox_.SetNextFramePresentationMode(mode);
#else
  std::lock_guard<std::mutex> lock(host_frame_mutex_);
  pending_presentation_mode_       = mode;
  pending_presentation_mode_valid_ = true;
#endif
}

auto QtEditViewer::GetViewerSurface() -> IEditViewerSurface* { return surface_.get(); }

auto QtEditViewer::GetViewerSurface() const -> const IEditViewerSurface* { return surface_.get(); }

void QtEditViewer::SetHistogramFrameExpected(bool expected_fast_preview) {
#ifdef HAVE_CUDA
  frame_mailbox_.SetHistogramFrameExpected(expected_fast_preview);
#else
  (void)expected_fast_preview;
#endif
}

void QtEditViewer::SetHistogramUpdateIntervalMs(int interval_ms) {
#ifdef HAVE_CUDA
  auto* gl_surface = dynamic_cast<IOpenGLEditViewerSurface*>(surface_.get());
  if (gl_surface) {
    gl_surface->setHistogramUpdateIntervalMs(interval_ms);
  }
#else
  (void)interval_ms;
#endif
}

void QtEditViewer::resizeEvent(QResizeEvent* event) {
  QWidget::resizeEvent(event);
  ResizeChildWidgets();
  UpdateSurface();
  UpdateOverlay();
}

void QtEditViewer::PaintOverlay(QWidget& widget) {
  const auto snapshot = CurrentOverlaySnapshot();
  if (!snapshot.viewer_state.crop_overlay.overlay_visible) {
    return;
  }

  const auto geometry = EditViewerOverlayGeometry::Build(snapshot);
  if (!geometry.image_rect_valid || !geometry.crop_corners_valid) {
    return;
  }

  QPolygonF crop_polygon;
  crop_polygon.reserve(static_cast<int>(geometry.crop_corners_widget.size()));
  for (const auto& point : geometry.crop_corners_widget) {
    crop_polygon.push_back(point);
  }

  QPainter painter(&widget);
  painter.setRenderHint(QPainter::Antialiasing, true);

  QPainterPath image_path;
  image_path.addRect(geometry.image_rect);
  QPainterPath crop_path;
  crop_path.addPolygon(crop_polygon);
  crop_path.closeSubpath();
  painter.fillPath(image_path.subtracted(crop_path), QColor(0, 0, 0, 110));

  painter.setPen(QPen(QColor(252, 199, 4, 220), 1.2));
  painter.setBrush(Qt::NoBrush);
  painter.drawPolygon(crop_polygon);
  painter.drawLine(geometry.rotate_stem_widget, geometry.rotate_handle_widget);

  painter.setPen(QPen(QColor(252, 199, 4, 150), 1.0, Qt::DashLine));
  for (const float t : {1.0f / 3.0f, 2.0f / 3.0f}) {
    painter.drawLine(CropGeometry::LerpPoint(geometry.crop_corners_widget[0],
                                             geometry.crop_corners_widget[1], t),
                     CropGeometry::LerpPoint(geometry.crop_corners_widget[3],
                                             geometry.crop_corners_widget[2], t));
    painter.drawLine(CropGeometry::LerpPoint(geometry.crop_corners_widget[0],
                                             geometry.crop_corners_widget[3], t),
                     CropGeometry::LerpPoint(geometry.crop_corners_widget[1],
                                             geometry.crop_corners_widget[2], t));
  }

  painter.setPen(QPen(QColor(18, 18, 18, 230), 1.0));
  painter.setBrush(QColor(252, 199, 4, 230));
  for (const auto& corner : geometry.crop_corners_widget) {
    painter.drawEllipse(corner, CropGeometry::kCropCornerDrawRadiusPx,
                        CropGeometry::kCropCornerDrawRadiusPx);
  }
  painter.setBrush(QColor(252, 199, 4, 245));
  painter.drawEllipse(geometry.rotate_handle_widget, CropGeometry::kCropRotateHandleDrawRadiusPx,
                      CropGeometry::kCropRotateHandleDrawRadiusPx);
}

void QtEditViewer::HandleOverlayWheel(QWheelEvent* event) {
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

void QtEditViewer::HandleOverlayMousePress(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    const auto hover = CurrentOverlayHover(event->position());
    const CropPressContext press_context{event->position(), hover.image_uv, hover.crop_hit,
                                         hover.inside_image};
    const auto crop_result =
        crop_interaction_controller_.HandlePress(viewer_state_, CurrentImageInfo(), press_context);
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

  event->ignore();
}

void QtEditViewer::HandleOverlayMouseMove(QMouseEvent* event) {
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

  const auto hover = CurrentOverlayHover(event->position());
  ApplyOverlayCursor(hover.cursor, !hover.cursor.has_value());
  event->accept();
}

void QtEditViewer::HandleOverlayMouseRelease(QMouseEvent* event) {
  bool consumed = false;

  if (event->button() == Qt::LeftButton) {
    const auto crop_result = crop_interaction_controller_.HandleRelease(viewer_state_);
    if (crop_result.consumed) {
      ApplyCropInteractionResult(crop_result);
      consumed = true;
    }
  }

  const auto crop_state = viewer_state_.GetCropOverlay();
  const auto result = view_transform_controller_.HandlePanRelease(
      viewer_state_, crop_state.tool_enabled && crop_state.overlay_visible, event->button(),
      event->position());
  if (result.consumed) {
    ApplyViewTransformResult(result);
    consumed = true;
  }

  const auto hover = CurrentOverlayHover(event->position());
  ApplyOverlayCursor(hover.cursor, !hover.cursor.has_value());

  if (consumed) {
    event->accept();
    return;
  }
  event->ignore();
}

void QtEditViewer::HandleOverlayMouseDoubleClick(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    const auto crop_state = viewer_state_.GetCropOverlay();
    if (crop_state.tool_enabled && crop_state.overlay_visible) {
      const auto crop_result = crop_interaction_controller_.HandleDoubleClick(viewer_state_);
      if (crop_result.consumed) {
        ApplyCropInteractionResult(crop_result);
        event->accept();
        return;
      }
    }

    const auto result = view_transform_controller_.HandleDoubleClick(
        viewer_state_, CurrentWidgetInfo(), CurrentImageInfo(), event->position());
    ApplyViewTransformResult(result);
    event->accept();
    return;
  }

  event->ignore();
}

void QtEditViewer::HandleOverlayLeave() { ApplyOverlayCursor(std::nullopt, true); }

void QtEditViewer::HandleQueuedUpdate() {
  UpdateSurface();
  UpdateOverlay();
}

void QtEditViewer::OnResizeSurface(int w, int h) {
#ifdef HAVE_CUDA
  if (!surface_ || !render_target_surface_) {
    return;
  }

  const int target_slot = frame_mailbox_.GetRenderTargetSlotIndex();
  if (render_target_surface_->hasRenderTarget(target_slot, w, h)) {
    frame_mailbox_.CommitResize(target_slot, w, h);
    UpdateSurface();
    UpdateOverlay();
    return;
  }

  const bool resized = render_target_surface_->ensureRenderTarget(target_slot, w, h);
  if (resized) {
    frame_mailbox_.CommitResize(target_slot, w, h);
  }
  UpdateSurface();
  UpdateOverlay();
#else
  (void)w;
  (void)h;
#endif
}

void QtEditViewer::ResizeChildWidgets() {
  const QRect area = rect();
  if (surface_) {
    if (QWidget* surface_widget = surface_->widget()) {
      surface_widget->setGeometry(area);
    }
  }
  if (overlay_) {
    overlay_->setGeometry(area);
    overlay_->raise();
  }
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

void QtEditViewer::RefreshFrameDerivedState() {
#ifdef HAVE_CUDA
  const auto active_frame = frame_mailbox_.GetActiveFrame();
#else
  ViewerFrame active_frame;
  {
    std::lock_guard<std::mutex> lock(host_frame_mutex_);
    active_frame = active_host_frame_;
  }
#endif
  if (active_frame.presentation_mode != FramePresentationMode::RoiFrame && active_frame.width > 0 &&
      active_frame.height > 0) {
    viewer_state_.SetRenderReferenceSize(active_frame.width, active_frame.height);
  }
  viewer_state_.SetCropOverlayMetricAspect(
      CropGeometry::SafeAspect(active_frame.width, active_frame.height));
  UpdateViewportRenderRegionCache();
}

void QtEditViewer::SyncSurfaceState() {
  if (!surface_) {
    return;
  }

#ifdef HAVE_METAL
  {
    std::lock_guard<std::mutex> lock(host_frame_mutex_);
    if (pending_host_frame_.has_value()) {
      active_host_frame_ = *pending_host_frame_;
      surface_->submitFrame(*pending_host_frame_);
      pending_host_frame_.reset();
    }
  }
#endif

  RefreshFrameDerivedState();
  surface_->setViewState({viewer_state_.Snapshot()});
}

void QtEditViewer::StopZoomAnimation() {
  if (zoom_animation_ && zoom_animation_->state() == QAbstractAnimation::Running) {
    zoom_animation_->stop();
  }
}

auto QtEditViewer::CurrentWidgetInfo() const -> ViewportWidgetInfo {
  return {width(), height(), static_cast<float>(devicePixelRatioF())};
}

auto QtEditViewer::CurrentImageInfo() const -> ViewportImageInfo {
#ifdef HAVE_CUDA
  const auto active_frame = frame_mailbox_.GetActiveFrame();
  return {active_frame.width, active_frame.height};
#else
  std::lock_guard<std::mutex> lock(host_frame_mutex_);
  return {active_host_frame_.width, active_host_frame_.height};
#endif
}

auto QtEditViewer::CurrentPresentationMode() const -> FramePresentationMode {
#ifdef HAVE_CUDA
  return frame_mailbox_.GetActiveFrame().presentation_mode;
#else
  std::lock_guard<std::mutex> lock(host_frame_mutex_);
  return active_host_frame_.presentation_mode;
#endif
}

auto QtEditViewer::CurrentOverlaySnapshot() const -> EditViewerOverlaySnapshot {
  return {viewer_state_.Snapshot(), CurrentWidgetInfo(), CurrentImageInfo(), CurrentPresentationMode()};
}

auto QtEditViewer::CurrentOverlayHover(const QPointF& event_pos) const -> EditViewerOverlayHover {
  const auto snapshot = CurrentOverlaySnapshot();
  const auto geometry = EditViewerOverlayGeometry::Build(snapshot);
  return EditViewerOverlayGeometry::ComputeHover(snapshot, geometry, event_pos);
}

void QtEditViewer::ApplyOverlayCursor(std::optional<Qt::CursorShape> cursor, bool unset) {
  if (!overlay_) {
    return;
  }
  if (unset) {
    overlay_->unsetCursor();
    return;
  }
  if (cursor.has_value()) {
    overlay_->setCursor(*cursor);
  }
}

void QtEditViewer::ApplyViewTransformResult(const ViewTransformResult& result) {
  if (result.stop_click_toggle_timer && click_toggle_timer_ && click_toggle_timer_->isActive()) {
    click_toggle_timer_->stop();
  }
  if (result.start_click_toggle_timer && click_toggle_timer_) {
    click_toggle_timer_->start(QApplication::doubleClickInterval());
  }
  ApplyOverlayCursor(result.cursor, result.unset_cursor);
  if (result.start_animation && zoom_animation_) {
    zoom_animation_->setDuration(kZoomAnimationDurationMs);
    zoom_animation_->setStartValue(0.0);
    zoom_animation_->setEndValue(1.0);
    zoom_animation_->start();
  }
  if (result.request_repaint) {
    UpdateViewportRenderRegionCache();
    UpdateSurface();
    UpdateOverlay();
  }
  if (result.emitted_zoom.has_value()) {
    emit ViewZoomChanged(*result.emitted_zoom);
  }
}

void QtEditViewer::ApplyCropInteractionResult(const CropInteractionResult& result) {
  ApplyOverlayCursor(result.cursor, result.unset_cursor);
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
    UpdateOverlay();
  }
}

void QtEditViewer::UpdateSurface() {
  if (surface_) {
    SyncSurfaceState();
    surface_->requestRedraw();
  }
}

void QtEditViewer::UpdateOverlay() {
  if (overlay_) {
    overlay_->update();
  }
}

}  // namespace puerhlab
