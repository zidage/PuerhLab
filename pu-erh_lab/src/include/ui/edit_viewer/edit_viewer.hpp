//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QOpenGLContext>
#include <QRectF>
#include <QResizeEvent>
#include <QTimer>
#include <QVariantAnimation>
#include <QWidget>

#include "ui/edit_viewer/crop_interaction_controller.hpp"
#include "ui/edit_viewer/crop_geometry.hpp"
#include "ui/edit_viewer/edit_viewer_overlay_geometry.hpp"
#include "ui/edit_viewer/frame_mailbox.hpp"
#include "ui/edit_viewer/frame_sink.hpp"
#include "ui/edit_viewer/view_transform_controller.hpp"
#include "ui/edit_viewer/viewer_state.hpp"
#include "ui/edit_viewer/viewport_mapper.hpp"

class QMouseEvent;
class QOpenGLWidget;
class QWheelEvent;

namespace puerhlab {

class OpenGLViewerRenderer;
class EditViewerSurfaceWidget;
class EditViewerOverlayWidget;

class QtEditViewer : public QWidget, public puerhlab::IFrameSink {
  Q_OBJECT
 public:
  explicit QtEditViewer(QWidget* parent = nullptr);
  ~QtEditViewer() override;

  // Reset zoom/pan to default view
  void ResetView();
  void SetCropToolEnabled(bool enabled);
  void SetCropOverlayVisible(bool visible);
  void SetCropOverlayRectNormalized(float x, float y, float w, float h);
  void SetCropOverlayRotationDegrees(float angle_degrees);
  void SetCropOverlayAspectLock(bool enabled, float aspect_ratio);
  void ResetCropOverlayRectToFull();
  auto GetViewZoom() const -> float;

  // Overrides from IFrameSink
  void    EnsureSize(int width, int height) override;
  auto    MapResourceForWrite() -> FrameWriteMapping override;
  void    UnmapResource() override;
  void    NotifyFrameReady() override;
  int     GetWidth() const override;
  int     GetHeight() const override;
  auto    GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> override;
  void    SetNextFramePresentationMode(FramePresentationMode mode) override;

  void    SetHistogramFrameExpected(bool expected_fast_preview);
  void    SetHistogramUpdateIntervalMs(int interval_ms);

  auto    GetHistogramBufferId() const -> GLuint;
  auto    GetHistogramBinCount() const -> int;
  auto    HasHistogramData() const -> bool;
  auto    GetRenderSurfaceContext() const -> QOpenGLContext*;
  void    MakeRenderSurfaceCurrent();
  void    DoneRenderSurfaceCurrent();

 signals:
  void RequestUpdate();

  void RequestResize(int width, int height);
  void HistogramDataUpdated();
  void CropOverlayRectChanged(float x, float y, float w, float h, bool is_final);
  void CropOverlayRotationChanged(float angle_degrees, bool is_final);
  void ViewZoomChanged(float zoom);

 private slots:
  void OnResizeGL(int w, int h);

 protected:
  void resizeEvent(QResizeEvent* event) override;

 private:
  friend class EditViewerSurfaceWidget;
  friend class EditViewerOverlayWidget;

  void                    HandleQueuedUpdate();
  void                    ResizeChildWidgets();
  void                    UpdateViewportRenderRegionCache();
  void                    StopZoomAnimation();
  auto                    CurrentWidgetInfo() const -> ViewportWidgetInfo;
  auto                    CurrentImageInfo() const -> ViewportImageInfo;
  auto                    CurrentPresentationMode() const -> FramePresentationMode;
  auto                    CurrentOverlaySnapshot() const -> EditViewerOverlaySnapshot;
  auto                    CurrentOverlayHover(const QPointF& event_pos) const -> EditViewerOverlayHover;
  void                    ApplyOverlayCursor(std::optional<Qt::CursorShape> cursor, bool unset);
  void                    ApplyViewTransformResult(const ViewTransformResult& result);
  void                    ApplyCropInteractionResult(const CropInteractionResult& result);
  void                    UpdateSurface();
  void                    UpdateOverlay();

  void                    InitializeSurfaceGL();
  void                    ResizeSurfaceGL(int w, int h);
  void                    PaintSurfaceGL(QOpenGLWidget& widget);
  void                    PaintOverlay(QWidget& widget);
  void                    HandleOverlayWheel(QWheelEvent* event);
  void                    HandleOverlayMousePress(QMouseEvent* event);
  void                    HandleOverlayMouseMove(QMouseEvent* event);
  void                    HandleOverlayMouseRelease(QMouseEvent* event);
  void                    HandleOverlayMouseDoubleClick(QMouseEvent* event);
  void                    HandleOverlayLeave();

  static constexpr int     kZoomAnimationDurationMs = 170;

  ViewerState              viewer_state_{};
  FrameMailbox             frame_mailbox_{};
  ViewTransformController  view_transform_controller_{};
  CropInteractionController crop_interaction_controller_{};
  EditViewerSurfaceWidget* surface_ = nullptr;
  EditViewerOverlayWidget* overlay_ = nullptr;
  OpenGLViewerRenderer*    renderer_ = nullptr;
  QVariantAnimation*       zoom_animation_ = nullptr;
  QTimer*                  click_toggle_timer_ = nullptr;
};
};  // namespace puerhlab
