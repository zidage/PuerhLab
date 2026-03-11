//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QRectF>
#include <QResizeEvent>
#include <QTimer>
#include <QVariantAnimation>
#include <QWidget>

#include <memory>
#include <mutex>
#include <optional>

#include "ui/edit_viewer/crop_interaction_controller.hpp"
#include "ui/edit_viewer/crop_geometry.hpp"
#include "ui/edit_viewer/edit_viewer_overlay_geometry.hpp"
#include "ui/edit_viewer/edit_viewer_surface.hpp"
#include "ui/edit_viewer/frame_sink.hpp"
#include "ui/edit_viewer/view_transform_controller.hpp"
#include "ui/edit_viewer/viewer_state.hpp"
#include "ui/edit_viewer/viewport_mapper.hpp"

#ifdef HAVE_CUDA
#include "ui/edit_viewer/frame_mailbox.hpp"
#endif

class QMouseEvent;
class QWheelEvent;

namespace puerhlab {

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
  void    SubmitHostFrame(const ViewerFrame& frame) override;
  int     GetWidth() const override;
  int     GetHeight() const override;
  auto    GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> override;
  void    SetNextFramePresentationMode(FramePresentationMode mode) override;
  auto    GetViewerSurface() -> IEditViewerSurface* override;
  auto    GetViewerSurface() const -> const IEditViewerSurface* override;

  void    SetHistogramFrameExpected(bool expected_fast_preview);
  void    SetHistogramUpdateIntervalMs(int interval_ms);

 signals:
  void RequestUpdate();

  void RequestResize(int width, int height);
  void HistogramDataUpdated();
  void CropOverlayRectChanged(float x, float y, float w, float h, bool is_final);
  void CropOverlayRotationChanged(float angle_degrees, bool is_final);
  void ViewZoomChanged(float zoom);

 private slots:
  void OnResizeSurface(int w, int h);

 protected:
  void resizeEvent(QResizeEvent* event) override;

 private:
  friend class EditViewerOverlayWidget;

  void                    HandleQueuedUpdate();
  void                    ResizeChildWidgets();
  void                    UpdateViewportRenderRegionCache();
  void                    RefreshFrameDerivedState();
  void                    SyncSurfaceState();
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
  void                    PaintOverlay(QWidget& widget);
  void                    HandleOverlayWheel(QWheelEvent* event);
  void                    HandleOverlayMousePress(QMouseEvent* event);
  void                    HandleOverlayMouseMove(QMouseEvent* event);
  void                    HandleOverlayMouseRelease(QMouseEvent* event);
  void                    HandleOverlayMouseDoubleClick(QMouseEvent* event);
  void                    HandleOverlayLeave();

  static constexpr int     kZoomAnimationDurationMs = 170;

  ViewerState              viewer_state_{};
  ViewTransformController  view_transform_controller_{};
  CropInteractionController crop_interaction_controller_{};
  std::unique_ptr<IEditViewerSurface> surface_{};
  IEditViewerRenderTargetSurface* render_target_surface_ = nullptr;
  EditViewerOverlayWidget* overlay_ = nullptr;
  QVariantAnimation*       zoom_animation_ = nullptr;
  QTimer*                  click_toggle_timer_ = nullptr;

#ifdef HAVE_CUDA
  FrameMailbox             frame_mailbox_{};
#endif

  ViewerFrame              active_host_frame_{};
  std::optional<ViewerFrame> pending_host_frame_{};
  bool                     pending_presentation_mode_valid_ = false;
  FramePresentationMode    pending_presentation_mode_ = FramePresentationMode::FullFrame;
  mutable std::mutex       host_frame_mutex_{};
};
};  // namespace puerhlab
