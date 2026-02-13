//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

#include <qopenglshaderprogram.h>
#include <qopenglwidget.h>

#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QTimer>
#include <QVariantAnimation>
#include <QWheelEvent>
#include <QRectF>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <cuda_gl_interop.h>

#include "frame_sink.hpp"

namespace puerhlab {

class QtEditViewer : public QOpenGLWidget, protected QOpenGLExtraFunctions, public puerhlab::IFrameSink {
  Q_OBJECT
 public:
  explicit QtEditViewer(QWidget* parent = nullptr);
  ~QtEditViewer();

  // Reset zoom/pan to default view
  void ResetView();
  void SetCropToolEnabled(bool enabled);
  void SetCropOverlayVisible(bool visible);
  void SetCropOverlayRectNormalized(float x, float y, float w, float h);
  void SetCropOverlayRotationDegrees(float angle_degrees);
  void ResetCropOverlayRectToFull();
  auto GetViewZoom() const -> float;

  // Overrides from IFrameSink
  void    EnsureSize(int width, int height) override;
  float4* MapResourceForWrite() override;
  void    UnmapResource() override;
  void    NotifyFrameReady() override;
  int     GetWidth() const override { return buffers_[active_idx_].width; };
  int     GetHeight() const override { return buffers_[active_idx_].height; };
  auto    GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> override;
  void    SetNextFramePresentationMode(FramePresentationMode mode) override;

  void    SetHistogramFrameExpected(bool expected_fast_preview);
  void    SetHistogramUpdateIntervalMs(int interval_ms);

  auto    GetHistogramBufferId() const -> GLuint;
  auto    GetHistogramBinCount() const -> int;
  auto    HasHistogramData() const -> bool;

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
  // Overrides from QOpenGLWidget
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;
  void wheelEvent(QWheelEvent* event) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void mouseDoubleClickEvent(QMouseEvent* event) override;

 private:
  struct GLBuffer {
    GLuint                texture       = 0;
    cudaGraphicsResource* cuda_resource = nullptr;
    int                   width         = 0;
    int                   height        = 0;
  };

  std::array<GLBuffer, 2> buffers_{};

  GLuint                  vbo_               = 0;
  QOpenGLShaderProgram*   program_           = nullptr;

  int                     active_idx_        = 0;  // Currently displayed buffer
  int                     write_idx_         = 1;  // Buffer to render next frame into
  int                     render_target_idx_ = 0;  // Which buffer next GPU frame should use

  // CUDA staging buffer (written by worker thread). We only map the PBO in paintGL
  // on the GUI thread to avoid "invalid OpenGL or DirectX context".
  float4*                 staging_ptr_     = nullptr;
  size_t                  staging_bytes_   = 0;
  std::atomic<int>        pending_frame_idx_{-1};

  // Thread synchronization
  std::mutex              mutex_;

  // Interaction state (UI thread only)
  float                   view_zoom_    = 1.0f;
  QVector2D               view_pan_     = {0.0f, 0.0f};
  QPoint                  last_mouse_pos_{};
  bool                    dragging_     = false;
  enum class CropDragMode { None, Create, Move, ResizeEdge, RotateCorner };
  enum class CropCorner { None, TopLeft, TopRight, BottomRight, BottomLeft };
  enum class CropEdge { None, Top, Right, Bottom, Left };
  bool                    crop_tool_enabled_   = false;
  bool                    crop_overlay_visible_ = false;
  QRectF                  crop_overlay_rect_    = QRectF(0.0, 0.0, 1.0, 1.0);
  float                   crop_overlay_rotation_degrees_ = 0.0f;
  float                   crop_overlay_metric_aspect_ = 1.0f;
  CropDragMode            crop_drag_mode_       = CropDragMode::None;
  CropCorner              crop_drag_corner_     = CropCorner::None;
  CropEdge                crop_drag_edge_       = CropEdge::None;
  QPointF                 crop_drag_anchor_uv_{};
  QPointF                 crop_drag_anchor_widget_pos_{};
  QRectF                  crop_drag_origin_rect_{};
  QPointF                 crop_drag_fixed_corner_uv_{};
  float                   crop_drag_rotation_degrees_ = 0.0f;
  mutable std::mutex      view_state_mutex_;
  std::optional<ViewportRenderRegion> viewport_render_region_cache_{};
  int                     render_reference_width_  = 0;
  int                     render_reference_height_ = 0;
  std::atomic<FramePresentationMode>  active_frame_presentation_mode_{
      FramePresentationMode::ViewportTransformed};
  std::atomic<FramePresentationMode>  pending_frame_presentation_mode_{
      FramePresentationMode::ViewportTransformed};
  std::atomic<bool>       pending_presentation_mode_valid_{false};

  bool                    InitBuffer(GLBuffer& buffer, int width, int height);
  void                    FreeBuffer(GLBuffer& buffer);
  void                    FreeAllBuffers();

  bool                    InitHistogramResources();
  void                    FreeHistogramResources();
  void                    UpdateViewportRenderRegionCache();
  auto                    ComputeViewportRenderRegion(int image_width, int image_height) const
      -> std::optional<ViewportRenderRegion>;
  auto                    WidgetPointToImageUv(const QPointF& widget_pos, int image_width,
                                               int image_height) const -> std::optional<QPointF>;
  auto                    ImageUvToWidgetPoint(const QPointF& uv, int image_width, int image_height) const
      -> std::optional<QPointF>;
  static auto             ClampCropRect(const QRectF& rect) -> QRectF;
  auto                    ComputeHistogram(GLuint texture_id, int width, int height) -> bool;
  auto                    ShouldComputeHistogramNow() -> bool;
  auto                    BuildComputeProgram(const char* source, const char* debug_name,
                                              GLuint& out_program) -> bool;
  void                    StopZoomAnimation();
  void                    ApplyViewTransform(float zoom, const QVector2D& pan, bool emit_zoom_signal);
  void                    AnimateViewTo(float target_zoom,
                                        const std::optional<QPointF>& anchor_widget_pos,
                                        const std::optional<QVector2D>& explicit_target_pan = std::nullopt);
  auto                    ClampPanForZoom(float zoom, const QVector2D& pan) const -> QVector2D;
  auto                    ComputeAnchoredPan(float target_zoom, const QPointF& anchor_widget_pos,
                                             const QVector2D& fallback_pan) const -> QVector2D;
  void                    ToggleClickZoomAt(const QPointF& anchor_widget_pos);
  void                    ToggleDoubleClickZoomAt(const QPointF& anchor_widget_pos);

  static constexpr int    kHistogramBins       = 256;
  static constexpr int    kHistogramSampleSize = 256;

  GLuint                  histogram_count_ssbo_       = 0;
  GLuint                  histogram_norm_ssbo_        = 0;
  GLuint                  histogram_clear_program_    = 0;
  GLuint                  histogram_compute_program_  = 0;
  GLuint                  histogram_normalize_program_ = 0;
  GLint                   histogram_clear_count_loc_  = -1;
  GLint                   histogram_compute_tex_loc_  = -1;
  GLint                   histogram_compute_bins_loc_ = -1;
  GLint                   histogram_compute_sample_loc_ = -1;
  GLint                   histogram_norm_bins_loc_    = -1;
  bool                    histogram_resources_ready_   = false;
  std::atomic<bool>       histogram_expect_fast_frame_{false};
  std::atomic<bool>       histogram_pending_frame_{false};
  std::atomic<bool>       histogram_has_data_{false};
  int                     histogram_update_interval_ms_ = 40;
  std::chrono::steady_clock::time_point last_histogram_update_time_{};

  static constexpr float   kMinInteractiveZoom       = 1.0f;
  static constexpr float   kMaxInteractiveZoom       = 8.0f;
  static constexpr float   kWheelZoomStep            = 1.12f;
  static constexpr float   kSingleClickZoomFactor    = 2.0f;
  static constexpr int     kZoomAnimationDurationMs  = 170;
  static constexpr int     kClickDragThresholdPixels = 3;

  QVariantAnimation*       zoom_animation_           = nullptr;
  float                    zoom_animation_start_     = 1.0f;
  float                    zoom_animation_target_    = 1.0f;
  QVector2D                pan_animation_start_      = {0.0f, 0.0f};
  QVector2D                pan_animation_target_     = {0.0f, 0.0f};
  QPoint                   drag_start_mouse_pos_{};
  bool                     dragged_since_press_      = false;
  bool                     click_zoom_toggle_active_ = false;
  float                    click_zoom_restore_zoom_  = 1.0f;
  QVector2D                click_zoom_restore_pan_   = {0.0f, 0.0f};
  float                    double_click_zoom_target_ = kSingleClickZoomFactor;
  bool                     double_click_zoom_in_next_ = true;
  QTimer*                  click_toggle_timer_       = nullptr;
  QPointF                  pending_click_toggle_pos_{};
  bool                     pending_click_toggle_      = false;
  bool                     suppress_next_click_release_toggle_ = false;
};
};  // namespace puerhlab
