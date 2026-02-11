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
#include <QWheelEvent>
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

  // Overrides from IFrameSink
  void    EnsureSize(int width, int height) override;
  float4* MapResourceForWrite() override;
  void    UnmapResource() override;
  void    NotifyFrameReady() override;
  int     GetWidth() const override { return buffers_[active_idx_].width; };
  int     GetHeight() const override { return buffers_[active_idx_].height; };

  void    SetHistogramFrameExpected(bool expected_fast_preview);
  void    SetHistogramUpdateIntervalMs(int interval_ms);

  auto    GetHistogramBufferId() const -> GLuint;
  auto    GetHistogramBinCount() const -> int;
  auto    HasHistogramData() const -> bool;

 signals:
  void RequestUpdate();

  void RequestResize(int width, int height);
  void HistogramDataUpdated();

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
    GLuint                pbo           = 0;
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

  bool                    InitBuffer(GLBuffer& buffer, int width, int height);
  void                    FreeBuffer(GLBuffer& buffer);
  void                    FreeAllBuffers();

  bool                    InitHistogramResources();
  void                    FreeHistogramResources();
  auto                    ComputeHistogram(GLuint texture_id, int width, int height) -> bool;
  auto                    ShouldComputeHistogramNow() -> bool;
  auto                    BuildComputeProgram(const char* source, const char* debug_name,
                                              GLuint& out_program) -> bool;

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
};
};  // namespace puerhlab
