//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <array>
#include <chrono>

#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>

#include <cuda_gl_interop.h>

#include "ui/edit_viewer/frame_mailbox.hpp"
#include "ui/edit_viewer/viewer_state.hpp"
#include "ui/edit_viewer/viewport_mapper.hpp"

namespace puerhlab {

class OpenGLViewerRenderer : protected QOpenGLExtraFunctions {
 public:
  struct RenderResult {
    bool histogram_data_updated = false;
  };

  OpenGLViewerRenderer() = default;
  ~OpenGLViewerRenderer() = default;

  void Initialize();
  void Shutdown();

  auto EnsureSlot(int slot_index, int width, int height) -> bool;
  auto HasSlot(int slot_index, int width, int height) const -> bool;
  auto UploadPendingFrame(const FrameMailbox::PendingFrame& pending_frame) -> bool;
  auto Render(QOpenGLWidget& widget, const FrameMailbox::ActiveFrame& active_frame,
              const ViewerStateSnapshot& state_snapshot, bool histogram_requested)
      -> RenderResult;

  void SetHistogramUpdateIntervalMs(int interval_ms);
  auto GetHistogramBufferId() const -> GLuint;
  auto GetHistogramBinCount() const -> int;
  auto HasHistogramData() const -> bool;

 private:
  struct GLBuffer {
    GLuint                texture       = 0;
    cudaGraphicsResource* cuda_resource = nullptr;
    int                   width         = 0;
    int                   height        = 0;
  };

  bool InitBuffer(GLBuffer& buffer, int width, int height);
  void FreeBuffer(GLBuffer& buffer);
  void FreeAllBuffers();
  bool InitHistogramResources();
  void FreeHistogramResources();
  auto ShouldComputeHistogramNow() -> bool;
  auto ComputeHistogram(GLuint texture_id, int width, int height) -> bool;
  auto BuildComputeProgram(const char* source, const char* debug_name, GLuint& out_program)
      -> bool;

  std::array<GLBuffer, 2> buffers_{};
  GLuint                  vbo_         = 0;
  QOpenGLShaderProgram*   program_     = nullptr;

  GLuint                  histogram_count_ssbo_         = 0;
  GLuint                  histogram_norm_ssbo_          = 0;
  GLuint                  histogram_clear_program_      = 0;
  GLuint                  histogram_compute_program_    = 0;
  GLuint                  histogram_normalize_program_  = 0;
  GLint                   histogram_clear_count_loc_    = -1;
  GLint                   histogram_compute_tex_loc_    = -1;
  GLint                   histogram_compute_bins_loc_   = -1;
  GLint                   histogram_compute_sample_loc_ = -1;
  GLint                   histogram_norm_bins_loc_      = -1;
  bool                    histogram_resources_ready_    = false;
  bool                    histogram_has_data_           = false;
  int                     histogram_update_interval_ms_ = 40;
  std::chrono::steady_clock::time_point last_histogram_update_time_{};

  static constexpr int kHistogramBins       = 256;
  static constexpr int kHistogramSampleSize = 256;
};

}  // namespace puerhlab
