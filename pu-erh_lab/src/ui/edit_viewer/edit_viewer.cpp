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

#include "ui/edit_viewer/edit_viewer.hpp"

#include <GL/gl.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <qopenglext.h>
#include <qoverload.h>
#include <QSurfaceFormat>
#include <QByteArray>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <vector>

namespace puerhlab {

static const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec2 position;

uniform vec2 uScale;
uniform vec2 uPan;
uniform float uZoom;

out vec2 vTexCoord;

void main() {
  // Letterbox scale, then user zoom/pan for interactive view controls
  vec2 pos = position * uScale * uZoom + uPan;
  gl_Position = vec4(pos, 0.0, 1.0);

  vec2 uv = (position + 1.0) * 0.5;
  vTexCoord = vec2(uv.x, 1.0 - uv.y); // flip Y
}
)";

static const char* fragmentShaderSource = R"(
#version 330 core
uniform sampler2D textureSampler;
in vec2 vTexCoord;
out vec4 FragColor;
void main() {
    FragColor = texture(textureSampler, vTexCoord);
}
)";

static const char* histogramClearShaderSource = R"(
#version 430 core
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer HistogramCounts {
  uint counts[];
};

uniform int uCount;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx < uint(uCount)) {
    counts[idx] = 0u;
  }
}
)";

static const char* histogramComputeShaderSource = R"(
#version 430 core
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D uSourceTex;

layout(std430, binding = 0) buffer HistogramCounts {
  uint counts[];
};

uniform int uBins;
uniform int uSampleSize;

void main() {
  uvec2 gid = gl_GlobalInvocationID.xy;
  if (gid.x >= uint(uSampleSize) || gid.y >= uint(uSampleSize)) {
    return;
  }

  vec2 uv = (vec2(gid) + vec2(0.5)) / float(uSampleSize);
  vec3 rgb = clamp(textureLod(uSourceTex, uv, 0.0).rgb, 0.0, 1.0);

  int r = int(rgb.r * float(uBins - 1) + 0.5);
  int g = int(rgb.g * float(uBins - 1) + 0.5);
  int b = int(rgb.b * float(uBins - 1) + 0.5);

  atomicAdd(counts[r], 1u);
  atomicAdd(counts[uBins + g], 1u);
  atomicAdd(counts[uBins * 2 + b], 1u);
}
)";

static const char* histogramNormalizeShaderSource = R"(
#version 430 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer HistogramCounts {
  uint counts[];
};

layout(std430, binding = 1) writeonly buffer HistogramNormalized {
  float normalized[];
};

uniform int uBins;

void main() {
  const int count = uBins * 3;
  uint max_count = 1u;
  for (int i = 0; i < count; ++i) {
    max_count = max(max_count, counts[i]);
  }

  const float inv = 1.0 / float(max_count);
  for (int i = 0; i < count; ++i) {
    normalized[i] = float(counts[i]) * inv;
  }
}
)";

QtEditViewer::QtEditViewer(QWidget* parent) : QOpenGLWidget(parent) {
  // Connect the frame ready signal to the update slot.
  // Use QueuedConnection explicitly so that signals from worker threads are
  // processed on the next event-loop iteration of the GUI thread.
  connect(this, &QtEditViewer::RequestUpdate, this, QOverload<>::of(&QtEditViewer::update),
          Qt::QueuedConnection);

  // Blocking resize requests until the current resize is done
  connect(this, &QtEditViewer::RequestResize, this, &QtEditViewer::OnResizeGL,
          Qt::BlockingQueuedConnection);
}

QtEditViewer::~QtEditViewer() {
  makeCurrent();
  // Clean up OpenGL resources
  FreeAllBuffers();
  FreeHistogramResources();
  delete program_;
  program_ = nullptr;
  doneCurrent();

  if (staging_ptr_) {
    cudaFree(staging_ptr_);
    staging_ptr_   = nullptr;
    staging_bytes_ = 0;
  }
}

void QtEditViewer::EnsureSize(int width, int height) {
  bool need_resize = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& target_buf = buffers_[render_target_idx_];
    if (target_buf.width != width || target_buf.height != height) {
      // Prepare the alternate buffer for the new size without dropping the currently shown one.
      render_target_idx_ = write_idx_;
      need_resize        = true;
    }
  }

  // Emit outside the mutex to avoid deadlock with the UI thread (slot locks the same mutex).
  if (need_resize) {
    emit RequestResize(width, height);
  }

  // Ensure staging buffer is available for worker thread writes.
  const size_t needed_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float4);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (needed_bytes > 0 && needed_bytes != staging_bytes_) {
      if (staging_ptr_) {
        cudaFree(staging_ptr_);
        staging_ptr_ = nullptr;
      }
      const cudaError_t alloc_err = cudaMalloc(reinterpret_cast<void**>(&staging_ptr_), needed_bytes);
      if (alloc_err != cudaSuccess) {
        qWarning("Failed to allocate CUDA staging buffer (%zu bytes): %s", needed_bytes,
                 cudaGetErrorString(alloc_err));
        staging_ptr_   = nullptr;
        staging_bytes_ = 0;
      } else {
        staging_bytes_ = needed_bytes;
      }
    }
  }

  // Resize request is emitted in the size-change branch above.
}

void QtEditViewer::ResetView() {
  view_zoom_ = 1.0f;
  view_pan_  = QVector2D(0.0f, 0.0f);
  update();
}

void QtEditViewer::SetHistogramFrameExpected(bool expected_fast_preview) {
  histogram_expect_fast_frame_.store(expected_fast_preview, std::memory_order_release);
}

void QtEditViewer::SetHistogramUpdateIntervalMs(int interval_ms) {
  histogram_update_interval_ms_ = std::max(0, interval_ms);
}

auto QtEditViewer::GetHistogramBufferId() const -> GLuint {
  if (!histogram_resources_ready_) {
    return 0;
  }
  return histogram_norm_ssbo_;
}

auto QtEditViewer::GetHistogramBinCount() const -> int { return kHistogramBins; }

auto QtEditViewer::HasHistogramData() const -> bool {
  return histogram_has_data_.load(std::memory_order_acquire);
}

float4* QtEditViewer::MapResourceForWrite() {
  mutex_.lock();

  // IMPORTANT: Do NOT map the OpenGL PBO from this thread. On Windows this often
  // fails with "invalid OpenGL or DirectX context" because the GL context is
  // owned by the GUI thread.
  if (!staging_ptr_ || staging_bytes_ == 0) {
    mutex_.unlock();
    return nullptr;
  }

  return staging_ptr_;
}

void QtEditViewer::UnmapResource() {
  // Mark which buffer should receive the pending frame.
  pending_frame_idx_.store(render_target_idx_, std::memory_order_release);
  mutex_.unlock();
}

void QtEditViewer::NotifyFrameReady() {
  histogram_pending_frame_.store(histogram_expect_fast_frame_.load(std::memory_order_acquire),
                                 std::memory_order_release);
  // Wake the UI thread to update the display
  emit RequestUpdate();
}

void QtEditViewer::initializeGL() {
  initializeOpenGLFunctions();

  program_ = new QOpenGLShaderProgram();
  if (!program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource)) {
    qWarning("Vertex shader compile failed: %s", program_->log().toUtf8().constData());
  }
  if (!program_->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource)) {
    qWarning("Fragment shader compile failed: %s", program_->log().toUtf8().constData());
  }
  if (!program_->link()) {
    qWarning("Shader program link failed: %s", program_->log().toUtf8().constData());
  }

  // Static full-screen quad (never modified)
  float vertices[] = {
      -1.0f, -1.0f,
       1.0f, -1.0f,
      -1.0f,  1.0f,
       1.0f,  1.0f,
  };
  glGenBuffers(1, &vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffers_[active_idx_].width <= 0 || buffers_[active_idx_].height <= 0) {
      buffers_[active_idx_].width  = std::max(1, this->width());
      buffers_[active_idx_].height = std::max(1, this->height());
    }
  }

  InitBuffer(buffers_[active_idx_], buffers_[active_idx_].width, buffers_[active_idx_].height);
  InitHistogramResources();
}

bool QtEditViewer::InitBuffer(GLBuffer& buffer, int width, int height) {
  if (width <= 0 || height <= 0) {
    qWarning("InitBuffer skipped: invalid size %dx%d", width, height);
    return false;
  }

  FreeBuffer(buffer);

  glGenBuffers(1, &buffer.pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer.pbo);

  size_t size = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float4);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_COPY);

  glGenTextures(1, &buffer.texture);
  glBindTexture(GL_TEXTURE_2D, buffer.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

  cudaError_t err = cudaGraphicsGLRegisterBuffer(&buffer.cuda_resource, buffer.pbo,
                                                 cudaGraphicsMapFlagsWriteDiscard);
  if (err != cudaSuccess) {
    qWarning("Failed to register PBO with CUDA: %s", cudaGetErrorString(err));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    FreeBuffer(buffer);
    return false;
  }

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  buffer.width  = width;
  buffer.height = height;
  return true;
}

auto QtEditViewer::BuildComputeProgram(const char* source, const char* debug_name,
                                       GLuint& out_program) -> bool {
  if (!source) {
    return false;
  }

  if (out_program != 0) {
    glDeleteProgram(out_program);
    out_program = 0;
  }

  const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint compile_ok = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_ok);
  if (compile_ok != GL_TRUE) {
    GLint log_len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
    std::vector<char> log(static_cast<size_t>(std::max(0, log_len)) + 1, '\0');
    if (log_len > 0) {
      glGetShaderInfoLog(shader, log_len, nullptr, log.data());
    }
    qWarning("%s compute shader compile failed: %s", debug_name, log.data());
    glDeleteShader(shader);
    return false;
  }

  const GLuint program = glCreateProgram();
  glAttachShader(program, shader);
  glLinkProgram(program);
  glDeleteShader(shader);

  GLint link_ok = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
  if (link_ok != GL_TRUE) {
    GLint log_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
    std::vector<char> log(static_cast<size_t>(std::max(0, log_len)) + 1, '\0');
    if (log_len > 0) {
      glGetProgramInfoLog(program, log_len, nullptr, log.data());
    }
    qWarning("%s compute program link failed: %s", debug_name, log.data());
    glDeleteProgram(program);
    return false;
  }

  out_program = program;
  return true;
}

bool QtEditViewer::InitHistogramResources() {
  if (histogram_resources_ready_) {
    return true;
  }

  auto* gl_context = context();
  if (!gl_context) {
    return false;
  }

  const QSurfaceFormat format = gl_context->format();
  const bool has_compute_support =
      (format.majorVersion() > 4 || (format.majorVersion() == 4 && format.minorVersion() >= 3)) ||
      gl_context->hasExtension(QByteArrayLiteral("GL_ARB_compute_shader"));
  if (!has_compute_support) {
    qWarning("QtEditViewer histogram disabled: OpenGL compute shaders are not supported.");
    return false;
  }

  if (!BuildComputeProgram(histogramClearShaderSource, "HistogramClear",
                           histogram_clear_program_)) {
    FreeHistogramResources();
    return false;
  }
  if (!BuildComputeProgram(histogramComputeShaderSource, "HistogramCompute",
                           histogram_compute_program_)) {
    FreeHistogramResources();
    return false;
  }
  if (!BuildComputeProgram(histogramNormalizeShaderSource, "HistogramNormalize",
                           histogram_normalize_program_)) {
    FreeHistogramResources();
    return false;
  }

  const GLsizeiptr count_bytes = static_cast<GLsizeiptr>(sizeof(GLuint) * kHistogramBins * 3);
  const GLsizeiptr norm_bytes  = static_cast<GLsizeiptr>(sizeof(float) * kHistogramBins * 3);

  glGenBuffers(1, &histogram_count_ssbo_);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, histogram_count_ssbo_);
  glBufferData(GL_SHADER_STORAGE_BUFFER, count_bytes, nullptr, GL_DYNAMIC_DRAW);

  glGenBuffers(1, &histogram_norm_ssbo_);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, histogram_norm_ssbo_);
  glBufferData(GL_SHADER_STORAGE_BUFFER, norm_bytes, nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  histogram_clear_count_loc_    = glGetUniformLocation(histogram_clear_program_, "uCount");
  histogram_compute_tex_loc_    = glGetUniformLocation(histogram_compute_program_, "uSourceTex");
  histogram_compute_bins_loc_   = glGetUniformLocation(histogram_compute_program_, "uBins");
  histogram_compute_sample_loc_ = glGetUniformLocation(histogram_compute_program_, "uSampleSize");
  histogram_norm_bins_loc_      = glGetUniformLocation(histogram_normalize_program_, "uBins");

  histogram_resources_ready_ = histogram_count_ssbo_ != 0 && histogram_norm_ssbo_ != 0;
  histogram_has_data_.store(false, std::memory_order_release);
  last_histogram_update_time_ = {};
  return histogram_resources_ready_;
}

void QtEditViewer::FreeHistogramResources() {
  if (histogram_count_ssbo_) {
    glDeleteBuffers(1, &histogram_count_ssbo_);
    histogram_count_ssbo_ = 0;
  }
  if (histogram_norm_ssbo_) {
    glDeleteBuffers(1, &histogram_norm_ssbo_);
    histogram_norm_ssbo_ = 0;
  }
  if (histogram_clear_program_) {
    glDeleteProgram(histogram_clear_program_);
    histogram_clear_program_ = 0;
  }
  if (histogram_compute_program_) {
    glDeleteProgram(histogram_compute_program_);
    histogram_compute_program_ = 0;
  }
  if (histogram_normalize_program_) {
    glDeleteProgram(histogram_normalize_program_);
    histogram_normalize_program_ = 0;
  }

  histogram_clear_count_loc_    = -1;
  histogram_compute_tex_loc_     = -1;
  histogram_compute_bins_loc_    = -1;
  histogram_compute_sample_loc_  = -1;
  histogram_norm_bins_loc_       = -1;
  histogram_resources_ready_     = false;
  histogram_has_data_.store(false, std::memory_order_release);
  histogram_pending_frame_.store(false, std::memory_order_release);
}

auto QtEditViewer::ShouldComputeHistogramNow() -> bool {
  if (histogram_update_interval_ms_ <= 0) {
    last_histogram_update_time_ = std::chrono::steady_clock::now();
    return true;
  }

  const auto now = std::chrono::steady_clock::now();
  if (last_histogram_update_time_.time_since_epoch().count() == 0) {
    last_histogram_update_time_ = now;
    return true;
  }

  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_histogram_update_time_)
          .count();
  if (elapsed_ms < histogram_update_interval_ms_) {
    return false;
  }
  last_histogram_update_time_ = now;
  return true;
}

auto QtEditViewer::ComputeHistogram(GLuint texture_id, int width, int height) -> bool {
  if (!histogram_resources_ready_ || texture_id == 0 || width <= 0 || height <= 0) {
    return false;
  }

  const int histogram_values = kHistogramBins * 3;
  glUseProgram(histogram_clear_program_);
  if (histogram_clear_count_loc_ >= 0) {
    glUniform1i(histogram_clear_count_loc_, histogram_values);
  }
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, histogram_count_ssbo_);
  const GLuint clear_groups = static_cast<GLuint>((histogram_values + 64 - 1) / 64);
  glDispatchCompute(clear_groups, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  glUseProgram(histogram_compute_program_);
  if (histogram_compute_tex_loc_ >= 0) {
    glUniform1i(histogram_compute_tex_loc_, 0);
  }
  if (histogram_compute_bins_loc_ >= 0) {
    glUniform1i(histogram_compute_bins_loc_, kHistogramBins);
  }
  if (histogram_compute_sample_loc_ >= 0) {
    glUniform1i(histogram_compute_sample_loc_, kHistogramSampleSize);
  }

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, histogram_count_ssbo_);

  const GLuint groups =
      static_cast<GLuint>((kHistogramSampleSize + 16 - 1) / 16);
  glDispatchCompute(groups, groups, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  glUseProgram(histogram_normalize_program_);
  if (histogram_norm_bins_loc_ >= 0) {
    glUniform1i(histogram_norm_bins_loc_, kHistogramBins);
  }
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, histogram_count_ssbo_);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, histogram_norm_ssbo_);
  glDispatchCompute(1, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  glUseProgram(0);
  glFlush();

  return true;
}

void QtEditViewer::paintGL() {
  std::lock_guard<std::mutex> lock(mutex_);

  // First, check if there is a pending frame to copy from staging buffer.
  // This must happen BEFORE we check the active buffer validity, because the
  // pending frame might switch active_idx_ to a newly initialized buffer.
  const int pending_idx = pending_frame_idx_.exchange(-1, std::memory_order_acq_rel);
  GLBuffer* target_buffer = nullptr;
  if (pending_idx >= 0 && pending_idx < static_cast<int>(buffers_.size())) {
    target_buffer = &buffers_[pending_idx];
  }

  if (target_buffer && target_buffer->cuda_resource && staging_ptr_ && staging_bytes_ > 0) {
    cudaError_t map_err = cudaGraphicsMapResources(1, &target_buffer->cuda_resource, 0);
    if (map_err != cudaSuccess) {
      qWarning("Failed to map CUDA resource (paintGL): %s", cudaGetErrorString(map_err));
    } else {
      float4* mapped_ptr = nullptr;
      size_t  mapped_bytes = 0;
      cudaError_t ptr_err = cudaGraphicsResourceGetMappedPointer(
          reinterpret_cast<void**>(&mapped_ptr), &mapped_bytes, target_buffer->cuda_resource);
      if (ptr_err != cudaSuccess || !mapped_ptr || mapped_bytes == 0) {
        qWarning("Failed to get mapped pointer (paintGL): %s", cudaGetErrorString(ptr_err));
      } else {
        const size_t copy_bytes = std::min(staging_bytes_, mapped_bytes);
        cudaError_t copy_err = cudaMemcpy(mapped_ptr, staging_ptr_, copy_bytes, cudaMemcpyDeviceToDevice);
        if (copy_err != cudaSuccess) {
          qWarning("Failed to copy staging->PBO: %s", cudaGetErrorString(copy_err));
        } else {
          active_idx_ = pending_idx;
          write_idx_  = 1 - active_idx_;
        }
      }

      cudaError_t unmap_err = cudaGraphicsUnmapResources(1, &target_buffer->cuda_resource, 0);
      if (unmap_err != cudaSuccess) {
        qWarning("Failed to unmap CUDA resource (paintGL): %s", cudaGetErrorString(unmap_err));
      }
    }
  }

  // Now check if the active buffer is valid for rendering.
  GLBuffer& active_buffer = buffers_[active_idx_];
  if (!active_buffer.pbo || !active_buffer.texture || !program_ || !program_->isLinked()) return;

  const float dpr = devicePixelRatioF();
  const float vw  = std::max(1.0f, float(width()) * dpr);
  const float vh  = std::max(1.0f, float(height()) * dpr);
  glViewport(0, 0, int(vw), int(vh));

  // Compute letterbox scale from IMAGE aspect vs WINDOW aspect
  const float imgW = float(std::max(1, active_buffer.width));
  const float imgH = float(std::max(1, active_buffer.height));
  const float winAspect = vw / vh;
  const float imgAspect = imgW / imgH;

  float sx = 1.0f, sy = 1.0f;
  if (imgAspect > winAspect) {
    sy = winAspect / imgAspect; // image wider -> reduce Y
  } else {
    sx = imgAspect / winAspect; // image taller -> reduce X
  }

  const float zoom = view_zoom_;
  const QVector2D pan = view_pan_;

  glClearColor(0.f, 0.f, 0.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);

  program_->bind();
  program_->setUniformValue("uScale", QVector2D(sx, sy));
  program_->setUniformValue("uPan", pan);
  program_->setUniformValue("uZoom", zoom);

  glActiveTexture(GL_TEXTURE0);
  program_->setUniformValue("textureSampler", 0);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, active_buffer.pbo);
  glBindTexture(GL_TEXTURE_2D, active_buffer.texture);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, active_buffer.width, active_buffer.height, GL_RGBA, GL_FLOAT, nullptr);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  int pos_loc = program_->attributeLocation("position");
  program_->enableAttributeArray(pos_loc);
  program_->setAttributeBuffer(pos_loc, GL_FLOAT, 0, 2, 2 * sizeof(float));

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  program_->release();

  const bool histogram_requested = histogram_pending_frame_.exchange(false, std::memory_order_acq_rel);
  if (histogram_requested && ShouldComputeHistogramNow()) {
    if (histogram_resources_ready_ || InitHistogramResources()) {
      if (ComputeHistogram(active_buffer.texture, active_buffer.width, active_buffer.height)) {
        histogram_has_data_.store(true, std::memory_order_release);
        emit HistogramDataUpdated();
      }
    }
  }
}

void QtEditViewer::resizeGL(int w, int h) {
  if (w <= 0 || h <= 0) return;

}

void QtEditViewer::FreeBuffer(GLBuffer& buffer) {
  if (buffer.cuda_resource) {
    cudaGraphicsUnregisterResource(buffer.cuda_resource);
    buffer.cuda_resource = nullptr;
  }
  if (buffer.pbo) {
    glDeleteBuffers(1, &buffer.pbo);
    buffer.pbo = 0;
  }
  if (buffer.texture) {
    glDeleteTextures(1, &buffer.texture);
    buffer.texture = 0;
  }
  buffer.width  = 0;
  buffer.height = 0;
}

void QtEditViewer::FreeAllBuffers() {
  for (auto& buffer : buffers_) {
    FreeBuffer(buffer);
  }
}

void QtEditViewer::OnResizeGL(int w, int h) {
  makeCurrent();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    GLBuffer& target = buffers_[write_idx_];
    FreeBuffer(target);
    if (InitBuffer(target, w, h)) {
      render_target_idx_ = write_idx_;
    }
  }
  // resizeGL(w, h);
  doneCurrent();

  // Ensure a repaint after resize
  update();
}

void QtEditViewer::wheelEvent(QWheelEvent* event) {
  const QPoint num_degrees = event->angleDelta();
  if (!num_degrees.isNull()) {
    const float steps   = static_cast<float>(num_degrees.y()) / 120.0f;  // 120 units per notch
    const float factor  = std::pow(1.1f, steps);
    const float oldZoom = view_zoom_;
    view_zoom_          = std::clamp(view_zoom_ * factor, 0.1f, 20.0f);

    // Optional: adjust pan so zoom stays centered on view
    if (std::abs(view_zoom_ - oldZoom) > 1e-4f) {
      // Zoom towards cursor by nudging pan in normalized device space
      const float dpr = devicePixelRatioF();
      const float vw  = std::max(1.0f, float(width()) * dpr);
      const float vh  = std::max(1.0f, float(height()) * dpr);
      const QPointF p = event->position();
      const float ndcX = (2.0f * float(p.x()) / vw) - 1.0f;
      const float ndcY = 1.0f - (2.0f * float(p.y()) / vh);
      const float zoomDelta = view_zoom_ - oldZoom;
      const float adjust    = (zoomDelta / std::max(view_zoom_, 1e-4f));
      view_pan_ -= QVector2D(ndcX, ndcY) * adjust * 0.5f;
    }

    update();
    event->accept();
    return;
  }
  QOpenGLWidget::wheelEvent(event);
}

void QtEditViewer::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
    dragging_      = true;
    last_mouse_pos_ = event->pos();
    setCursor(Qt::ClosedHandCursor);
    event->accept();
    return;
  }
  QOpenGLWidget::mousePressEvent(event);
}

void QtEditViewer::mouseMoveEvent(QMouseEvent* event) {
  if (dragging_) {
    const QPoint delta = event->pos() - last_mouse_pos_;
    last_mouse_pos_    = event->pos();

    const float dpr = devicePixelRatioF();
    const float vw  = std::max(1.0f, float(width()) * dpr);
    const float vh  = std::max(1.0f, float(height()) * dpr);

    // Convert pixel delta to normalized device coordinates
    QVector2D ndc_delta(2.0f * float(delta.x()) / vw, -2.0f * float(delta.y()) / vh);
    view_pan_ += ndc_delta;
    update();
    event->accept();
    return;
  }
  QOpenGLWidget::mouseMoveEvent(event);
}

void QtEditViewer::mouseReleaseEvent(QMouseEvent* event) {
  if (dragging_ && (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton)) {
    dragging_ = false;
    unsetCursor();
    event->accept();
    return;
  }
  QOpenGLWidget::mouseReleaseEvent(event);
}

void QtEditViewer::mouseDoubleClickEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    ResetView();
    event->accept();
    return;
  }
  QOpenGLWidget::mouseDoubleClickEvent(event);
}
};  // namespace puerhlab
