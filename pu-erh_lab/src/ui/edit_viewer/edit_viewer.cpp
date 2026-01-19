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

#include <algorithm>
#include <cmath>
#include <mutex>

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
  delete program_;
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