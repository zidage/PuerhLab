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

#include <mutex>

namespace puerhlab {

static const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec2 position;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    vec2 uv = (position + 1.0) * 0.5;
    vTexCoord = vec2(uv.x, 1.0 - uv.y); // flip Y
}
)";

static const char* fragmentShaderSource = R"(
    uniform sampler2D textureSampler;
    varying vec2 vTexCoord;
    void main() {
        gl_FragColor = texture2D(textureSampler, vTexCoord);
    }
)";

QtEditViewer::QtEditViewer(QWidget* parent) : QOpenGLWidget(parent) {
  // Connect the frame ready signal to the update slot
  connect(this, &QtEditViewer::RequestUpdate, this, QOverload<>::of(&QtEditViewer::update));

  // Blocking resize requests until the current resize is done
  connect(this, &QtEditViewer::RequestResize, this, &QtEditViewer::OnResizeGL,
          Qt::BlockingQueuedConnection);
}

QtEditViewer::~QtEditViewer() {
  makeCurrent();
  // Clean up OpenGL resources
  FreeResources();
  delete program_;
  doneCurrent();

  if (staging_ptr_) {
    cudaFree(staging_ptr_);
    staging_ptr_   = nullptr;
    staging_bytes_ = 0;
  }
}

void QtEditViewer::EnsureSize(int width, int height) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (alloc_width_ == width && alloc_height_ == height) {
      return;
    }
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

  emit RequestResize(width, height);
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
  frame_pending_.store(true, std::memory_order_release);
  mutex_.unlock();
}

void QtEditViewer::NotifyFrameReady() {
  // Wake the UI thread to update the display
  emit RequestUpdate();
}

void QtEditViewer::initializeGL() {
  initializeOpenGLFunctions();

  // Initialize shaders
  program_ = new QOpenGLShaderProgram();
  program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  program_->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  program_->link();

  // Initialize PBO
  float vertices[] = {
      -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
  };
  glGenBuffers(1, &vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (alloc_width_ <= 0 || alloc_height_ <= 0) {
      alloc_width_  = std::max(1, this->width());
      alloc_height_ = std::max(1, this->height());
    }
  }

  InitPBO();
}

void QtEditViewer::InitPBO() {
  if (GetWidth() <= 0 || GetHeight() <= 0) {
    qWarning("InitPBO skipped: invalid size %dx%d", GetWidth(), GetHeight());
    return;
  }
  // Create PBO
  glGenBuffers(1, &pbo_);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);

  size_t size = GetWidth() * GetHeight() * sizeof(float4);
  // Allocate data for PBO
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_COPY);

  // Create OpenGL texture
  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  // Set texture parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // Allocate texture storage
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, GetWidth(), GetHeight(), 0, GL_RGBA, GL_FLOAT,
               nullptr);

  // Register PBO with CUDA
  cudaError_t err =
      cudaGraphicsGLRegisterBuffer(&cuda_resource_, pbo_, cudaGraphicsMapFlagsWriteDiscard);

  if (err != cudaSuccess) {
    qWarning("Failed to register PBO with CUDA: %s", cudaGetErrorString(err));
  }

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void QtEditViewer::paintGL() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pbo_ || !texture_ || !program_) return;

  const float dpr = devicePixelRatioF();
  const float vw  = std::max(1.0f, float(width()) * dpr);
  const float vh  = std::max(1.0f, float(height()) * dpr);
  glViewport(0, 0, int(vw), int(vh));

  // Keep aspect ratio (letterbox)
  const float imgW = std::max(1, GetWidth());
  const float imgH = std::max(1, GetHeight());
  const float winAspect = vw / vh;
  const float imgAspect = imgW / imgH;

  float sx = 1.0f, sy = 1.0f;
  if (imgAspect > winAspect) {
    sy = winAspect / imgAspect; // image wider -> reduce Y
  } else {
    sx = imgAspect / winAspect; // image taller -> reduce X
  }

  float vertices[] = {
      -sx, -sy,
       sx, -sy,
      -sx,  sy,
       sx,  sy,
  };
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

  // If a new frame is pending, copy staging buffer into the mapped PBO.
  if (frame_pending_.load(std::memory_order_acquire) && cuda_resource_ && staging_ptr_ && staging_bytes_ > 0) {
    cudaError_t map_err = cudaGraphicsMapResources(1, &cuda_resource_, 0);
    if (map_err != cudaSuccess) {
      qWarning("Failed to map CUDA resource (paintGL): %s", cudaGetErrorString(map_err));
    } else {
      float4* mapped_ptr = nullptr;
      size_t  mapped_bytes = 0;
      cudaError_t ptr_err = cudaGraphicsResourceGetMappedPointer(
          reinterpret_cast<void**>(&mapped_ptr), &mapped_bytes, cuda_resource_);
      if (ptr_err != cudaSuccess || !mapped_ptr || mapped_bytes == 0) {
        qWarning("Failed to get mapped pointer (paintGL): %s", cudaGetErrorString(ptr_err));
      } else {
        const size_t copy_bytes = std::min(staging_bytes_, mapped_bytes);
        cudaError_t copy_err = cudaMemcpy(mapped_ptr, staging_ptr_, copy_bytes, cudaMemcpyDeviceToDevice);
        if (copy_err != cudaSuccess) {
          qWarning("Failed to copy staging->PBO: %s", cudaGetErrorString(copy_err));
        } else {
          frame_pending_.store(false, std::memory_order_release);
        }
      }

      cudaError_t unmap_err = cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
      if (unmap_err != cudaSuccess) {
        qWarning("Failed to unmap CUDA resource (paintGL): %s", cudaGetErrorString(unmap_err));
      }
    }
  }

  glClear(GL_COLOR_BUFFER_BIT);
  program_->bind();

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBindTexture(GL_TEXTURE_2D, texture_);

  // Update texture from PBO
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GetWidth(), GetHeight(), GL_RGBA, GL_FLOAT, nullptr);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  // Draw call
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  int pos_loc = program_->attributeLocation("position");
  program_->enableAttributeArray(pos_loc);
  program_->setAttributeBuffer(pos_loc, GL_FLOAT, 0, 2, 0);

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  program_->release();
}

void QtEditViewer::resizeGL(int w, int h) {
  if (w <= 0 || h <= 0) return;

  makeCurrent();

  std::lock_guard<std::mutex> lock(mutex_);
  FreeResources();

  alloc_width_  = w;
  alloc_height_ = h;

  InitPBO();
}

void QtEditViewer::FreeResources() {
  if (cuda_resource_) {
    cudaGraphicsUnregisterResource(cuda_resource_);
    cuda_resource_ = nullptr;
  }
  if (pbo_) {
    glDeleteBuffers(1, &pbo_);
    pbo_ = 0;
  }
  if (texture_) {
    glDeleteTextures(1, &texture_);
    texture_ = 0;
  }
}

void QtEditViewer::OnResizeGL(int w, int h) {
  makeCurrent();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    alloc_width_  = w;
    alloc_height_ = h;
  }
  resizeGL(w, h);
  doneCurrent();
}
};  // namespace puerhlab