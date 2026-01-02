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
#include <driver_types.h>
#include <qopenglext.h>
#include <qoverload.h>

#include <mutex>

namespace puerhlab {

static const char* vertexShaderSource   = R"(
    attribute vec2 position;
    varying vec2 vTexCoord;
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        vTexCoord = (position + 1.0) * 0.5;
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
}

void QtEditViewer::EnsureSize(int width, int height) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (alloc_width_ == width && alloc_height_ == height) {
      return;
    }
  }

  emit RequestResize(width, height);
}

float4* QtEditViewer::MapResourceForWrite() {
  mutex_.lock();

  if (!cuda_resource_) {
    // No resource to map
    mutex_.unlock();
    return nullptr;
  }

  cudaGraphicsMapResources(1, &cuda_resource_, 0);

  float4* d_ptr = nullptr;
  size_t  num_bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, &num_bytes, cuda_resource_);

  return d_ptr;
}

void QtEditViewer::UnmapResource() {
  if (cuda_resource_) {
    cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
  }
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

  InitPBO();
}

void QtEditViewer::InitPBO() {
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

  if (!pbo_ || !texture_) {
    // Resources not initialized
    return;
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