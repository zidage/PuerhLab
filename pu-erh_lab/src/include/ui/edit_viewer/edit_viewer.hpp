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
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <mutex>
#include <cuda_gl_interop.h>

#include "frame_sink.hpp"

namespace puerhlab {

class QtEditViewer : public QOpenGLWidget, protected QOpenGLFunctions, public puerhlab::IFrameSink {
  Q_OBJECT
 public:
  explicit QtEditViewer(QWidget* parent = nullptr);
  ~QtEditViewer();

  // Overrides from IFrameSink
  void    EnsureSize(int width, int height) override;
  float4* MapResourceForWrite() override;
  void    UnmapResource() override;
  void    NotifyFrameReady() override;
  int     GetWidth() const override { return alloc_width_; };
  int     GetHeight() const override { return alloc_height_; };

 signals:
  void RequestUpdate();

  void RequestResize(int width, int height);

 private slots:
  void OnResizeGL(int w, int h);

 protected:
  // Overrides from QOpenGLWidget
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

 private:
  // OpenGL resources
  GLuint                pbo_           = 0;
  GLuint                texture_       = 0;
  GLuint                vbo_           = 0;
  QOpenGLShaderProgram* program_       = nullptr;

  // CUDA resources
  cudaGraphicsResource* cuda_resource_ = nullptr;

  // Thread synchronization
  std::mutex            mutex_;

  // Current size
  int                   alloc_width_  = 0;
  int                   alloc_height_ = 0;

  void                  InitPBO();

  void                  FreeResources();
};
};  // namespace puerhlab
