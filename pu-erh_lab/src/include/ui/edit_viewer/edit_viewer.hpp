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

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <qopenglwidget.h>
#include <mutex>

#include "frame_sink.hpp"

namespace puerhlab {

class QtEditViewer : public QOpenGLWidget, protected QOpenGLFunctions, public puerhlab::IFrameSink {
  Q_OBJECT
 public:
  explicit QtEditViewer(QWidget* parent = nullptr);
  ~QtEditViewer();

	// Overrides from IFrameSink
  float4* MapResourceForWrite() override;
  void    UnmapResource() override;
  void    NotifyFrameReady() override;
  int     GetWidth() const override;
  int     GetHeight() const override;

 signals:
  void FrameReadySignal();

 protected:
  // Overrides from QOpenGLWidget
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

 private:
  // OpenGL resources
  GLuint pbo_ = 0;
  
};
};



