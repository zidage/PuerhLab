//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QWidget>

#ifdef ALCEDO_HAS_LEGACY_GL_VIEWER
#include <QOpenGLExtraFunctions>
#include <QOpenGLWidget>
#endif

class QOpenGLShaderProgram;

namespace alcedo {
class QtEditViewer;
}

namespace alcedo::ui {

class HistogramWidget final
#ifdef ALCEDO_HAS_LEGACY_GL_VIEWER
    : public QOpenGLWidget, protected QOpenGLExtraFunctions {
#else
    : public QWidget {
#endif
 public:
  explicit HistogramWidget(QtEditViewer* source_viewer, QWidget* parent = nullptr);
  ~HistogramWidget() override;

  void SetSourceViewer(QtEditViewer* source_viewer);

 protected:
#ifdef ALCEDO_HAS_LEGACY_GL_VIEWER
  void initializeGL() override;
  void paintGL() override;
#else
  void paintEvent(QPaintEvent*) override;
#endif

 private:
#ifdef ALCEDO_HAS_LEGACY_GL_VIEWER
  auto InitPrograms() -> bool;
  void CleanupGl();
#endif

  QtEditViewer*         source_viewer_          = nullptr;
#ifdef ALCEDO_HAS_LEGACY_GL_VIEWER
  QOpenGLShaderProgram* fill_program_           = nullptr;
  QOpenGLShaderProgram* line_program_           = nullptr;
  unsigned int          vao_                    = 0;
  bool                  gl_ready_               = false;
  bool                  warned_context_sharing_ = false;
#endif
};

class HistogramRulerWidget final : public QWidget {
 public:
  explicit HistogramRulerWidget(int bins, QWidget* parent = nullptr);

  void SetBins(int bins);

 protected:
  void paintEvent(QPaintEvent*) override;

 private:
  int bins_ = 256;
};

}  // namespace alcedo::ui
