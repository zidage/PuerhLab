#pragma once

#include <QOpenGLExtraFunctions>
#include <QOpenGLWidget>
#include <QWidget>

class QOpenGLShaderProgram;

namespace puerhlab {
class QtEditViewer;
}

namespace puerhlab::ui {

class HistogramWidget final : public QOpenGLWidget, protected QOpenGLExtraFunctions {
 public:
  explicit HistogramWidget(QtEditViewer* source_viewer, QWidget* parent = nullptr);
  ~HistogramWidget() override;

  void SetSourceViewer(QtEditViewer* source_viewer);

 protected:
  void initializeGL() override;
  void paintGL() override;

 private:
  auto InitPrograms() -> bool;
  void CleanupGl();

  QtEditViewer*         source_viewer_          = nullptr;
  QOpenGLShaderProgram* fill_program_           = nullptr;
  QOpenGLShaderProgram* line_program_           = nullptr;
  unsigned int          vao_                    = 0;
  bool                  gl_ready_               = false;
  bool                  warned_context_sharing_ = false;
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

}  // namespace puerhlab::ui
