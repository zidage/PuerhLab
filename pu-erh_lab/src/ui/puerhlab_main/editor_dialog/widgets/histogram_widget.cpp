#include "ui/puerhlab_main/editor_dialog/widgets/histogram_widget.hpp"

#include <QByteArray>
#include <QOpenGLContext>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QPainter>
#include <QPen>

#include <array>
#include <algorithm>
#include <cmath>

#include "ui/edit_viewer/edit_viewer.hpp"

namespace puerhlab::ui {

HistogramWidget::HistogramWidget(QtEditViewer* source_viewer, QWidget* parent)
    : QOpenGLWidget(parent), source_viewer_(source_viewer) {
  setMinimumHeight(126);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  setAutoFillBackground(false);
}

HistogramWidget::~HistogramWidget() {
  if (!context()) {
    return;
  }
  makeCurrent();
  CleanupGl();
  doneCurrent();
}

void HistogramWidget::SetSourceViewer(QtEditViewer* source_viewer) {
  source_viewer_ = source_viewer;
  update();
}

void HistogramWidget::initializeGL() {
  initializeOpenGLFunctions();
  glDisable(GL_DEPTH_TEST);
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);
  glBindVertexArray(0);
  gl_ready_ = InitPrograms();
}

void HistogramWidget::paintGL() {
  const float dpr = devicePixelRatioF();
  const int   vw  = std::max(1, static_cast<int>(std::lround(width() * dpr)));
  const int   vh  = std::max(1, static_cast<int>(std::lround(height() * dpr)));
  glViewport(0, 0, vw, vh);
  glClearColor(0.07f, 0.07f, 0.07f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  if (!gl_ready_ || !source_viewer_ || !source_viewer_->HasHistogramData()) {
    return;
  }

  if (context() && source_viewer_->context() &&
      !QOpenGLContext::areSharing(context(), source_viewer_->context())) {
    if (!warned_context_sharing_) {
      qWarning("HistogramWidget disabled: OpenGL contexts are not sharing resources.");
      warned_context_sharing_ = true;
    }
    return;
  }

  const GLuint hist_buffer = source_viewer_->GetHistogramBufferId();
  const int    bins        = source_viewer_->GetHistogramBinCount();
  if (hist_buffer == 0 || bins <= 1 || !glIsBuffer(hist_buffer)) {
    return;
  }

  glBindVertexArray(vao_);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, hist_buffer);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  auto draw_fill = [&](int channel, const QVector4D& color) {
    if (!fill_program_) {
      return;
    }
    fill_program_->bind();
    fill_program_->setUniformValue("uBins", bins);
    fill_program_->setUniformValue("uChannel", channel);
    fill_program_->setUniformValue("uColor", color);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, bins * 2);
    fill_program_->release();
  };

  auto draw_line = [&](int channel, const QVector4D& color) {
    if (!line_program_) {
      return;
    }
    line_program_->bind();
    line_program_->setUniformValue("uBins", bins);
    line_program_->setUniformValue("uChannel", channel);
    line_program_->setUniformValue("uColor", color);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINE_STRIP, 0, bins);
    line_program_->release();
  };

  draw_fill(0, QVector4D(1.0f, 0.20f, 0.20f, 0.30f));
  draw_fill(1, QVector4D(0.20f, 1.0f, 0.20f, 0.28f));
  draw_fill(2, QVector4D(0.20f, 0.45f, 1.0f, 0.28f));

  draw_line(0, QVector4D(1.0f, 0.45f, 0.45f, 0.24f));
  draw_line(1, QVector4D(0.45f, 1.0f, 0.45f, 0.22f));
  draw_line(2, QVector4D(0.45f, 0.68f, 1.0f, 0.22f));

  glDisable(GL_BLEND);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
  glBindVertexArray(0);
}

auto HistogramWidget::InitPrograms() -> bool {
  if (!context()) {
    return false;
  }

  const auto format = context()->format();
  const bool has_compute_compatible_ssbo =
      (format.majorVersion() > 4 || (format.majorVersion() == 4 && format.minorVersion() >= 3)) ||
      context()->hasExtension(QByteArrayLiteral("GL_ARB_shader_storage_buffer_object"));
  if (!has_compute_compatible_ssbo) {
    qWarning("HistogramWidget disabled: OpenGL SSBO support is not available.");
    return false;
  }

  static const char* kFillVertex = R"(
#version 430 core
layout(std430, binding = 0) readonly buffer HistogramBuffer {
  float hist[];
};
uniform int uBins;
uniform int uChannel;
void main() {
  int bin = gl_VertexID >> 1;
  int top = gl_VertexID & 1;
  float x = (uBins > 1) ? float(bin) / float(uBins - 1) : 0.0;
  float y = (top == 0) ? 0.0 : clamp(hist[uChannel * uBins + bin], 0.0, 1.0);
  gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
)";

  static const char* kLineVertex = R"(
#version 430 core
layout(std430, binding = 0) readonly buffer HistogramBuffer {
  float hist[];
};
uniform int uBins;
uniform int uChannel;
void main() {
  int bin = gl_VertexID;
  float x = (uBins > 1) ? float(bin) / float(uBins - 1) : 0.0;
  float y = clamp(hist[uChannel * uBins + bin], 0.0, 1.0);
  gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
)";

  static const char* kFragment   = R"(
#version 430 core
uniform vec4 uColor;
out vec4 FragColor;
void main() {
  FragColor = uColor;
}
)";

  fill_program_                  = new QOpenGLShaderProgram();
  if (!fill_program_->addShaderFromSourceCode(QOpenGLShader::Vertex, kFillVertex) ||
      !fill_program_->addShaderFromSourceCode(QOpenGLShader::Fragment, kFragment) ||
      !fill_program_->link()) {
    qWarning("HistogramWidget fill program failed: %s",
             fill_program_->log().toUtf8().constData());
    CleanupGl();
    return false;
  }

  line_program_ = new QOpenGLShaderProgram();
  if (!line_program_->addShaderFromSourceCode(QOpenGLShader::Vertex, kLineVertex) ||
      !line_program_->addShaderFromSourceCode(QOpenGLShader::Fragment, kFragment) ||
      !line_program_->link()) {
    qWarning("HistogramWidget line program failed: %s",
             line_program_->log().toUtf8().constData());
    CleanupGl();
    return false;
  }
  return true;
}

void HistogramWidget::CleanupGl() {
  if (fill_program_) {
    delete fill_program_;
    fill_program_ = nullptr;
  }
  if (line_program_) {
    delete line_program_;
    line_program_ = nullptr;
  }
  if (vao_) {
    glDeleteVertexArrays(1, &vao_);
    vao_ = 0;
  }
  gl_ready_ = false;
}

HistogramRulerWidget::HistogramRulerWidget(int bins, QWidget* parent)
    : QWidget(parent), bins_(std::max(2, bins)) {
  setMinimumHeight(28);
  setMaximumHeight(36);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  setAttribute(Qt::WA_TransparentForMouseEvents);
}

void HistogramRulerWidget::SetBins(int bins) {
  bins_ = std::max(2, bins);
  update();
}

void HistogramRulerWidget::paintEvent(QPaintEvent*) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);

  QFont ruler_font = painter.font();
  ruler_font.setPixelSize(9);
  painter.setFont(ruler_font);

  const QRectF area(10.0, 6.0, std::max(10.0, width() - 20.0), std::max(10.0, height() - 12.0));
  const qreal  baseline_y = area.top() + 2.0;
  const qreal  tick_h     = 7.0;

  painter.setPen(QPen(QColor(0x4A, 0x4A, 0x4A), 1.0));
  painter.drawLine(QPointF(area.left(), baseline_y), QPointF(area.right(), baseline_y));

  constexpr std::array<double, 5> stops = {0.0, 0.25, 0.50, 0.75, 1.0};
  painter.setPen(QPen(QColor(0x6F, 0x6F, 0x6F), 1.0));
  for (const double t : stops) {
    const qreal x = area.left() + t * area.width();
    painter.drawLine(QPointF(x, baseline_y), QPointF(x, baseline_y + tick_h));
  }

  painter.setPen(QColor(0x9A, 0x9A, 0x9A));
  for (const double t : stops) {
    const qreal   x    = area.left() + t * area.width();
    const QString text = QString::number(t, 'f', 2);
    const QRectF  text_rect(x - 20.0, baseline_y + tick_h + 1.0, 40.0, 14.0);
    painter.drawText(text_rect, Qt::AlignHCenter | Qt::AlignTop, text);
  }
}

}  // namespace puerhlab::ui
