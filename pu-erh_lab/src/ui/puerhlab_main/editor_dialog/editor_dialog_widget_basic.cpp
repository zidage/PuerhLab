class SpinnerWidget final : public QWidget {
 public:
  explicit SpinnerWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setFixedSize(22, 22);
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_TranslucentBackground);

    timer_ = new QTimer(this);
    timer_->setInterval(16);
    QObject::connect(timer_, &QTimer::timeout, this, [this]() {
      angle_deg_ = (angle_deg_ + 18) % 360;
      update();
    });
    hide();
  }

  void Start() {
    show();
    raise();
    if (!timer_->isActive()) {
      timer_->start();
    }
  }

  void Stop() {
    if (timer_->isActive()) {
      timer_->stop();
    }
    hide();
  }

 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF r = QRectF(2.5, 2.5, width() - 5.0, height() - 5.0);

    // Subtle background ring.
    {
      QPen pen(QColor(0x3A, 0x3A, 0x3A, 180));
      pen.setWidthF(2.0);
      pen.setCapStyle(Qt::RoundCap);
      painter.setPen(pen);
      painter.drawArc(r, 0 * 16, 360 * 16);
    }

    // Foreground arc.
    {
      QPen pen(QColor(0xFC, 0xC7, 0x04, 230));
      pen.setWidthF(2.2);
      pen.setCapStyle(Qt::RoundCap);
      painter.setPen(pen);
      painter.drawArc(r, (90 - angle_deg_) * 16, 100 * 16);
    }
  }

 private:
  QTimer* timer_     = nullptr;
  int     angle_deg_ = 0;
};

class HistogramWidget final : public QOpenGLWidget, protected QOpenGLExtraFunctions {
 public:
  explicit HistogramWidget(QtEditViewer* source_viewer, QWidget* parent = nullptr)
      : QOpenGLWidget(parent), source_viewer_(source_viewer) {
    setMinimumHeight(126);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setAutoFillBackground(false);
  }

  ~HistogramWidget() override {
    if (!context()) {
      return;
    }
    makeCurrent();
    CleanupGl();
    doneCurrent();
  }

  void SetSourceViewer(QtEditViewer* source_viewer) {
    source_viewer_ = source_viewer;
    update();
  }

 protected:
  void initializeGL() override {
    initializeOpenGLFunctions();
    glDisable(GL_DEPTH_TEST);
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glBindVertexArray(0);
    gl_ready_ = InitPrograms();
  }

  void paintGL() override {
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

 private:
  auto InitPrograms() -> bool {
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

  void CleanupGl() {
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

  QtEditViewer*         source_viewer_          = nullptr;
  QOpenGLShaderProgram* fill_program_           = nullptr;
  QOpenGLShaderProgram* line_program_           = nullptr;
  GLuint                vao_                    = 0;
  bool                  gl_ready_               = false;
  bool                  warned_context_sharing_ = false;
};

class HistogramRulerWidget final : public QWidget {
 public:
  explicit HistogramRulerWidget(int bins, QWidget* parent = nullptr)
      : QWidget(parent), bins_(std::max(2, bins)) {
    setMinimumHeight(28);
    setMaximumHeight(36);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setAttribute(Qt::WA_TransparentForMouseEvents);
  }

  void SetBins(int bins) {
    bins_ = std::max(2, bins);
    update();
  }

 protected:
  void paintEvent(QPaintEvent*) override {
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

 private:
  int bins_ = 256;
};

class HistoryLaneWidget final : public QWidget {
 public:
  HistoryLaneWidget(QColor dot, QColor line, bool draw_top, bool draw_bottom,
                    QWidget* parent = nullptr)
      : QWidget(parent),
        dot_(std::move(dot)),
        line_(std::move(line)),
        draw_top_(draw_top),
        draw_bottom_(draw_bottom) {
    setFixedWidth(18);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    setAttribute(Qt::WA_TransparentForMouseEvents);
  }

  void SetConnectors(bool draw_top, bool draw_bottom) {
    draw_top_    = draw_top;
    draw_bottom_ = draw_bottom;
    update();
  }

 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const int cx = width() / 2;
    const int cy = height() / 2;

    // Vertical lane.
    {
      QPen pen(line_);
      pen.setWidthF(2.0);
      pen.setCapStyle(Qt::RoundCap);
      p.setPen(pen);

      if (draw_top_) {
        p.drawLine(QPointF(cx, 2.0), QPointF(cx, cy - 6.0));
      }
      if (draw_bottom_) {
        p.drawLine(QPointF(cx, cy + 6.0), QPointF(cx, height() - 2.0));
      }
    }

    // Node.
    {
      p.setPen(Qt::NoPen);
      p.setBrush(dot_);
      p.drawEllipse(QPointF(cx, cy), 4.4, 4.4);
      p.setBrush(QColor(0x12, 0x12, 0x12));
      p.drawEllipse(QPointF(cx, cy), 2.0, 2.0);
    }
  }

 private:
  QColor dot_;
  QColor line_;
  bool   draw_top_    = false;
  bool   draw_bottom_ = false;
};

class HistoryCardWidget final : public QFrame {
 public:
  explicit HistoryCardWidget(QWidget* parent = nullptr) : QFrame(parent) {
    setObjectName("HistoryCard");
    setAttribute(Qt::WA_StyledBackground, true);
    setAttribute(Qt::WA_Hover, true);
    setProperty("selected", false);

    setStyleSheet(
        "QFrame#HistoryCard {"
        "  background: #1A1A1A;"
        "  border: none;"
        "  border-radius: 10px;"
        "}"
        "QFrame#HistoryCard:hover {"
        "  background: #202020;"
        "}"
        "QFrame#HistoryCard[selected=\"true\"] {"
        "  background: rgba(252, 199, 4, 0.20);"
        "  border: 2px solid #FCC704;"
        "}");
  }

  void SetSelected(bool selected) {
    if (property("selected").toBool() == selected) {
      return;
    }
    setProperty("selected", selected);
    style()->unpolish(this);
    style()->polish(this);
    update();
  }
};

static QLabel* MakePillLabel(const QString& text, const QString& fg, const QString& bg,
                             const QString& border, QWidget* parent) {
  auto* l = new QLabel(text, parent);
  l->setStyleSheet(QString("QLabel {"
                           "  color: %1;"
                           "  background: %2;"
                           "  border: 1px solid %3;"
                           "  border-radius: 10px;"
                           "  padding: 1px 7px;"
                           "  font-size: 11px;"
                           "}")
                       .arg(fg, bg, border));
  return l;
}

