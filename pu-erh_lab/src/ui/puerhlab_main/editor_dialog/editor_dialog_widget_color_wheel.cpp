class CdlTrackballDiscWidget final : public QWidget {
 public:
  using PositionCallback = std::function<void(const QPointF&)>;

  explicit CdlTrackballDiscWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumSize(80, 80);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    setMouseTracking(true);
    setCursor(Qt::CrossCursor);
  }

  void SetPosition(const QPointF& position) {
    const QPointF clamped = ClampDiscPoint(position);
    if (std::abs(clamped.x() - position_.x()) <= 1e-6 && std::abs(clamped.y() - position_.y()) <= 1e-6) {
      return;
    }
    position_ = clamped;
    update();
  }

  auto GetPosition() const -> QPointF { return position_; }

  void SetPositionChangedCallback(PositionCallback cb) { on_position_changed_ = std::move(cb); }

  void SetPositionReleasedCallback(PositionCallback cb) { on_position_released_ = std::move(cb); }

 protected:
  void resizeEvent(QResizeEvent*) override { wheel_cache_ = QImage(); }

  void paintEvent(QPaintEvent*) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF disc = DiscRect();
    EnsureWheelCache(disc);

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(0x0F, 0x0F, 0x0F));
    painter.drawEllipse(disc.adjusted(-2, -2, 2, 2));

    painter.save();
    QPainterPath clip_path;
    clip_path.addEllipse(disc);
    painter.setClipPath(clip_path);
    if (!wheel_cache_.isNull()) {
      painter.drawImage(disc.topLeft(), wheel_cache_);
    }
    painter.restore();

    painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.5));
    painter.setBrush(Qt::NoBrush);
    painter.drawEllipse(disc);

    const QPointF center = disc.center();
    painter.setPen(QPen(QColor(0xB3, 0xB3, 0xB3, 180), 1.0));
    painter.drawLine(QPointF(center.x() - 6.0, center.y()), QPointF(center.x() + 6.0, center.y()));
    painter.drawLine(QPointF(center.x(), center.y() - 6.0), QPointF(center.x(), center.y() + 6.0));

    const QPointF handle = ToWidgetPoint(position_, disc);
    painter.setPen(QPen(QColor(0x12, 0x12, 0x12), 2.0));
    painter.setBrush(QColor(0xFC, 0xC7, 0x04));
    painter.drawEllipse(handle, 6.0, 6.0);
    painter.setPen(QPen(QColor(0xF2, 0xF2, 0xF2), 1.0));
    painter.setBrush(Qt::NoBrush);
    painter.drawEllipse(handle, 9.0, 9.0);
  }

  void mousePressEvent(QMouseEvent* event) override {
    if (event->button() != Qt::LeftButton) {
      QWidget::mousePressEvent(event);
      return;
    }
    dragging_ = true;
    UpdateFromMouse(QPointF(event->pos()), true, false);
    event->accept();
  }

  void mouseMoveEvent(QMouseEvent* event) override {
    if (!dragging_) {
      QWidget::mouseMoveEvent(event);
      return;
    }
    UpdateFromMouse(QPointF(event->pos()), true, false);
    event->accept();
  }

  void mouseReleaseEvent(QMouseEvent* event) override {
    if (event->button() != Qt::LeftButton) {
      QWidget::mouseReleaseEvent(event);
      return;
    }
    if (!dragging_) {
      event->accept();
      return;
    }
    dragging_ = false;
    UpdateFromMouse(QPointF(event->pos()), true, true);
    event->accept();
  }

  void mouseDoubleClickEvent(QMouseEvent* event) override {
    if (event->button() != Qt::LeftButton) {
      QWidget::mouseDoubleClickEvent(event);
      return;
    }
    dragging_ = false;
    position_ = QPointF(0.0, 0.0);
    update();
    if (on_position_changed_) {
      on_position_changed_(position_);
    }
    if (on_position_released_) {
      on_position_released_(position_);
    }
    event->accept();
  }

 private:
  auto DiscRect() const -> QRectF {
    const float side   = std::max(1.0f, static_cast<float>(std::min(width(), height())) - 6.0f);
    const float left   = (static_cast<float>(width()) - side) * 0.5f;
    const float top    = (static_cast<float>(height()) - side) * 0.5f;
    return QRectF(left, top, side, side);
  }

  auto ToWidgetPoint(const QPointF& normalized, const QRectF& disc) const -> QPointF {
    const float radius = static_cast<float>(disc.width() * 0.5);
    const QPointF center = disc.center();
    const QPointF p = ClampDiscPoint(normalized);
    return QPointF(center.x() + p.x() * radius, center.y() - p.y() * radius);
  }

  auto ToNormalizedPoint(const QPointF& widget_point, const QRectF& disc) const -> QPointF {
    const float radius = static_cast<float>(disc.width() * 0.5);
    if (radius <= 0.0f) {
      return QPointF(0.0, 0.0);
    }
    const QPointF center = disc.center();
    const float x        = static_cast<float>((widget_point.x() - center.x()) / radius);
    const float y        = static_cast<float>((center.y() - widget_point.y()) / radius);
    return ClampDiscPoint(QPointF(x, y));
  }

  void UpdateFromMouse(const QPointF& widget_point, bool emit_changed, bool emit_release) {
    const QPointF next = ToNormalizedPoint(widget_point, DiscRect());
    if (std::abs(next.x() - position_.x()) > 1e-6 || std::abs(next.y() - position_.y()) > 1e-6) {
      position_ = next;
      update();
      if (emit_changed && on_position_changed_) {
        on_position_changed_(position_);
      }
    } else if (emit_changed && on_position_changed_) {
      on_position_changed_(position_);
    }
    if (emit_release && on_position_released_) {
      on_position_released_(position_);
    }
  }

  void EnsureWheelCache(const QRectF& disc) {
    const int size = static_cast<int>(std::lround(disc.width()));
    if (size <= 0) {
      wheel_cache_ = QImage();
      return;
    }
    if (!wheel_cache_.isNull() && wheel_cache_.width() == size && wheel_cache_.height() == size) {
      return;
    }

    wheel_cache_ = QImage(size, size, QImage::Format_ARGB32_Premultiplied);
    wheel_cache_.fill(Qt::transparent);

    const float radius = static_cast<float>(size) * 0.5f;
    const float cx     = radius - 0.5f;
    const float cy     = radius - 0.5f;
    for (int y = 0; y < size; ++y) {
      QRgb* row = reinterpret_cast<QRgb*>(wheel_cache_.scanLine(y));
      for (int x = 0; x < size; ++x) {
        const float dx = (static_cast<float>(x) - cx) / std::max(radius, 1.0f);
        const float dy = (cy - static_cast<float>(y)) / std::max(radius, 1.0f);
        const float r  = std::sqrt(dx * dx + dy * dy);
        if (r > 1.0f) {
          row[x] = qRgba(0, 0, 0, 0);
          continue;
        }
        float h = std::atan2(dy, dx) / (2.0f * std::numbers::pi_v<float>);
        if (h < 0.0f) {
          h += 1.0f;
        }
        const QColor color = QColor::fromHsvF(h, std::clamp(r, 0.0f, 1.0f), 1.0f);
        row[x]             = qRgba(color.red(), color.green(), color.blue(), 255);
      }
    }
  }

  QPointF          position_{0.0, 0.0};
  bool             dragging_ = false;
  QImage           wheel_cache_;
  PositionCallback on_position_changed_{};
  PositionCallback on_position_released_{};
};

