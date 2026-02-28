#pragma once

#include <QImage>
#include <QPointF>
#include <QRectF>
#include <QWidget>

#include <functional>

namespace puerhlab::ui {

class CdlTrackballDiscWidget final : public QWidget {
 public:
  using PositionCallback = std::function<void(const QPointF&)>;

  explicit CdlTrackballDiscWidget(QWidget* parent = nullptr);

  void SetPosition(const QPointF& position);
  auto GetPosition() const -> QPointF;

  void SetPositionChangedCallback(PositionCallback cb);
  void SetPositionReleasedCallback(PositionCallback cb);

 protected:
  void resizeEvent(QResizeEvent*) override;
  void paintEvent(QPaintEvent*) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void mouseDoubleClickEvent(QMouseEvent* event) override;

 private:
  auto DiscRect() const -> QRectF;
  auto ToWidgetPoint(const QPointF& normalized, const QRectF& disc) const -> QPointF;
  auto ToNormalizedPoint(const QPointF& widget_point, const QRectF& disc) const -> QPointF;

  void UpdateFromMouse(const QPointF& widget_point, bool emit_changed, bool emit_release);
  void EnsureWheelCache(const QRectF& disc);

  QPointF          position_{0.0, 0.0};
  bool             dragging_ = false;
  QImage           wheel_cache_;
  PositionCallback on_position_changed_{};
  PositionCallback on_position_released_{};
};

}  // namespace puerhlab::ui
