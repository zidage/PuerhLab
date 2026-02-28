#pragma once

#include <QPointF>
#include <QRectF>
#include <QWidget>

#include <functional>
#include <vector>

namespace puerhlab::ui {

class ToneCurveWidget final : public QWidget {
 public:
  using CurveCallback = std::function<void(const std::vector<QPointF>&)>;

  explicit ToneCurveWidget(QWidget* parent = nullptr);

  void SetControlPoints(const std::vector<QPointF>& points);
  auto GetControlPoints() const -> const std::vector<QPointF>&;

  void SetCurveChangedCallback(CurveCallback cb);
  void SetCurveReleasedCallback(CurveCallback cb);

 protected:
  void paintEvent(QPaintEvent*) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;

 private:
  auto PlotRect() const -> QRectF;
  auto ToWidgetPoint(const QPointF& normalized) const -> QPointF;
  auto ToNormalizedPoint(const QPointF& widget_point) const -> QPointF;
  auto HitTestPoint(const QPointF& widget_point) const -> int;
  auto FindClosestPointIndex(const QPointF& normalized) const -> int;

  void NotifyCurveChanged();
  void NotifyCurveReleased();

  std::vector<QPointF> points_{};
  int                  active_idx_ = -1;
  bool                 dragging_   = false;
  CurveCallback        on_curve_changed_{};
  CurveCallback        on_curve_released_{};
};

}  // namespace puerhlab::ui
