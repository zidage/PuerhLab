#include "ui/puerhlab_main/editor_dialog/widgets/tone_curve_widget.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPen>

#include <algorithm>
#include <cmath>

#include "ui/puerhlab_main/editor_dialog/modules/curve.hpp"

namespace puerhlab::ui {

ToneCurveWidget::ToneCurveWidget(QWidget* parent) : QWidget(parent) {
  setMinimumSize(270, 220);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  setMouseTracking(true);
  setCursor(Qt::CrossCursor);
  points_ = curve::DefaultCurveControlPoints();
}

void ToneCurveWidget::SetControlPoints(const std::vector<QPointF>& points) {
  points_     = curve::NormalizeCurveControlPoints(points);
  active_idx_ = -1;
  dragging_   = false;
  update();
}

auto ToneCurveWidget::GetControlPoints() const -> const std::vector<QPointF>& { return points_; }

void ToneCurveWidget::SetCurveChangedCallback(CurveCallback cb) { on_curve_changed_ = std::move(cb); }

void ToneCurveWidget::SetCurveReleasedCallback(CurveCallback cb) {
  on_curve_released_ = std::move(cb);
}

void ToneCurveWidget::paintEvent(QPaintEvent*) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);

  painter.fillRect(rect(), QColor(0x1A, 0x1A, 0x1A));
  painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
  painter.drawRoundedRect(rect().adjusted(0, 0, -1, -1), 10.0, 10.0);

  const QRectF plot = PlotRect();
  painter.fillRect(plot, QColor(0x12, 0x12, 0x12));

  painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
  for (int i = 1; i < 4; ++i) {
    const qreal t  = static_cast<qreal>(i) / 4.0;
    const qreal gx = plot.left() + t * plot.width();
    const qreal gy = plot.top() + t * plot.height();
    painter.drawLine(QPointF(gx, plot.top()), QPointF(gx, plot.bottom()));
    painter.drawLine(QPointF(plot.left(), gy), QPointF(plot.right(), gy));
  }

  painter.setPen(QPen(QColor(0x8C, 0x8C, 0x8C), 1.0, Qt::DashLine));
  painter.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.top()));

  const auto    cache = curve::BuildCurveHermiteCache(points_);
  QPainterPath  curve_path;
  constexpr int kSamples = 240;
  for (int i = 0; i <= kSamples; ++i) {
    const float   x = static_cast<float>(i) / static_cast<float>(kSamples);
    const float   y = curve::EvaluateCurveHermite(x, points_, cache);
    const QPointF p = ToWidgetPoint(QPointF(x, y));
    if (i == 0) {
      curve_path.moveTo(p);
    } else {
      curve_path.lineTo(p);
    }
  }

  painter.setPen(QPen(QColor(0xFC, 0xC7, 0x04), 2.0));
  painter.drawPath(curve_path);

  for (size_t i = 0; i < points_.size(); ++i) {
    const QPointF p       = ToWidgetPoint(points_[i]);
    const bool    active  = static_cast<int>(i) == active_idx_;
    const bool    pinned  = (i == 0 || i + 1 == points_.size());
    const QColor  fill    = active ? QColor(0xFC, 0xC7, 0x04) : QColor(0xE6, 0xE6, 0xE6);
    const QColor  outline = pinned ? QColor(0xFC, 0xC7, 0x04) : QColor(0x2A, 0x2A, 0x2A);

    painter.setPen(QPen(outline, 1.5));
    painter.setBrush(fill);
    painter.drawEllipse(p, active ? 5.5 : 4.5, active ? 5.5 : 4.5);
  }

  painter.setPen(QColor(0x8C, 0x8C, 0x8C));
  painter.drawText(QRectF(plot.left() - 2.0, plot.bottom() + 4.0, 32.0, 14.0), "0");
  painter.drawText(QRectF(plot.right() - 10.0, plot.bottom() + 4.0, 20.0, 14.0), "1");
  painter.drawText(QRectF(plot.left() - 16.0, plot.top() - 2.0, 14.0, 14.0), "1");
  painter.drawText(QRectF(plot.left() - 16.0, plot.bottom() - 10.0, 14.0, 14.0), "0");
}

void ToneCurveWidget::mousePressEvent(QMouseEvent* event) {
  if (!event) {
    return;
  }

  const QPointF pos = event->position();
  if (event->button() == Qt::RightButton) {
    const int hit_idx = HitTestPoint(pos);
    if (hit_idx > 0 && hit_idx + 1 < static_cast<int>(points_.size())) {
      points_.erase(points_.begin() + hit_idx);
      points_     = curve::NormalizeCurveControlPoints(points_);
      active_idx_ = -1;
      dragging_   = false;
      NotifyCurveChanged();
      NotifyCurveReleased();
      update();
    }
    return;
  }

  if (event->button() != Qt::LeftButton) {
    return;
  }

  const int hit_idx = HitTestPoint(pos);
  if (hit_idx >= 0) {
    active_idx_ = hit_idx;
    dragging_   = true;
    update();
    return;
  }

  if (!PlotRect().contains(pos) ||
      static_cast<int>(points_.size()) >= curve::kCurveMaxControlPoints) {
    return;
  }

  const QPointF normalized = ToNormalizedPoint(pos);
  if (normalized.x() <= curve::kCurveMinPointSpacing ||
      normalized.x() >= (1.0 - curve::kCurveMinPointSpacing)) {
    return;
  }

  points_.push_back(normalized);
  points_     = curve::NormalizeCurveControlPoints(points_);
  active_idx_ = FindClosestPointIndex(normalized);
  dragging_   = true;
  NotifyCurveChanged();
  update();
}

void ToneCurveWidget::mouseMoveEvent(QMouseEvent* event) {
  if (!event || !dragging_ || active_idx_ < 0 ||
      active_idx_ >= static_cast<int>(points_.size())) {
    return;
  }

  const int     last_idx   = static_cast<int>(points_.size()) - 1;
  const QPointF normalized = ToNormalizedPoint(event->position());
  if (active_idx_ == 0) {
    points_[0] = QPointF(0.0, curve::Clamp01(static_cast<float>(normalized.y())));
    NotifyCurveChanged();
    update();
    return;
  }
  if (active_idx_ == last_idx) {
    points_[last_idx] = QPointF(1.0, curve::Clamp01(static_cast<float>(normalized.y())));
    NotifyCurveChanged();
    update();
    return;
  }

  const float min_x    = static_cast<float>(points_[active_idx_ - 1].x()) + curve::kCurveMinPointSpacing;
  const float max_x    = static_cast<float>(points_[active_idx_ + 1].x()) - curve::kCurveMinPointSpacing;

  const float x        = std::clamp(static_cast<float>(normalized.x()), min_x, max_x);
  const float y        = curve::Clamp01(static_cast<float>(normalized.y()));
  points_[active_idx_] = QPointF(x, y);

  NotifyCurveChanged();
  update();
}

void ToneCurveWidget::mouseReleaseEvent(QMouseEvent* event) {
  if (!event || event->button() != Qt::LeftButton) {
    return;
  }
  if (!dragging_) {
    return;
  }
  dragging_ = false;
  NotifyCurveReleased();
}

auto ToneCurveWidget::PlotRect() const -> QRectF {
  constexpr qreal kLeft   = 22.0;
  constexpr qreal kTop    = 14.0;
  constexpr qreal kRight  = 12.0;
  constexpr qreal kBottom = 24.0;
  return QRectF(kLeft, kTop, std::max(30.0, width() - kLeft - kRight),
                std::max(30.0, height() - kTop - kBottom));
}

auto ToneCurveWidget::ToWidgetPoint(const QPointF& normalized) const -> QPointF {
  const QRectF plot = PlotRect();
  const qreal  x    = plot.left() + normalized.x() * plot.width();
  const qreal  y    = plot.bottom() - normalized.y() * plot.height();
  return QPointF(x, y);
}

auto ToneCurveWidget::ToNormalizedPoint(const QPointF& widget_point) const -> QPointF {
  const QRectF plot = PlotRect();
  const qreal  nx =
      std::clamp((widget_point.x() - plot.left()) / std::max(1.0, plot.width()), 0.0, 1.0);
  const qreal ny =
      std::clamp((plot.bottom() - widget_point.y()) / std::max(1.0, plot.height()), 0.0, 1.0);
  return QPointF(nx, ny);
}

auto ToneCurveWidget::HitTestPoint(const QPointF& widget_point) const -> int {
  constexpr qreal kHitRadiusSq = 10.0 * 10.0;
  for (int i = static_cast<int>(points_.size()) - 1; i >= 0; --i) {
    const QPointF p       = ToWidgetPoint(points_[i]);
    const qreal   dx      = p.x() - widget_point.x();
    const qreal   dy      = p.y() - widget_point.y();
    const qreal   dist_sq = dx * dx + dy * dy;
    if (dist_sq <= kHitRadiusSq) {
      return i;
    }
  }
  return -1;
}

auto ToneCurveWidget::FindClosestPointIndex(const QPointF& normalized) const -> int {
  if (points_.empty()) {
    return -1;
  }
  int   best_idx = 0;
  qreal best_dist =
      std::abs(points_[0].x() - normalized.x()) + std::abs(points_[0].y() - normalized.y());
  for (int i = 1; i < static_cast<int>(points_.size()); ++i) {
    const qreal dist =
        std::abs(points_[i].x() - normalized.x()) + std::abs(points_[i].y() - normalized.y());
    if (dist < best_dist) {
      best_dist = dist;
      best_idx  = i;
    }
  }
  return best_idx;
}

void ToneCurveWidget::NotifyCurveChanged() {
  if (on_curve_changed_) {
    on_curve_changed_(points_);
  }
}

void ToneCurveWidget::NotifyCurveReleased() {
  if (on_curve_released_) {
    on_curve_released_(points_);
  }
}

}  // namespace puerhlab::ui
