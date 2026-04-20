//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QColor>
#include <QObject>
#include <QPaintEvent>
#include <QPainter>
#include <QPen>
#include <QPointF>
#include <QRectF>
#include <QSlider>
#include <QString>
#include <QStringLiteral>

#include <algorithm>

#include "ui/alcedo_main/app_theme.hpp"

namespace alcedo::ui {

struct SliderPaintMetrics {
  int track_height;
  int handle_diameter;
};

inline constexpr SliderPaintMetrics kRegularSliderMetrics{10, 16};
inline constexpr SliderPaintMetrics kCompactSliderMetrics{6, 12};

enum class SliderVisualStyle {
  Accent,
  Native,
};

inline auto WithAlpha(const QColor& color, int alpha) -> QColor {
  QColor tinted(color);
  tinted.setAlpha(std::clamp(alpha, 0, 255));
  return tinted;
}

class AccentBalanceSlider final : public QSlider {
 public:
  explicit AccentBalanceSlider(const SliderPaintMetrics& metrics, QWidget* parent = nullptr)
      : QSlider(Qt::Horizontal, parent), metrics_(metrics) {
    const int handle_margin = std::max(0, (metrics_.handle_diameter - metrics_.track_height) / 2);
    setStyleSheet(QStringLiteral(
                      "QSlider::groove:horizontal {"
                      "  background: transparent;"
                      "  height: %1px;"
                      "}"
                      "QSlider::sub-page:horizontal,"
                      "QSlider::add-page:horizontal {"
                      "  background: transparent;"
                      "}"
                      "QSlider::handle:horizontal {"
                      "  background: transparent;"
                      "  width: %2px;"
                      "  margin: -%3px 0;"
                      "}")
                      .arg(metrics_.track_height)
                      .arg(metrics_.handle_diameter)
                      .arg(handle_margin));
    QObject::connect(&AppTheme::Instance(), &AppTheme::ThemeChanged, this,
                     [this]() { update(); });
  }

 protected:
  void paintEvent(QPaintEvent* event) override {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF track_rect = TrackRect();
    if (track_rect.width() <= 0.0) {
      return;
    }

    const int   anchor_value    = AnchorValue();
    const qreal anchor_x        = PositionForValue(anchor_value, track_rect);
    const qreal value_x         = PositionForValue(value(), track_rect);
    const bool  has_center_zero = minimum() < 0 && maximum() > 0;
    const bool  is_positive     = value() > anchor_value;
    const bool  is_negative     = value() < anchor_value;

    QColor track_fill   = AppTheme::EditorSliderTrackColor();
    QColor border_color = WithAlpha(AppTheme::Instance().dividerColor(), 72);
    if (is_positive) {
      border_color = AppTheme::EditorSliderBorderColor(true);
    } else if (is_negative) {
      border_color = AppTheme::EditorSliderBorderColor(false);
    }

    if (!isEnabled()) {
      track_fill   = WithAlpha(track_fill, 140);
      border_color = WithAlpha(AppTheme::Instance().dividerColor(), 48);
    }

    painter.setPen(QPen(border_color, 1.0));
    painter.setBrush(track_fill);
    painter.drawRoundedRect(track_rect, track_rect.height() * 0.5, track_rect.height() * 0.5);

    if (has_center_zero) {
      painter.setPen(QPen(WithAlpha(AppTheme::Instance().textMutedColor(), 88), 1.0));
      painter.drawLine(QPointF(anchor_x, track_rect.top() + 1.0),
                       QPointF(anchor_x, track_rect.bottom() - 1.0));
    }

    if (!qFuzzyCompare(anchor_x, value_x)) {
      QRectF fill_rect(QPointF(std::min(anchor_x, value_x), track_rect.top() + 1.0),
                       QPointF(std::max(anchor_x, value_x), track_rect.bottom() - 1.0));
      if (fill_rect.width() < 1.0) {
        fill_rect.setWidth(1.0);
      }

      QColor accent_fill = AppTheme::EditorSliderAccentColor(!is_negative);
      if (!isEnabled()) {
        accent_fill = WithAlpha(accent_fill, 120);
      }

      painter.setPen(Qt::NoPen);
      painter.setBrush(accent_fill);
      painter.drawRoundedRect(fill_rect, std::max(1.0, fill_rect.height() * 0.45),
                              std::max(1.0, fill_rect.height() * 0.45));
    }

    const qreal  handle_radius = metrics_.handle_diameter * 0.5;
    const QRectF handle_rect(value_x - handle_radius,
                             (rect().height() - metrics_.handle_diameter) * 0.5,
                             metrics_.handle_diameter, metrics_.handle_diameter);

    QColor handle_fill   = AppTheme::EditorSliderHandleColor();
    QColor handle_border = AppTheme::EditorSliderHandleBorderColor();
    if (!isEnabled()) {
      handle_fill   = WithAlpha(handle_fill, 136);
      handle_border = WithAlpha(handle_border, 120);
    }

    painter.setPen(QPen(handle_border, 1.0));
    painter.setBrush(handle_fill);
    painter.drawEllipse(handle_rect);

    if (hasFocus()) {
      const QColor focus_color = is_negative ? AppTheme::EditorSliderBorderColor(false)
                                             : AppTheme::EditorSliderBorderColor(true);
      painter.setPen(QPen(WithAlpha(focus_color, 176), 1.0));
      painter.setBrush(Qt::NoBrush);
      painter.drawEllipse(handle_rect.adjusted(-2.0, -2.0, 2.0, 2.0));
    }
  }

 private:
  auto AnchorValue() const -> int {
    if (minimum() <= 0 && maximum() >= 0) {
      return 0;
    }
    return minimum() > 0 ? minimum() : maximum();
  }

  auto TrackRect() const -> QRectF {
    const qreal handle_radius = metrics_.handle_diameter * 0.5;
    const qreal left          = handle_radius + 2.0;
    const qreal width         = std::max(0.0, rect().width() - (left * 2.0));
    const qreal top           = (rect().height() - metrics_.track_height) * 0.5;
    return QRectF(left, top, width, metrics_.track_height);
  }

  auto PositionForValue(int slider_value, const QRectF& track_rect) const -> qreal {
    if (maximum() <= minimum() || track_rect.width() <= 0.0) {
      return track_rect.center().x();
    }

    const qreal ratio =
        static_cast<qreal>(slider_value - minimum()) / static_cast<qreal>(maximum() - minimum());
    const qreal visual_ratio = invertedAppearance() ? (1.0 - ratio) : ratio;
    return track_rect.left() + std::clamp(visual_ratio, 0.0, 1.0) * track_rect.width();
  }

  SliderPaintMetrics metrics_;
};

}  // namespace alcedo::ui
