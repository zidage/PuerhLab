//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/scope/scope_plot_widgets.hpp"

#include <QPainter>
#include <QPainterPath>
#include <QPen>

#include <algorithm>

#include "ui/puerhlab_main/app_theme.hpp"

namespace puerhlab::ui {

ScopeHistogramWidget::ScopeHistogramWidget(QWidget* parent) : QWidget(parent) {
  setMinimumHeight(126);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  setAutoFillBackground(false);
}

void ScopeHistogramWidget::SetPresentation(const ScopeHistogramPresentation& presentation) {
  presentation_ = presentation;
  update();
}

void ScopeHistogramWidget::paintEvent(QPaintEvent*) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setRenderHint(QPainter::TextAntialiasing, true);
  painter.fillRect(rect(), QColor(0x10, 0x10, 0x10));

  const QRectF plot_rect = rect().adjusted(10, 10, -10, -10);
  painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
  painter.drawRoundedRect(plot_rect, 8.0, 8.0);

  if (!presentation_.valid || presentation_.bins <= 1 ||
      presentation_.rgb.size() < static_cast<size_t>(presentation_.bins * 3)) {
    painter.setPen(QColor(0x9A, 0x9A, 0x9A));
    painter.setFont(AppTheme::Font(AppTheme::FontRole::DataCaption));
    painter.drawText(plot_rect, Qt::AlignCenter, QStringLiteral("Waiting for histogram"));
    return;
  }

  const QColor fill_colors[3] = {QColor(255, 64, 64, 76), QColor(64, 255, 96, 72),
                                 QColor(64, 128, 255, 72)};
  const QColor line_colors[3] = {QColor(255, 140, 140), QColor(140, 255, 160),
                                 QColor(128, 184, 255)};

  for (int channel = 0; channel < 3; ++channel) {
    QPainterPath fill_path;
    QPainterPath line_path;

    fill_path.moveTo(plot_rect.left(), plot_rect.bottom());
    for (int bin = 0; bin < presentation_.bins; ++bin) {
      const float value = std::clamp(
          presentation_.rgb[static_cast<size_t>(channel) * static_cast<size_t>(presentation_.bins) +
                            static_cast<size_t>(bin)],
          0.0f, 1.0f);
      const qreal x = plot_rect.left() +
                      (static_cast<qreal>(bin) / static_cast<qreal>(presentation_.bins - 1)) *
                          plot_rect.width();
      const qreal y = plot_rect.bottom() - static_cast<qreal>(value) * plot_rect.height();
      if (bin == 0) {
        line_path.moveTo(x, y);
      } else {
        line_path.lineTo(x, y);
      }
      fill_path.lineTo(x, y);
    }
    fill_path.lineTo(plot_rect.right(), plot_rect.bottom());
    fill_path.closeSubpath();

    painter.fillPath(fill_path, fill_colors[channel]);
    painter.setPen(QPen(line_colors[channel], 1.2));
    painter.drawPath(line_path);
  }
}

ScopeWaveformWidget::ScopeWaveformWidget(QWidget* parent) : QWidget(parent) {
  setMinimumHeight(184);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  setAutoFillBackground(false);
}

void ScopeWaveformWidget::SetPresentation(const ScopeWaveformPresentation& presentation) {
  presentation_ = presentation;
  update();
}

void ScopeWaveformWidget::paintEvent(QPaintEvent*) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setRenderHint(QPainter::TextAntialiasing, true);
  painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
  painter.fillRect(rect(), QColor(0x10, 0x10, 0x10));

  const QRect draw_rect = rect().adjusted(10, 10, -10, -10);
  painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
  painter.drawRoundedRect(draw_rect, 8, 8);

  if (!presentation_.valid || presentation_.image.isNull()) {
    painter.setPen(QColor(0x9A, 0x9A, 0x9A));
    painter.setFont(AppTheme::Font(AppTheme::FontRole::DataCaption));
    painter.drawText(draw_rect, Qt::AlignCenter, QStringLiteral("Waiting for waveform"));
    return;
  }

  painter.drawImage(draw_rect, presentation_.image);
}

}  // namespace puerhlab::ui
