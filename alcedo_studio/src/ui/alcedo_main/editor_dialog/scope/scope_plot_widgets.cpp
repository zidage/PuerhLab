//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/scope/scope_plot_widgets.hpp"

#include <QtGlobal>

#include <QPainter>
#include <QPainterPath>
#include <QPen>

#include <algorithm>

#include "ui/alcedo_main/app_theme.hpp"

namespace alcedo::ui {
namespace {

auto FormatClipBadge(const char* label, float ratio) -> QString {
  return QStringLiteral("%1 %2%").arg(QString::fromLatin1(label)).arg(qRound(ratio * 100.0f));
}

void DrawClipBadge(QPainter& painter, const QRectF& rect, const QColor& fill_color,
                   const QColor& text_color, const QString& text, Qt::Alignment alignment) {
  painter.setFont(AppTheme::Font(AppTheme::FontRole::DataCaption));
  const QRectF badge_rect =
      (alignment == Qt::AlignLeft)
          ? QRectF(rect.left() + 8.0, rect.top() + 8.0, 76.0, 24.0)
          : QRectF(rect.right() - 84.0, rect.top() + 8.0, 76.0, 24.0);

  painter.setPen(Qt::NoPen);
  painter.setBrush(fill_color);
  painter.drawRoundedRect(badge_rect, 12.0, 12.0);
  painter.setPen(text_color);
  painter.drawText(badge_rect, Qt::AlignCenter, text);
}

}  // namespace

ScopeHistogramWidget::ScopeHistogramWidget(QWidget* parent) : QWidget(parent) {
  setMinimumHeight(150);
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

  const QRectF plot_rect = rect().adjusted(10, 10, -10, -10);
  QPainterPath plot_clip;
  plot_clip.addRoundedRect(plot_rect, 8.0, 8.0);
  painter.fillPath(plot_clip, QColor(0x10, 0x10, 0x10));
  painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
  painter.drawPath(plot_clip);

  if (!presentation_.valid || presentation_.bins <= 1 ||
      presentation_.rgb.size() < static_cast<size_t>(presentation_.bins * 3)) {
    painter.setPen(QColor(0x9A, 0x9A, 0x9A));
    painter.setFont(AppTheme::Font(AppTheme::FontRole::DataCaption));
    painter.drawText(plot_rect, Qt::AlignCenter, QStringLiteral("Waiting for histogram"));
    return;
  }

  painter.save();
  painter.setClipPath(plot_clip);

  const QColor fill_colors[3] = {QColor(255, 64, 64, 76), QColor(64, 255, 96, 72),
                                 QColor(64, 128, 255, 72)};
  const QColor line_colors[3] = {QColor(255, 140, 140), QColor(140, 255, 160),
                                 QColor(128, 184, 255)};

  const int tail_bins = std::clamp(presentation_.clip_tail_bins, 0, presentation_.bins);
  if (tail_bins > 0) {
    if (presentation_.shadow_clip_warning) {
      const qreal tail_width =
          (static_cast<qreal>(tail_bins) / static_cast<qreal>(presentation_.bins)) * plot_rect.width();
      painter.fillRect(QRectF(plot_rect.left(), plot_rect.top(), tail_width, plot_rect.height()),
                       QColor(64, 170, 255, 28));
    }
    if (presentation_.highlight_clip_warning) {
      const qreal tail_width =
          (static_cast<qreal>(tail_bins) / static_cast<qreal>(presentation_.bins)) * plot_rect.width();
      painter.fillRect(
          QRectF(plot_rect.right() - tail_width, plot_rect.top(), tail_width, plot_rect.height()),
          QColor(255, 186, 64, 32));
    }
  }

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

  if (presentation_.shadow_clip_warning) {
    DrawClipBadge(painter, plot_rect, QColor(64, 170, 255, 44), QColor(210, 236, 255),
                  FormatClipBadge("BLK", presentation_.shadow_clip_ratio), Qt::AlignLeft);
  }
  if (presentation_.highlight_clip_warning) {
    DrawClipBadge(painter, plot_rect, QColor(255, 186, 64, 48), QColor(255, 241, 208),
                  FormatClipBadge("HOT", presentation_.highlight_clip_ratio), Qt::AlignRight);
  }
  painter.restore();

  painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
  painter.drawPath(plot_clip);
}

ScopeWaveformWidget::ScopeWaveformWidget(QWidget* parent) : QWidget(parent) {
  setMinimumHeight(150);
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

  const QRect draw_rect = rect().adjusted(10, 10, -10, -10);
  QPainterPath draw_clip;
  draw_clip.addRoundedRect(QRectF(draw_rect), 8.0, 8.0);
  painter.fillPath(draw_clip, QColor(0x10, 0x10, 0x10));
  painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
  painter.drawPath(draw_clip);

  if (!presentation_.valid || presentation_.image.isNull()) {
    painter.setPen(QColor(0x9A, 0x9A, 0x9A));
    painter.setFont(AppTheme::Font(AppTheme::FontRole::DataCaption));
    painter.drawText(draw_rect, Qt::AlignCenter, QStringLiteral("Waiting for waveform"));
    return;
  }

  painter.save();
  painter.setClipPath(draw_clip);
  painter.drawImage(draw_rect, presentation_.image);
  painter.restore();

  painter.setPen(QPen(QColor(0x2A, 0x2A, 0x2A), 1.0));
  painter.drawPath(draw_clip);
}

}  // namespace alcedo::ui
