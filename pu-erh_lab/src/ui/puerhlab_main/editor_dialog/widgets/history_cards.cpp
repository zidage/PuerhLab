//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/widgets/history_cards.hpp"

#include <QFontMetrics>
#include <QHBoxLayout>
#include <QPainter>
#include <QPen>
#include <QStyle>

#include <utility>

#include "ui/puerhlab_main/app_theme.hpp"

namespace puerhlab::ui {

HistoryLaneWidget::HistoryLaneWidget(QColor dot, QColor line, bool draw_top, bool draw_bottom,
                                     bool solid_dot,
                                     QWidget* parent)
    : QWidget(parent),
      dot_(std::move(dot)),
      line_(std::move(line)),
      draw_top_(draw_top),
      draw_bottom_(draw_bottom),
      solid_dot_(solid_dot) {
  setFixedWidth(18);
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
  setAttribute(Qt::WA_TransparentForMouseEvents);
}

void HistoryLaneWidget::SetConnectors(bool draw_top, bool draw_bottom) {
  draw_top_    = draw_top;
  draw_bottom_ = draw_bottom;
  update();
}

void HistoryLaneWidget::paintEvent(QPaintEvent*) {
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  const int cx = width() / 2;
  const int cy = height() / 2;

  {
    QPen pen(line_);
    pen.setWidthF(1.5);
    pen.setCapStyle(Qt::RoundCap);
    p.setPen(pen);

    if (draw_top_) {
      p.drawLine(QPointF(cx, 2.0), QPointF(cx, cy - 6.0));
    }
    if (draw_bottom_) {
      p.drawLine(QPointF(cx, cy + 6.0), QPointF(cx, height() - 2.0));
    }
  }

  {
    if (solid_dot_) {
      p.setPen(Qt::NoPen);
      p.setBrush(dot_);
      p.drawEllipse(QPointF(cx, cy), 3.8, 3.8);
    } else {
      QPen pen(dot_);
      pen.setWidthF(1.6);
      p.setPen(pen);
      p.setBrush(Qt::NoBrush);
      p.drawEllipse(QPointF(cx, cy), 3.8, 3.8);
    }
  }
}

HistoryCardWidget::HistoryCardWidget(QWidget* parent) : QFrame(parent) {
  setObjectName("HistoryCard");
  setAttribute(Qt::WA_StyledBackground, true);
  setAttribute(Qt::WA_Hover, true);
  setProperty("selected", false);

  setStyleSheet(AppTheme::EditorHistoryCardStyle());
}

void HistoryCardWidget::SetSelected(bool selected) {
  if (property("selected").toBool() == selected) {
    return;
  }
  setProperty("selected", selected);
  style()->unpolish(this);
  style()->polish(this);
  update();
}

ElidedLabel::ElidedLabel(const QString& text, QWidget* parent) : QLabel(parent) {
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  setTextFormat(Qt::PlainText);
  SetRawText(text);
}

void ElidedLabel::SetRawText(const QString& text) {
  raw_text_ = text;
  UpdateElidedText();
}

void ElidedLabel::resizeEvent(QResizeEvent* event) {
  QLabel::resizeEvent(event);
  UpdateElidedText();
}

void ElidedLabel::UpdateElidedText() {
  if (raw_text_.isEmpty()) {
    QLabel::setText(QString());
    return;
  }

  const int available_width = std::max(0, contentsRect().width());
  if (available_width <= 0) {
    QLabel::setText(raw_text_);
  } else {
    const QFontMetrics metrics(font());
    QLabel::setText(metrics.elidedText(raw_text_, Qt::ElideRight, available_width));
  }
  setToolTip(raw_text_);
}

auto MakePillLabel(const QString& text, QWidget* parent) -> QLabel* {
  auto* l = new QLabel(text, parent);
  const auto& theme = AppTheme::Instance();
  QFont badge_font = AppTheme::Font(AppTheme::FontRole::UiCaptionStrong);
  badge_font.setPointSizeF(9.0);
  badge_font.setWeight(QFont::Medium);
  badge_font.setStyleStrategy(QFont::PreferAntialias);
  l->setFont(badge_font);
  l->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  l->setStyleSheet(QStringLiteral("QLabel {"
                                  "  color: %1;"
                                  "  background: %2;"
                                  "  border: 1px solid %3;"
                                  "  border-radius: 6px;"
                                  "  font-size: 9px;"
                                  "  font-weight: 500;"
                                  "  padding: 2px 6px;"
                                  "}")
                       .arg(theme.accentColor().name(QColor::HexRgb),
                            QColor(theme.accentColor().red(), theme.accentColor().green(),
                                   theme.accentColor().blue(), 32)
                                .name(QColor::HexArgb),
                            QColor(theme.accentColor().red(), theme.accentColor().green(),
                                   theme.accentColor().blue(), 72)
                                .name(QColor::HexArgb)));
  return l;
}

}  // namespace puerhlab::ui
