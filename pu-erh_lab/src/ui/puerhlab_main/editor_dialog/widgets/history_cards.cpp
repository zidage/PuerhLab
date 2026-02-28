#include "ui/puerhlab_main/editor_dialog/widgets/history_cards.hpp"

#include <QHBoxLayout>
#include <QPainter>
#include <QPen>
#include <QStyle>

#include <utility>

namespace puerhlab::ui {

HistoryLaneWidget::HistoryLaneWidget(QColor dot, QColor line, bool draw_top, bool draw_bottom,
                                     QWidget* parent)
    : QWidget(parent),
      dot_(std::move(dot)),
      line_(std::move(line)),
      draw_top_(draw_top),
      draw_bottom_(draw_bottom) {
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

  {
    p.setPen(Qt::NoPen);
    p.setBrush(dot_);
    p.drawEllipse(QPointF(cx, cy), 4.4, 4.4);
    p.setBrush(QColor(0x12, 0x12, 0x12));
    p.drawEllipse(QPointF(cx, cy), 2.0, 2.0);
  }
}

HistoryCardWidget::HistoryCardWidget(QWidget* parent) : QFrame(parent) {
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

void HistoryCardWidget::SetSelected(bool selected) {
  if (property("selected").toBool() == selected) {
    return;
  }
  setProperty("selected", selected);
  style()->unpolish(this);
  style()->polish(this);
  update();
}

auto MakePillLabel(const QString& text, const QString& fg, const QString& bg,
                   const QString& border, QWidget* parent) -> QLabel* {
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

}  // namespace puerhlab::ui
