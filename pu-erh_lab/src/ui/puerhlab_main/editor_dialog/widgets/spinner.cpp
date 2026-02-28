#include "ui/puerhlab_main/editor_dialog/widgets/spinner.hpp"

#include <QPainter>
#include <QPen>
#include <QTimer>

namespace puerhlab::ui {

SpinnerWidget::SpinnerWidget(QWidget* parent) : QWidget(parent) {
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

void SpinnerWidget::Start() {
  show();
  raise();
  if (!timer_->isActive()) {
    timer_->start();
  }
}

void SpinnerWidget::Stop() {
  if (timer_->isActive()) {
    timer_->stop();
  }
  hide();
}

void SpinnerWidget::paintEvent(QPaintEvent*) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);

  const QRectF r = QRectF(2.5, 2.5, width() - 5.0, height() - 5.0);

  {
    QPen pen(QColor(0x3A, 0x3A, 0x3A, 180));
    pen.setWidthF(2.0);
    pen.setCapStyle(Qt::RoundCap);
    painter.setPen(pen);
    painter.drawArc(r, 0 * 16, 360 * 16);
  }

  {
    QPen pen(QColor(0xFC, 0xC7, 0x04, 230));
    pen.setWidthF(2.2);
    pen.setCapStyle(Qt::RoundCap);
    painter.setPen(pen);
    painter.drawArc(r, (90 - angle_deg_) * 16, 100 * 16);
  }
}

}  // namespace puerhlab::ui
