#pragma once

#include <QWidget>

class QTimer;

namespace puerhlab::ui {

class SpinnerWidget final : public QWidget {
 public:
  explicit SpinnerWidget(QWidget* parent = nullptr);

  void Start();
  void Stop();

 protected:
  void paintEvent(QPaintEvent*) override;

 private:
  QTimer* timer_     = nullptr;
  int     angle_deg_ = 0;
};

}  // namespace puerhlab::ui
