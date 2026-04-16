//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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
