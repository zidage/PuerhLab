//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QColor>
#include <QFrame>
#include <QLabel>
#include <QResizeEvent>
#include <QWidget>

namespace puerhlab::ui {

class HistoryLaneWidget final : public QWidget {
 public:
  HistoryLaneWidget(QColor dot, QColor line, bool draw_top, bool draw_bottom,
                    bool solid_dot,
                    QWidget* parent = nullptr);

  void SetConnectors(bool draw_top, bool draw_bottom);

 protected:
  void paintEvent(QPaintEvent*) override;

 private:
  QColor dot_;
  QColor line_;
  bool   draw_top_    = false;
  bool   draw_bottom_ = false;
  bool   solid_dot_   = false;
};

class HistoryCardWidget final : public QFrame {
 public:
  explicit HistoryCardWidget(QWidget* parent = nullptr);

  void SetSelected(bool selected);
};

class ElidedLabel final : public QLabel {
 public:
  explicit ElidedLabel(const QString& text = QString(), QWidget* parent = nullptr);

  void SetRawText(const QString& text);
  auto RawText() const -> const QString& { return raw_text_; }

 protected:
  void resizeEvent(QResizeEvent* event) override;

 private:
  void UpdateElidedText();

  QString raw_text_{};
};

auto MakePillLabel(const QString& text, QWidget* parent) -> QLabel*;

}  // namespace puerhlab::ui
