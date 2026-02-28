#pragma once

#include <QColor>
#include <QFrame>
#include <QLabel>
#include <QWidget>

namespace puerhlab::ui {

class HistoryLaneWidget final : public QWidget {
 public:
  HistoryLaneWidget(QColor dot, QColor line, bool draw_top, bool draw_bottom,
                    QWidget* parent = nullptr);

  void SetConnectors(bool draw_top, bool draw_bottom);

 protected:
  void paintEvent(QPaintEvent*) override;

 private:
  QColor dot_;
  QColor line_;
  bool   draw_top_    = false;
  bool   draw_bottom_ = false;
};

class HistoryCardWidget final : public QFrame {
 public:
  explicit HistoryCardWidget(QWidget* parent = nullptr);

  void SetSelected(bool selected);
};

auto MakePillLabel(const QString& text, const QString& fg, const QString& bg,
                   const QString& border, QWidget* parent) -> QLabel*;

}  // namespace puerhlab::ui
