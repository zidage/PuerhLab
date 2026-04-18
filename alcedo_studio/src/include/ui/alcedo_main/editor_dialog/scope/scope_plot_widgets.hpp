//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QWidget>

#include "ui/alcedo_main/editor_dialog/scope/scope_renderer.hpp"

namespace alcedo::ui {

class ScopeHistogramWidget final : public QWidget {
 public:
  explicit ScopeHistogramWidget(QWidget* parent = nullptr);

  void SetPresentation(const ScopeHistogramPresentation& presentation);

 protected:
  void paintEvent(QPaintEvent*) override;

 private:
  ScopeHistogramPresentation presentation_{};
};

class ScopeWaveformWidget final : public QWidget {
 public:
  explicit ScopeWaveformWidget(QWidget* parent = nullptr);

  void SetPresentation(const ScopeWaveformPresentation& presentation);

 protected:
  void paintEvent(QPaintEvent*) override;

 private:
  ScopeWaveformPresentation presentation_{};
};

}  // namespace alcedo::ui
