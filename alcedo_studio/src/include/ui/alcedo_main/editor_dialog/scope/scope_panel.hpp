//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QWidget>

#include <functional>
#include <memory>

#include "edit/scope/scope_analyzer.hpp"
#include "ui/alcedo_main/editor_dialog/scope/scope_renderer.hpp"

class QLabel;
class QPushButton;
class QStackedWidget;
class QTimer;

namespace alcedo::ui {

class ScopeHistogramWidget;
class ScopeWaveformWidget;

class ScopePanel final : public QWidget {
 public:
  explicit ScopePanel(QWidget* parent = nullptr);

  void SetAnalyzer(std::shared_ptr<IScopeAnalyzer> analyzer);
  void SetRequestChangedCallback(std::function<void(const ScopeRequest&)> callback);
  auto CurrentRequest() const -> ScopeRequest;

 private:
  enum class ScopeView : int {
    Histogram = 0,
    Waveform  = 1,
  };

  void ApplyCurrentRequest();
  void RefreshOutputs();
  void SetActiveScopeView(ScopeView view);
  void RefreshScopeSwitchUi();

  std::shared_ptr<IScopeAnalyzer>             analyzer_{};
  std::function<void(const ScopeRequest&)>    request_changed_callback_{};
  ScopeRenderer                               renderer_{};
  QTimer*                                     refresh_timer_ = nullptr;
  QLabel*                                     backend_status_label_ = nullptr;
  QPushButton*                                histogram_switch_btn_ = nullptr;
  QPushButton*                                waveform_switch_btn_ = nullptr;
  QStackedWidget*                             scope_stack_ = nullptr;
  ScopeHistogramWidget*                       histogram_widget_ = nullptr;
  ScopeWaveformWidget*                        waveform_widget_ = nullptr;
  ScopeView                                   active_scope_view_ = ScopeView::Histogram;
  uint64_t                                    last_generation_ = 0;
};

}  // namespace alcedo::ui
