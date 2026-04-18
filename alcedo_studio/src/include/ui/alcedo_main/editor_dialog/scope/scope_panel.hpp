//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QWidget>

#include <array>
#include <functional>
#include <memory>

#include "edit/scope/scope_analyzer.hpp"
#include "image/metadata.hpp"
#include "ui/alcedo_main/editor_dialog/scope/scope_renderer.hpp"

class QComboBox;
class QLabel;
class QStackedWidget;
class QTimer;

namespace alcedo::ui {

class ScopeHistogramWidget;
class ScopeWaveformWidget;

class ScopePanel final : public QWidget {
 public:
  explicit ScopePanel(QWidget* parent = nullptr);

  void SetAnalyzer(std::shared_ptr<IScopeAnalyzer> analyzer);
  void SetExifDisplayMetaData(const ExifDisplayMetaData& metadata);
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
  void RefreshExifUi();

  std::shared_ptr<IScopeAnalyzer>             analyzer_{};
  ExifDisplayMetaData                         exif_display_{};
  std::function<void(const ScopeRequest&)>    request_changed_callback_{};
  ScopeRenderer                               renderer_{};
  QTimer*                                     refresh_timer_ = nullptr;
  QComboBox*                                  scope_type_combo_ = nullptr;
  QStackedWidget*                             scope_stack_ = nullptr;
  ScopeHistogramWidget*                       histogram_widget_ = nullptr;
  ScopeWaveformWidget*                        waveform_widget_ = nullptr;
  std::array<QLabel*, 4>                      exif_value_labels_{};
  ScopeView                                   active_scope_view_ = ScopeView::Histogram;
  uint64_t                                    last_generation_ = 0;
};

}  // namespace alcedo::ui
