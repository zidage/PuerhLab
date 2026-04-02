//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/scope/scope_panel.hpp"

#include <QCoreApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>

#include "ui/puerhlab_main/app_theme.hpp"
#include "ui/puerhlab_main/editor_dialog/scope/scope_plot_widgets.hpp"
#include "ui/puerhlab_main/editor_dialog/scope/scope_renderer.hpp"

namespace puerhlab::ui {
namespace {

auto TrScope(const char* text) -> QString {
  return QCoreApplication::translate("PuerhLab.Main", text);
}

constexpr int kDefaultHistogramBins      = 256;
constexpr int kDefaultWaveformWidth      = 384;
constexpr int kDefaultWaveformHeight     = 192;
constexpr int kDefaultAnalysisDownsample = 4;
constexpr int kDefaultTargetFps          = 20;

}  // namespace

ScopePanel::ScopePanel(QWidget* parent) : QWidget(parent) {
  auto* root = new QVBoxLayout(this);
  root->setContentsMargins(14, 12, 14, 12);
  root->setSpacing(10);

  auto* header = new QHBoxLayout();
  auto* title = new QLabel(TrScope("Scopes"), this);
  title->setObjectName("EditorSectionTitle");
  backend_status_label_ = new QLabel(TrScope("Backend unavailable"), this);
  backend_status_label_->setObjectName("EditorSectionSub");
  header->addWidget(title, 1);
  header->addWidget(backend_status_label_, 0, Qt::AlignRight);
  root->addLayout(header);

  auto* scope_switch_row = new QWidget(this);
  auto* scope_switch_layout = new QHBoxLayout(scope_switch_row);
  scope_switch_layout->setContentsMargins(0, 0, 0, 0);
  scope_switch_layout->setSpacing(8);

  histogram_switch_btn_ = new QPushButton(TrScope("Histogram"), scope_switch_row);
  waveform_switch_btn_  = new QPushButton(TrScope("Waveform"), scope_switch_row);
  histogram_switch_btn_->setCheckable(true);
  waveform_switch_btn_->setCheckable(true);
  histogram_switch_btn_->setCursor(Qt::PointingHandCursor);
  waveform_switch_btn_->setCursor(Qt::PointingHandCursor);
  histogram_switch_btn_->setFixedHeight(30);
  waveform_switch_btn_->setFixedHeight(30);
  const auto& theme = AppTheme::Instance();
  scope_switch_row->setStyleSheet(
      QStringLiteral(
          "QPushButton {"
          "  background: %1;"
          "  color: %2;"
          "  border: 1px solid %3;"
          "  border-radius: 10px;"
          "  padding: 6px 10px;"
          "}"
          "QPushButton:hover {"
          "  border-color: %4;"
          "  background: %5;"
          "  color: %6;"
          "}"
          "QPushButton:checked {"
          "  background: %7;"
          "  color: %8;"
          "  border-color: %9;"
          "  font-weight: 600;"
          "}")
          .arg(theme.bgDeepColor().name(QColor::HexArgb), theme.textMutedColor().name(QColor::HexRgb),
               theme.glassStrokeColor().name(QColor::HexArgb),
               theme.accentSecondaryColor().name(QColor::HexRgb),
               theme.hoverColor().name(QColor::HexArgb), theme.textColor().name(QColor::HexRgb),
               QColor(theme.accentColor().red(), theme.accentColor().green(),
                      theme.accentColor().blue(), 224)
                   .name(QColor::HexArgb),
               theme.bgCanvasColor().name(QColor::HexRgb),
               theme.accentSecondaryColor().name(QColor::HexRgb)));
  scope_switch_layout->addWidget(histogram_switch_btn_, 1);
  scope_switch_layout->addWidget(waveform_switch_btn_, 1);
  root->addWidget(scope_switch_row, 0);

  scope_stack_ = new QStackedWidget(this);
  histogram_widget_ = new ScopeHistogramWidget(scope_stack_);
  waveform_widget_  = new ScopeWaveformWidget(scope_stack_);
  scope_stack_->addWidget(histogram_widget_);
  scope_stack_->addWidget(waveform_widget_);
  root->addWidget(scope_stack_, 1);

  refresh_timer_ = new QTimer(this);
  refresh_timer_->setInterval(50);
  QObject::connect(refresh_timer_, &QTimer::timeout, this, [this]() { RefreshOutputs(); });
  refresh_timer_->start();

  QObject::connect(histogram_switch_btn_, &QPushButton::clicked, this,
                   [this]() { SetActiveScopeView(ScopeView::Histogram); });
  QObject::connect(waveform_switch_btn_, &QPushButton::clicked, this,
                   [this]() { SetActiveScopeView(ScopeView::Waveform); });

  RefreshScopeSwitchUi();
  SetActiveScopeView(ScopeView::Histogram);
}

void ScopePanel::SetAnalyzer(std::shared_ptr<IScopeAnalyzer> analyzer) {
  analyzer_ = std::move(analyzer);
  last_generation_ = 0;
  backend_status_label_->setText(analyzer_ ? TrScope("Live") : TrScope("Backend unavailable"));
  ApplyCurrentRequest();
}

void ScopePanel::SetRequestChangedCallback(std::function<void(const ScopeRequest&)> callback) {
  request_changed_callback_ = std::move(callback);
  ApplyCurrentRequest();
}

auto ScopePanel::CurrentRequest() const -> ScopeRequest {
  ScopeRequest request;
  request.enabled_mask = 0U;
  if (active_scope_view_ == ScopeView::Histogram) {
    request.enabled_mask |= static_cast<uint32_t>(ScopeType::Histogram);
  } else if (active_scope_view_ == ScopeView::Waveform) {
    request.enabled_mask |= static_cast<uint32_t>(ScopeType::Waveform);
  }
  request.histogram_bins      = kDefaultHistogramBins;
  request.waveform_width      = kDefaultWaveformWidth;
  request.waveform_height     = kDefaultWaveformHeight;
  request.analysis_downsample = kDefaultAnalysisDownsample;
  request.target_fps          = kDefaultTargetFps;
  return request;
}

void ScopePanel::ApplyCurrentRequest() {
  const ScopeRequest request = CurrentRequest();
  if (analyzer_) {
    analyzer_->ResizeResources(request);
  }
  if (request_changed_callback_) {
    request_changed_callback_(request);
  }
}

void ScopePanel::RefreshOutputs() {
  if (!analyzer_) {
    return;
  }

  const ScopeOutputSet output = analyzer_->GetLatestOutput();
  if (output.generation == 0 || output.generation == last_generation_) {
    return;
  }

  const ScopePresentation presentation = renderer_.Render(output);
  last_generation_                     = presentation.generation;
  if (histogram_widget_) {
    histogram_widget_->SetPresentation(presentation.histogram);
  }
  if (waveform_widget_) {
    waveform_widget_->SetPresentation(presentation.waveform);
  }
}

void ScopePanel::SetActiveScopeView(ScopeView view) {
  if (active_scope_view_ == view && scope_stack_) {
    RefreshScopeSwitchUi();
    return;
  }

  active_scope_view_ = view;
  if (scope_stack_) {
    scope_stack_->setCurrentIndex(active_scope_view_ == ScopeView::Histogram ? 0 : 1);
  }
  RefreshScopeSwitchUi();
  ApplyCurrentRequest();
}

void ScopePanel::RefreshScopeSwitchUi() {
  if (histogram_switch_btn_) {
    histogram_switch_btn_->setChecked(active_scope_view_ == ScopeView::Histogram);
  }
  if (waveform_switch_btn_) {
    waveform_switch_btn_->setChecked(active_scope_view_ == ScopeView::Waveform);
  }
}

}  // namespace puerhlab::ui
