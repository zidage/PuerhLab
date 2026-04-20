//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/scope/scope_panel.hpp"

#include <QComboBox>
#include <QCoreApplication>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QSignalBlocker>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>

#include <cmath>

#include "ui/alcedo_main/app_theme.hpp"
#include "ui/alcedo_main/editor_dialog/scope/scope_plot_widgets.hpp"
#include "ui/alcedo_main/editor_dialog/scope/scope_renderer.hpp"

namespace alcedo::ui {
namespace {

auto TrScope(const char* text) -> QString {
  return QCoreApplication::translate("Alcedo.Main", text);
}

constexpr int kDefaultHistogramBins      = 256;
constexpr int kDefaultWaveformWidth      = 384;
constexpr int kDefaultWaveformHeight     = 192;
constexpr int kDefaultAnalysisDownsample = 4;
constexpr int kDefaultTargetFps          = 20;
constexpr int kScopePlotInset            = 6;

auto FormatCompactFloat(float value) -> QString {
  const float rounded = std::round(value);
  if (std::fabs(value - rounded) < 0.05f) {
    return QString::number(static_cast<int>(rounded));
  }
  return QString::number(value, 'f', 1);
}

auto FormatExifIso(const ExifDisplayMetaData& metadata) -> QString {
  if (metadata.iso_ == 0) {
    return QStringLiteral("ISO --");
  }
  return QStringLiteral("ISO %1").arg(static_cast<qulonglong>(metadata.iso_));
}

auto FormatExifFocalLength(const ExifDisplayMetaData& metadata) -> QString {
  if (!std::isfinite(metadata.focal_) || metadata.focal_ <= 0.0f) {
    return QStringLiteral("--");
  }
  return QStringLiteral("%1mm").arg(FormatCompactFloat(metadata.focal_));
}

auto FormatExifAperture(const ExifDisplayMetaData& metadata) -> QString {
  if (!std::isfinite(metadata.aperture_) || metadata.aperture_ <= 0.0f) {
    return QStringLiteral("f/--");
  }
  return QStringLiteral("f/%1").arg(FormatCompactFloat(metadata.aperture_));
}

auto FormatExifShutter(const ExifDisplayMetaData& metadata) -> QString {
  const int numerator   = metadata.shutter_speed_.first;
  const int denominator = metadata.shutter_speed_.second;
  if (numerator <= 0 || denominator <= 0) {
    return QStringLiteral("--");
  }
  if (denominator == 1) {
    return QStringLiteral("%1s").arg(numerator);
  }
  if (numerator == 1) {
    return QStringLiteral("1/%1s").arg(denominator);
  }
  return QStringLiteral("%1/%2s").arg(numerator).arg(denominator);
}

}  // namespace

ScopePanel::ScopePanel(QWidget* parent) : QWidget(parent) {
  const auto& theme = AppTheme::Instance();

  auto* root = new QVBoxLayout(this);
  root->setContentsMargins(10, 8, 10, 6);
  root->setSpacing(6);

  auto* header_row        = new QWidget(this);
  auto* header_row_layout = new QHBoxLayout(header_row);
  header_row_layout->setContentsMargins(0, 0, 0, 0);
  header_row_layout->setSpacing(8);

  auto* title = new QLabel(TrScope("Scope"), header_row);
  title->setObjectName("EditorSectionTitle");
  header_row_layout->addWidget(title, 0);
  header_row_layout->addStretch(1);

  scope_type_combo_ = new QComboBox(header_row);
  scope_type_combo_->addItem(TrScope("Histogram"));
  scope_type_combo_->addItem(TrScope("Waveform"));
  scope_type_combo_->setCursor(Qt::PointingHandCursor);
  scope_type_combo_->setFixedHeight(26);
  scope_type_combo_->setMinimumWidth(120);
  scope_type_combo_->setMaximumWidth(180);
  scope_type_combo_->setStyleSheet(
      QStringLiteral(
          "QComboBox {"
          "  background: %1;"
          "  color: %2;"
          "  border: none;"
          "  border-radius: 6px;"
          "  padding: 0 10px;"
          "}"
          "QComboBox:hover {"
          "  background: %3;"
          "}"
          "QComboBox::drop-down {"
          "  border: 0px;"
          "  width: 22px;"
          "}"
          "QComboBox QAbstractItemView {"
          "  background: %4;"
          "  color: %2;"
          "  border: 1px solid %5;"
          "  selection-background-color: %6;"
          "  selection-color: %7;"
          "  outline: 0px;"
          "}"
          "QComboBox QAbstractItemView::item {"
          "  min-height: 28px;"
          "}"
          "QComboBox QAbstractItemView::item:hover {"
          "  background: %8;"
          "}"
          "QComboBox QAbstractItemView::item:selected {"
          "  background: %6;"
          "  color: %7;"
          "}")
          .arg(QColor(0x38, 0x38, 0x38).name(QColor::HexRgb), theme.textColor().name(QColor::HexRgb),
               QColor(0x42, 0x42, 0x42).name(QColor::HexRgb),
               QColor(0x2A, 0x2A, 0x2A).name(QColor::HexRgb),
               theme.glassStrokeColor().name(QColor::HexArgb),
               QColor(theme.accentColor().red(), theme.accentColor().green(),
                      theme.accentColor().blue(), 224)
                   .name(QColor::HexArgb),
               theme.bgCanvasColor().name(QColor::HexRgb),
               QColor(0x36, 0x36, 0x36).name(QColor::HexRgb)));
  AppTheme::MarkFontRole(scope_type_combo_, AppTheme::FontRole::UiCaptionStrong);
  header_row_layout->addWidget(scope_type_combo_, 0, Qt::AlignRight | Qt::AlignVCenter);
  root->addWidget(header_row, 0);

  scope_stack_ = new QStackedWidget(this);
  scope_stack_->setFrameShape(QFrame::NoFrame);
  scope_stack_->setLineWidth(0);
  scope_stack_->setAttribute(Qt::WA_StyledBackground, true);
  scope_stack_->setStyleSheet(QStringLiteral(
      "QStackedWidget {"
      "  background: transparent;"
      "  border: none;"
      "}"));
  histogram_widget_ = new ScopeHistogramWidget(scope_stack_);
  waveform_widget_  = new ScopeWaveformWidget(scope_stack_);
  scope_stack_->addWidget(histogram_widget_);
  scope_stack_->addWidget(waveform_widget_);
  root->addWidget(scope_stack_, 1);

  auto* exif_row_host = new QWidget(this);
  auto* exif_host_layout = new QVBoxLayout(exif_row_host);
  exif_host_layout->setContentsMargins(kScopePlotInset, 0, kScopePlotInset, 0);
  exif_host_layout->setSpacing(0);

  auto* exif_row = new QWidget(exif_row_host);
  exif_row->setStyleSheet(QStringLiteral(
      "QWidget { background: transparent; }"
      "QLabel { color: %1; }")
                              .arg(theme.textMutedColor().name(QColor::HexRgb)));
  auto* exif_layout = new QHBoxLayout(exif_row);
  exif_layout->setContentsMargins(0, 2, 0, 0);
  exif_layout->setSpacing(0);
  constexpr int        kExifColumnStretch[4] = {1, 1, 1, 1};
  constexpr Qt::Alignment kExifAlignment[4]  = {Qt::AlignLeft | Qt::AlignVCenter,
                                               Qt::AlignLeft | Qt::AlignVCenter,
                                               Qt::AlignLeft | Qt::AlignVCenter,
                                               Qt::AlignRight | Qt::AlignVCenter};
  for (size_t i = 0; i < exif_value_labels_.size(); ++i) {
    QLabel*& label = exif_value_labels_[i];
    label = new QLabel(QStringLiteral("--"), exif_row);
    label->setAlignment(kExifAlignment[i]);
    label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    AppTheme::MarkFontRole(label, AppTheme::FontRole::DataCaption);
    exif_layout->addWidget(label, kExifColumnStretch[i], kExifAlignment[i]);
  }
  exif_host_layout->addWidget(exif_row, 0);
  root->addWidget(exif_row_host, 0);

  refresh_timer_ = new QTimer(this);
  refresh_timer_->setInterval(50);
  QObject::connect(refresh_timer_, &QTimer::timeout, this, [this]() { RefreshOutputs(); });
  refresh_timer_->start();

  QObject::connect(scope_type_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                   [this](int index) {
                     SetActiveScopeView(index == 0 ? ScopeView::Histogram : ScopeView::Waveform);
                   });

  RefreshExifUi();
  RefreshScopeSwitchUi();
  SetActiveScopeView(ScopeView::Histogram);
}

void ScopePanel::SetAnalyzer(std::shared_ptr<IScopeAnalyzer> analyzer) {
  analyzer_ = std::move(analyzer);
  last_generation_ = 0;
  ApplyCurrentRequest();
}

void ScopePanel::SetExifDisplayMetaData(const ExifDisplayMetaData& metadata) {
  exif_display_ = metadata;
  RefreshExifUi();
}

void ScopePanel::SetRequestChangedCallback(std::function<void(const ScopeRequest&)> callback) {
  request_changed_callback_ = std::move(callback);
  ApplyCurrentRequest();
}

void ScopePanel::SetNeedsRenderCallback(std::function<void()> callback) {
  needs_render_callback_ = std::move(callback);
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

  if (needs_render_callback_) {
    needs_render_callback_();
  }
}

void ScopePanel::RefreshScopeSwitchUi() {
  if (scope_type_combo_) {
    const QSignalBlocker blocker(scope_type_combo_);
    scope_type_combo_->setCurrentIndex(active_scope_view_ == ScopeView::Histogram ? 0 : 1);
  }
}

void ScopePanel::RefreshExifUi() {
  if (exif_value_labels_.size() < 4) {
    return;
  }

  exif_value_labels_[0]->setText(FormatExifIso(exif_display_));
  exif_value_labels_[1]->setText(FormatExifFocalLength(exif_display_));
  exif_value_labels_[2]->setText(FormatExifAperture(exif_display_));
  exif_value_labels_[3]->setText(FormatExifShutter(exif_display_));
  for (QLabel* label : exif_value_labels_) {
    if (!label) {
      continue;
    }
    label->setToolTip(label->text());
  }
}

}  // namespace alcedo::ui
