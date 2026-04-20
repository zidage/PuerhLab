//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"
#include "ui/alcedo_main/editor_dialog/editor_slider_styling.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/look_control_panel_widget.hpp"

namespace alcedo::ui {

LookControlPanelWidget::LookControlPanelWidget(QWidget* parent) : QWidget(parent) {}

void EditorDialog::BuildLookControlPanel(EditorControlPanelWidget* controls_panel,
                                         const QString&             scroll_style) {
  look_controls_scroll_ = new QScrollArea(controls_panel);
  look_controls_scroll_->setFrameShape(QFrame::NoFrame);
  look_controls_scroll_->setWidgetResizable(true);
  look_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  look_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  look_controls_scroll_->setStyleSheet(scroll_style);

  look_controls_ = new LookControlPanelWidget(look_controls_scroll_);
  look_controls_->setMinimumWidth(0);
  look_controls_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  look_controls_layout_ = new QVBoxLayout(look_controls_);
  look_controls_layout_->setContentsMargins(10, 8, 10, 10);
  look_controls_layout_->setSpacing(8);

  // ===== Header =====
  auto* header = new QLabel(Tr("Color"), look_controls_);
  header->setObjectName("SectionTitle");
  header->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
  AppTheme::MarkFontRole(header, AppTheme::FontRole::UiHeadline);
  look_controls_layout_->addWidget(header, 0);

  // Trailing stretch — all subsequent sections insert before this.
  look_controls_layout_->addStretch();

  // ===== Shared helpers: section header, compact slider row. =====
  auto addSection = [this](const QString& title, const QString& subtitle) {
    auto* frame = new QWidget(look_controls_);
    auto* v     = new QVBoxLayout(frame);
    v->setContentsMargins(0, 8, 0, 2);
    v->setSpacing(4);

    auto* header_row    = new QWidget(frame);
    auto* header_layout = new QHBoxLayout(header_row);
    header_layout->setContentsMargins(0, 0, 0, 0);
    header_layout->setSpacing(6);

    auto* t = new QLabel(title.toUpper(), header_row);
    t->setObjectName("EditorSectionTitle");
    t->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
    AppTheme::MarkFontRole(t, AppTheme::FontRole::UiOverline);
    if (!subtitle.isEmpty()) {
      t->setToolTip(subtitle);
    }
    header_layout->addWidget(t, 0);
    header_layout->addStretch(1);

    auto* divider = new QFrame(frame);
    divider->setFrameShape(QFrame::HLine);
    divider->setFixedHeight(1);
    divider->setStyleSheet(
        QStringLiteral("QFrame { background: %1; border: none; }")
            .arg(WithAlpha(AppTheme::Instance().dividerColor(), 110).name(QColor::HexArgb)));

    v->addWidget(header_row, 0);
    v->addWidget(divider, 0);
    look_controls_layout_->insertWidget(look_controls_layout_->count() - 1, frame, 0);
  };

  const QString value_chip_style =
      QStringLiteral("QLabel {"
                     "  color: %1;"
                     "  background: transparent;"
                     "  border: none;"
                     "  padding: 0;"
                     "}")
          .arg(AppTheme::Instance().textMutedColor().name(QColor::HexRgb));

  auto addSlider = [this, value_chip_style](
                       const QString& name, int min, int max, int value, auto&& onChange,
                       auto&& onRelease, auto&& onReset, auto&& formatter) {
    auto* name_label = new QLabel(name, look_controls_);
    name_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    AppTheme::MarkFontRole(name_label, AppTheme::FontRole::UiCaption);
    name_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    auto* value_chip = new QLabel(formatter(value), look_controls_);
    value_chip->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    value_chip->setMinimumWidth(40);
    value_chip->setMaximumWidth(72);
    value_chip->setFixedHeight(16);
    value_chip->setStyleSheet(value_chip_style);
    AppTheme::MarkFontRole(value_chip, AppTheme::FontRole::DataCaption);

    auto* slider = new AccentBalanceSlider(kRegularSliderMetrics, look_controls_);
    slider->setRange(min, max);
    slider->setValue(value);
    slider->setSingleStep(1);
    slider->setPageStep(std::max(1, (max - min) / 20));
    slider->setMinimumWidth(0);
    slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    slider->setFixedHeight(22);

    QObject::connect(slider, &QSlider::valueChanged, look_controls_,
                     [this, value_chip, formatter,
                      onChange = std::forward<decltype(onChange)>(onChange)](int v) {
                       value_chip->setText(formatter(v));
                       if (syncing_controls_) {
                         return;
                       }
                       onChange(v);
                     });
    QObject::connect(slider, &QSlider::sliderReleased, look_controls_,
                     [this, onRelease = std::forward<decltype(onRelease)>(onRelease)]() {
                       if (syncing_controls_) {
                         return;
                       }
                       onRelease();
                     });
    RegisterSliderReset(slider,
                        [this, onReset = std::forward<decltype(onReset)>(onReset)]() mutable {
                          if (syncing_controls_) {
                            return;
                          }
                          onReset();
                        });

    auto* head_row    = new QWidget(look_controls_);
    auto* head_layout = new QHBoxLayout(head_row);
    head_layout->setContentsMargins(0, 0, 0, 0);
    head_layout->setSpacing(8);
    head_layout->addWidget(name_label, 1);
    head_layout->addWidget(value_chip, 0, Qt::AlignRight | Qt::AlignVCenter);

    auto* row        = new QWidget(look_controls_);
    auto* row_layout = new QVBoxLayout(row);
    row_layout->setContentsMargins(0, 0, 0, 0);
    row_layout->setSpacing(2);
    row_layout->addWidget(head_row, 0);
    row_layout->addWidget(slider, 0);

    look_controls_layout_->insertWidget(look_controls_layout_->count() - 1, row, 0);
    return slider;
  };

  // ===== LUT section =====
  addSection(Tr("LUT"), Tr("Browse and apply look-up tables."));
  lut_browser_widget_ = new LutBrowserWidget(look_controls_);
  look_controls_layout_->insertWidget(look_controls_layout_->count() - 1, lut_browser_widget_, 0);

  // ===== HSL / Color section =====
  addSection(Tr("HSL / Color"), Tr("Per-hue lightness and saturation adjustments."));
  {
    auto* frame  = new QWidget(look_controls_);
    auto* layout = new QVBoxLayout(frame);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);

    hls_target_label_ = new QLabel(frame);
    hls_target_label_->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    AppTheme::MarkFontRole(hls_target_label_, AppTheme::FontRole::UiCaption);
    layout->addWidget(hls_target_label_, 0);

    auto* swatch_row        = new QWidget(frame);
    auto* swatch_row_layout = new QHBoxLayout(swatch_row);
    swatch_row_layout->setContentsMargins(0, 0, 0, 0);
    swatch_row_layout->setSpacing(6);

    hls_candidate_buttons_.clear();
    hls_candidate_buttons_.reserve(kHlsCandidateHues.size());
    for (int i = 0; i < static_cast<int>(kHlsCandidateHues.size()); ++i) {
      auto* btn = new QPushButton(swatch_row);
      btn->setFixedSize(20, 20);
      btn->setCursor(Qt::PointingHandCursor);
      btn->setToolTip(
          Tr("Hue %1 deg").arg(kHlsCandidateHues[static_cast<size_t>(i)], 0, 'f', 0));
      QObject::connect(btn, &QPushButton::clicked, this, [this, i]() {
        if (syncing_controls_) {
          return;
        }
        SaveActiveHlsProfile(state_);
        state_.hls_target_hue_ = kHlsCandidateHues[static_cast<size_t>(i)];
        LoadActiveHlsProfile(state_);
        SyncControlsFromState();
      });
      hls_candidate_buttons_.push_back(btn);
      swatch_row_layout->addWidget(btn);
    }
    swatch_row_layout->addStretch();
    layout->addWidget(swatch_row, 0);

    look_controls_layout_->insertWidget(look_controls_layout_->count() - 1, frame, 0);
    RefreshHlsTargetUi();
  }

  hls_hue_adjust_slider_ = addSlider(
      Tr("Hue Shift"), -15, 15, static_cast<int>(std::lround(state_.hls_hue_adjust_)),
      [this](int v) {
        state_.hls_hue_adjust_ =
            std::clamp(static_cast<float>(v), -kHlsMaxHueShiftDegrees, kHlsMaxHueShiftDegrees);
        SaveActiveHlsProfile(state_);
        RequestRender();
      },
      [this]() { CommitAdjustment(AdjustmentField::Hls); },
      [this]() {
        ResetFieldToDefault(AdjustmentField::Hls, [this](const AdjustmentState& defaults) {
          state_.hls_hue_adjust_ = defaults.hls_hue_adjust_;
          SaveActiveHlsProfile(state_);
        });
      },
      [](int v) { return QString("%1 deg").arg(v); });

  hls_lightness_adjust_slider_ = addSlider(
      Tr("Lightness"), -100, 100, static_cast<int>(std::lround(state_.hls_lightness_adjust_)),
      [this](int v) {
        state_.hls_lightness_adjust_ =
            std::clamp(static_cast<float>(v), kHlsAdjUiMin, kHlsAdjUiMax);
        SaveActiveHlsProfile(state_);
        RequestRender();
      },
      [this]() { CommitAdjustment(AdjustmentField::Hls); },
      [this]() {
        ResetFieldToDefault(AdjustmentField::Hls, [this](const AdjustmentState& defaults) {
          state_.hls_lightness_adjust_ = defaults.hls_lightness_adjust_;
          SaveActiveHlsProfile(state_);
        });
      },
      [](int v) { return QString::number(v, 'f', 0); });

  hls_saturation_adjust_slider_ = addSlider(
      Tr("HSL Saturation"), -100, 100,
      static_cast<int>(std::lround(state_.hls_saturation_adjust_)),
      [this](int v) {
        state_.hls_saturation_adjust_ =
            std::clamp(static_cast<float>(v), kHlsAdjUiMin, kHlsAdjUiMax);
        SaveActiveHlsProfile(state_);
        RequestRender();
      },
      [this]() { CommitAdjustment(AdjustmentField::Hls); },
      [this]() {
        ResetFieldToDefault(AdjustmentField::Hls, [this](const AdjustmentState& defaults) {
          state_.hls_saturation_adjust_ = defaults.hls_saturation_adjust_;
          SaveActiveHlsProfile(state_);
        });
      },
      [](int v) { return QString::number(v, 'f', 0); });

  hls_hue_range_slider_ = addSlider(
      Tr("Hue Range"), 1, 180, static_cast<int>(std::lround(state_.hls_hue_range_)),
      [this](int v) {
        state_.hls_hue_range_ = static_cast<float>(v);
        SaveActiveHlsProfile(state_);
        RequestRender();
      },
      [this]() { CommitAdjustment(AdjustmentField::Hls); },
      [this]() {
        ResetFieldToDefault(AdjustmentField::Hls, [this](const AdjustmentState& defaults) {
          state_.hls_hue_range_ = defaults.hls_hue_range_;
          SaveActiveHlsProfile(state_);
        });
      },
      [](int v) { return QString("%1 deg").arg(v); });

  // ===== CDL Wheels section (triangle layout: Gamma on top, Lift + Gain below) =====
  addSection(Tr("Color Wheels"), Tr("CDL: Lift / Gamma / Gain with master offset."));
  {
    auto* wheel_frame = new QWidget(look_controls_);
    wheel_frame->setMinimumWidth(0);
    wheel_frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    auto* wheel_layout = new QVBoxLayout(wheel_frame);
    wheel_layout->setContentsMargins(0, 4, 0, 0);
    wheel_layout->setSpacing(10);

    auto makeWheelUnit = [this, wheel_frame](
                             const QString& title, CdlWheelState& wheel_state, bool add_unity,
                             bool invert_delta, CdlTrackballDiscWidget*& disc_widget,
                             QLabel*& offset_label, QSlider*& slider_widget) -> QWidget* {
      auto* unit = new QWidget(wheel_frame);
      unit->setMinimumWidth(0);
      unit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
      auto* unit_layout = new QVBoxLayout(unit);
      unit_layout->setContentsMargins(0, 0, 0, 0);
      unit_layout->setSpacing(4);

      auto* title_label = new QLabel(title, unit);
      title_label->setStyleSheet(AppTheme::EditorLabelStyle(QColor(0xCF, 0xCF, 0xCF)));
      AppTheme::MarkFontRole(title_label, AppTheme::FontRole::UiOverline);
      unit_layout->addWidget(title_label, 0, Qt::AlignHCenter);

      disc_widget = new CdlTrackballDiscWidget(unit);
      disc_widget->setMinimumSize(128, 128);
      disc_widget->setMaximumSize(180, 180);
      disc_widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
      disc_widget->SetPosition(wheel_state.disc_position_);
      disc_widget->SetPositionChangedCallback(
          [this, &wheel_state, add_unity, invert_delta](const QPointF& pos) {
            if (syncing_controls_) {
              return;
            }
            wheel_state.disc_position_ = ClampDiscPoint(pos);
            UpdateCdlWheelDerivedColor(wheel_state, add_unity, invert_delta);
            RefreshCdlOffsetLabels();
            RequestRender();
          });
      disc_widget->SetPositionReleasedCallback(
          [this, &wheel_state, add_unity, invert_delta](const QPointF& pos) {
            if (syncing_controls_) {
              return;
            }
            wheel_state.disc_position_ = ClampDiscPoint(pos);
            UpdateCdlWheelDerivedColor(wheel_state, add_unity, invert_delta);
            RefreshCdlOffsetLabels();
            RequestRender();
            CommitAdjustment(AdjustmentField::ColorWheel);
          });
      unit_layout->addWidget(disc_widget, 0, Qt::AlignHCenter);

      offset_label = new QLabel(FormatWheelDeltaText(wheel_state, add_unity), unit);
      offset_label->setStyleSheet(AppTheme::EditorLabelStyle(QColor(0xA9, 0xA9, 0xA9)));
      AppTheme::MarkFontRole(offset_label, AppTheme::FontRole::DataCaption);
      offset_label->setAlignment(Qt::AlignHCenter);
      unit_layout->addWidget(offset_label, 0);

      slider_widget = new AccentBalanceSlider(kCompactSliderMetrics, unit);
      slider_widget->setRange(kCdlWheelSliderUiMin, kCdlWheelSliderUiMax);
      const float sign = invert_delta ? -1.0f : 1.0f;
      slider_widget->setValue(CdlMasterToSliderUi(wheel_state.master_offset_ * sign));
      slider_widget->setSingleStep(1);
      slider_widget->setPageStep(100);
      slider_widget->setFixedHeight(14);
      slider_widget->setMinimumWidth(0);
      slider_widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
      QObject::connect(slider_widget, &QSlider::valueChanged, unit,
                       [this, &wheel_state, sign](int value) {
                         if (syncing_controls_) {
                           return;
                         }
                         wheel_state.master_offset_ = CdlSliderUiToMaster(value) * sign;
                         RefreshCdlOffsetLabels();
                         RequestRender();
                       });
      QObject::connect(slider_widget, &QSlider::sliderReleased, unit, [this]() {
        if (syncing_controls_) {
          return;
        }
        CommitAdjustment(AdjustmentField::ColorWheel);
      });
      RegisterSliderReset(slider_widget, [this, &wheel_state]() {
        wheel_state.master_offset_ = 0.0f;
        SyncControlsFromState();
        RequestRender();
        CommitAdjustment(AdjustmentField::ColorWheel);
      });
      unit_layout->addWidget(slider_widget, 0);

      return unit;
    };

    auto* gamma_unit =
        makeWheelUnit(Tr("Gamma"), state_.gamma_wheel_, true, true, gamma_disc_widget_,
                      gamma_offset_label_, gamma_master_slider_);
    auto* lift_unit =
        makeWheelUnit(Tr("Lift"), state_.lift_wheel_, false, false, lift_disc_widget_,
                      lift_offset_label_, lift_master_slider_);
    auto* gain_unit =
        makeWheelUnit(Tr("Gain"), state_.gain_wheel_, true, false, gain_disc_widget_,
                      gain_offset_label_, gain_master_slider_);

    // Triangle layout: Gamma centered on top, Lift and Gain on the bottom.
    auto* top_row        = new QWidget(wheel_frame);
    auto* top_row_layout = new QHBoxLayout(top_row);
    top_row_layout->setContentsMargins(0, 0, 0, 0);
    top_row_layout->setSpacing(0);
    top_row_layout->addStretch(1);
    top_row_layout->addWidget(gamma_unit, 2);
    top_row_layout->addStretch(1);
    wheel_layout->addWidget(top_row, 0);

    auto* bottom_row        = new QWidget(wheel_frame);
    auto* bottom_row_layout = new QHBoxLayout(bottom_row);
    bottom_row_layout->setContentsMargins(0, 0, 0, 0);
    bottom_row_layout->setSpacing(10);
    bottom_row_layout->addWidget(lift_unit, 1);
    bottom_row_layout->addWidget(gain_unit, 1);
    wheel_layout->addWidget(bottom_row, 0);

    look_controls_layout_->insertWidget(look_controls_layout_->count() - 1, wheel_frame, 0);
    RefreshCdlOffsetLabels();
  }

  look_controls_scroll_->setWidget(look_controls_);
}

void EditorDialog::WireLookControlPanel() {
  if (!lut_browser_widget_) {
    return;
  }

  const auto lut_view_model = lut_controller_.Refresh(state_.lut_path_);
  lut_browser_widget_->SetDirectoryInfo(lut_view_model.directory_text_, lut_view_model.status_text_,
                                        lut_view_model.can_open_directory_);
  lut_browser_widget_->SetEntries(lut_view_model.entries_, lut_view_model.selected_path_);

  QObject::connect(lut_browser_widget_, &LutBrowserWidget::RefreshRequested, this,
                   [this]() { ForceRefreshLutBrowserUi(); });
  QObject::connect(lut_browser_widget_, &LutBrowserWidget::OpenFolderRequested, this,
                   [this]() { OpenLutFolder(); });
  QObject::connect(lut_browser_widget_, &LutBrowserWidget::LutPathActivated, this,
                   [this](const QString& entry_path) {
                     const auto resolved_path = lut_controller_.TryResolveSelection(entry_path);
                     if (!resolved_path.has_value() || *resolved_path == state_.lut_path_) {
                       return;
                     }
                     state_.lut_path_ = *resolved_path;
                     CommitAdjustment(AdjustmentField::Lut);
                   });
  RefreshLutBrowserUi();
}

}  // namespace alcedo::ui
