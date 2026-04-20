//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"
#include "ui/alcedo_main/editor_dialog/editor_slider_styling.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/tone_control_panel_widget.hpp"

namespace alcedo::ui {

ToneControlPanelWidget::ToneControlPanelWidget(QWidget* parent) : QWidget(parent) {}

void EditorDialog::BuildToneControlPanel() {
    auto* controls_header = new QLabel(Tr("Adjustments"), controls_);
    controls_header->setObjectName("SectionTitle");
    controls_header->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    AppTheme::MarkFontRole(controls_header, AppTheme::FontRole::UiHeadline);
    controls_layout_->addWidget(controls_header, 0);

    auto addSection = [&](const QString& title, const QString& subtitle) {
      auto* frame = new QWidget(controls_);
      auto* v     = new QVBoxLayout(frame);
      v->setContentsMargins(0, 8, 0, 2);
      v->setSpacing(4);

      auto* header_row    = new QWidget(frame);
      auto* header_layout = new QHBoxLayout(header_row);
      header_layout->setContentsMargins(0, 0, 0, 0);
      header_layout->setSpacing(6);

      auto* t = new QLabel(title.toUpper(), header_row);
      t->setObjectName("EditorSectionTitle");
      t->setStyleSheet(
          AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
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
              .arg(WithAlpha(AppTheme::Instance().dividerColor(), 110)
                       .name(QColor::HexArgb)));

      v->addWidget(header_row, 0);
      v->addWidget(divider, 0);
      controls_layout_->insertWidget(controls_layout_->count() - 1, frame, 0);
    };

    controls_layout_->addStretch();

    const auto default_lut_path = lut_controller_.DefaultLutPath();

    // If the pipeline already has operator params (loaded from PipelineService/storage),
    // initialize UI state from those params rather than overwriting them.
    const bool loaded_state_from_pipeline = LoadStateFromPipelineIfPresent();
    if (!loaded_state_from_pipeline) {
      // Demo-friendly default: apply a LUT only for brand-new pipelines with no saved params.
      state_.lut_path_ = default_lut_path;
      UpdateAllCdlWheelDerivedColors(state_);
    }
    committed_state_ = state_;

    // Seed a working version from the latest committed one (if any).
    working_version_ =
        controllers::SeedWorkingVersionFromLatest(element_id_, history_guard_);
    working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);
    WireLookControlPanel();

    auto addComboBox = [&](const QString& name, const QStringList& items, int initial_index,
                           auto&& onChange) {
      auto* label = new QLabel(name, controls_);
      label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
      AppTheme::MarkFontRole(label, AppTheme::FontRole::UiBody);
      label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

      auto* combo = new QComboBox(controls_);
      combo->addItems(items);
      combo->setCurrentIndex(initial_index);
      combo->setMinimumWidth(96);
      combo->setMaximumWidth(160);
      combo->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
      combo->setFixedHeight(26);
      combo->setStyleSheet(AppTheme::EditorComboBoxStyle());

      QObject::connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), controls_,
                       [this, onChange = std::forward<decltype(onChange)>(onChange)](int idx) {
                         if (syncing_controls_) {
                           return;
                         }
                         onChange(idx);
                       });

      auto* row       = new QWidget(controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(8);
      rowLayout->addWidget(label, /*stretch*/ 1);
      rowLayout->addWidget(combo, /*stretch*/ 0, Qt::AlignRight | Qt::AlignVCenter);

      controls_layout_->insertWidget(controls_layout_->count() - 1, row);
      return combo;
    };

    const QString value_chip_style =
        QStringLiteral("QLabel {"
                       "  color: %1;"
                       "  background: transparent;"
                       "  border: none;"
                       "  padding: 0;"
                       "}")
            .arg(AppTheme::Instance().textMutedColor().name(QColor::HexRgb));

    auto addSlider = [&, value_chip_style](
                         const QString& name, int min, int max, int value, auto&& onChange,
                         auto&& onRelease, auto&& onReset, auto&& formatter,
                         SliderVisualStyle visual_style = SliderVisualStyle::Accent) {
      auto* name_label = new QLabel(name, controls_);
      name_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
      AppTheme::MarkFontRole(name_label, AppTheme::FontRole::UiCaption);
      name_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

      auto* value_chip = new QLabel(formatter(value), controls_);
      value_chip->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
      value_chip->setMinimumWidth(40);
      value_chip->setMaximumWidth(72);
      value_chip->setFixedHeight(16);
      value_chip->setStyleSheet(value_chip_style);
      AppTheme::MarkFontRole(value_chip, AppTheme::FontRole::DataCaption);

      QSlider* slider = nullptr;
      if (visual_style == SliderVisualStyle::Accent) {
        slider = new AccentBalanceSlider(kRegularSliderMetrics, controls_);
      } else {
        slider = new QSlider(Qt::Horizontal, controls_);
      }
      slider->setRange(min, max);
      slider->setValue(value);
      slider->setSingleStep(1);
      slider->setPageStep(std::max(1, (max - min) / 20));
      slider->setMinimumWidth(0);
      slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
      slider->setFixedHeight(22);

      QObject::connect(slider, &QSlider::valueChanged, controls_,
                       [this, value_chip, formatter,
                        onChange = std::forward<decltype(onChange)>(onChange)](int v) {
                         value_chip->setText(formatter(v));
                         if (syncing_controls_) {
                           return;
                         }
                         onChange(v);
                       });

      QObject::connect(slider, &QSlider::sliderReleased, controls_,
                       [this, onRelease = std::forward<decltype(onRelease)>(onRelease)]() {
                         if (syncing_controls_) {
                           return;
                         }
                         onRelease();
                       });

      RegisterSliderReset(
          slider, [this, onReset = std::forward<decltype(onReset)>(onReset)]() mutable {
            if (syncing_controls_) {
              return;
            }
            onReset();
          });

      auto* head_row    = new QWidget(controls_);
      auto* head_layout = new QHBoxLayout(head_row);
      head_layout->setContentsMargins(0, 0, 0, 0);
      head_layout->setSpacing(8);
      head_layout->addWidget(name_label, /*stretch*/ 1);
      head_layout->addWidget(value_chip, /*stretch*/ 0, Qt::AlignRight | Qt::AlignVCenter);

      auto* row       = new QWidget(controls_);
      auto* rowLayout = new QVBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(2);
      rowLayout->addWidget(head_row, /*stretch*/ 0);
      rowLayout->addWidget(slider, /*stretch*/ 0);

      controls_layout_->insertWidget(controls_layout_->count() - 1, row);
      return slider;
    };

    addSection(Tr("Tone"), Tr("Primary tonal shaping controls."));

    exposure_slider_ = addSlider(
        Tr("Exposure"), -1000, 1000, static_cast<int>(std::lround(state_.exposure_ * 100.0f)),
        [&](int v) {
          state_.exposure_ = static_cast<float>(v) / 100.0f;
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Exposure); },
        [this]() {
          ResetFieldToDefault(AdjustmentField::Exposure, [this](const AdjustmentState& defaults) {
            state_.exposure_ = defaults.exposure_;
          });
        },
        [](int v) { return QString::number(v / 100.0, 'f', 2); });

    contrast_slider_ = addSlider(
        Tr("Contrast"), -100, 100, static_cast<int>(std::lround(state_.contrast_)),
        [&](int v) {
          state_.contrast_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Contrast); },
        [this]() {
          ResetFieldToDefault(AdjustmentField::Contrast, [this](const AdjustmentState& defaults) {
            state_.contrast_ = defaults.contrast_;
          });
        },
        [](int v) { return QString::number(v, 'f', 2); });

    highlights_slider_ = addSlider(
        Tr("Highlights"), -100, 100, static_cast<int>(std::lround(state_.highlights_)),
        [&](int v) {
          state_.highlights_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Highlights); },
        [this]() {
          ResetFieldToDefault(AdjustmentField::Highlights,
                              [this](const AdjustmentState& defaults) {
                                state_.highlights_ = defaults.highlights_;
                              });
        },
        [](int v) { return QString::number(v, 'f', 2); });

    shadows_slider_ = addSlider(
        Tr("Shadows"), -100, 100, static_cast<int>(std::lround(state_.shadows_)),
        [&](int v) {
          state_.shadows_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Shadows); },
        [this]() {
          ResetFieldToDefault(AdjustmentField::Shadows, [this](const AdjustmentState& defaults) {
            state_.shadows_ = defaults.shadows_;
          });
        },
        [](int v) { return QString::number(v, 'f', 2); });

    whites_slider_ = addSlider(
        Tr("Whites"), -100, 100, static_cast<int>(std::lround(state_.whites_)),
        [&](int v) {
          state_.whites_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Whites); },
        [this]() {
          ResetFieldToDefault(AdjustmentField::Whites, [this](const AdjustmentState& defaults) {
            state_.whites_ = defaults.whites_;
          });
        },
        [](int v) { return QString::number(v, 'f', 2); });

    blacks_slider_ = addSlider(
        Tr("Blacks"), -100, 100, static_cast<int>(std::lround(state_.blacks_)),
        [&](int v) {
          state_.blacks_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Blacks); },
        [this]() {
          ResetFieldToDefault(AdjustmentField::Blacks, [this](const AdjustmentState& defaults) {
            state_.blacks_ = defaults.blacks_;
          });
        },
        [](int v) { return QString::number(v, 'f', 2); });

    addSection(Tr("Tone Curve"),
               Tr("Smooth tone curve mapped from input [0, 1] to output [0, 1]."));
    {
      auto* frame  = new QFrame(controls_);
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(0, 0, 0, 0);
      layout->setSpacing(4);

      curve_widget_ = new ToneCurveWidget(frame);
      curve_widget_->SetControlPoints(state_.curve_points_);
      curve_widget_->SetCurveChangedCallback([this](const std::vector<QPointF>& points) {
        if (syncing_controls_) {
          return;
        }
        state_.curve_points_ = NormalizeCurveControlPoints(points);
        RequestRender();
      });
      curve_widget_->SetCurveReleasedCallback([this](const std::vector<QPointF>& points) {
        if (syncing_controls_) {
          return;
        }
        state_.curve_points_ = NormalizeCurveControlPoints(points);
        CommitAdjustment(AdjustmentField::Curve);
      });
      RegisterCurveReset(curve_widget_, [this]() { ResetCurveToDefault(); });

      auto* actions_row        = new QWidget(frame);
      auto* actions_row_layout = new QHBoxLayout(actions_row);
      actions_row_layout->setContentsMargins(0, 0, 0, 0);
      actions_row_layout->setSpacing(8);

      auto* curve_hint = new QLabel(
          Tr("Left click/drag to shape. Right click a point to remove. Double click to reset."),
          actions_row);
      curve_hint->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
      AppTheme::MarkFontRole(curve_hint, AppTheme::FontRole::UiHint);
      curve_hint->setWordWrap(true);

      auto* reset_curve_btn = new QPushButton(Tr("Reset Curve"), actions_row);
      reset_curve_btn->setFixedHeight(28);
      reset_curve_btn->setStyleSheet(AppTheme::EditorPrimaryButtonStyle());
      QObject::connect(reset_curve_btn, &QPushButton::clicked, this,
                       [this]() { ResetCurveToDefault(); });

      actions_row_layout->addWidget(curve_hint, 1);
      actions_row_layout->addWidget(reset_curve_btn, 0);

      layout->addWidget(curve_widget_, 1);
      layout->addWidget(actions_row, 0);

      controls_layout_->insertWidget(controls_layout_->count() - 1, frame);
    }

    addSection(Tr("Color"), Tr("Color balance and saturation."));

    saturation_slider_ = addSlider(
        Tr("Saturation"), -100, 100, static_cast<int>(std::lround(state_.saturation_)),
        [&](int v) {
          state_.saturation_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Saturation); },
        [this]() {
          ResetFieldToDefault(AdjustmentField::Saturation,
                              [this](const AdjustmentState& defaults) {
                                state_.saturation_ = defaults.saturation_;
                              });
        },
        [](int v) { return QString::number(v, 'f', 2); });

    color_temp_mode_combo_ = addComboBox(
        Tr("White Balance"), {Tr("As Shot"), Tr("Custom")},
        ColorTempModeToComboIndex(state_.color_temp_mode_), [&](int idx) {
          const auto new_mode = ComboIndexToColorTempMode(idx);
          if (new_mode == state_.color_temp_mode_) {
            return;
          }
          if (state_.color_temp_mode_ == ColorTempMode::AS_SHOT &&
              new_mode == ColorTempMode::CUSTOM) {
            state_.color_temp_custom_cct_  = DisplayedColorTempCct(state_);
            state_.color_temp_custom_tint_ = DisplayedColorTempTint(state_);
          }
          state_.color_temp_mode_ = new_mode;
          if (new_mode == ColorTempMode::AS_SHOT) {
            PrimeColorTempDisplayForAsShot();
          }
          SyncControlsFromState();
          RequestRender();
          CommitAdjustment(AdjustmentField::ColorTemp);
        });

    color_temp_cct_slider_ = addSlider(
        Tr("Color Temp"), kColorTempSliderUiMin, kColorTempSliderUiMax,
        ColorTempCctToSliderPos(DisplayedColorTempCct(state_)),
        [&](int v) {
          PromoteColorTempToCustomForEditing();
          state_.color_temp_custom_cct_ = ColorTempSliderPosToCct(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::ColorTemp); },
        [this]() { ResetColorTempToAsShot(); },
        [](int v) {
          return QString("%1 K").arg(static_cast<int>(std::lround(ColorTempSliderPosToCct(v))));
        },
        SliderVisualStyle::Native);
    color_temp_cct_slider_->setStyleSheet(
        "QSlider::groove:horizontal {"
        "  border: 1px solid #2A2A2A;"
        "  height: 8px;"
        "  border-radius: 4px;"
        "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        "stop:0 #9BD8FF, stop:0.5 #FFE8B0, stop:1 #FF8A3D);"
        "}"
        "QSlider::handle:horizontal {"
        "  background: #F2F2F2;"
        "  border: 1px solid #2A2A2A;"
        "  width: 14px;"
        "  margin: -4px 0;"
        "  border-radius: 7px;"
        "}");

    color_temp_tint_slider_ = addSlider(
        Tr("Color Tint"), kColorTempTintMin, kColorTempTintMax,
        static_cast<int>(std::lround(DisplayedColorTempTint(state_))),
        [&](int v) {
          PromoteColorTempToCustomForEditing();
          state_.color_temp_custom_tint_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::ColorTemp); },
        [this]() { ResetColorTempToAsShot(); },
        [](int v) { return QString::number(v, 'f', 0); }, SliderVisualStyle::Native);
    color_temp_tint_slider_->setStyleSheet(
        "QSlider::groove:horizontal {"
        "  border: 1px solid #2A2A2A;"
        "  height: 8px;"
        "  border-radius: 4px;"
        "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        "stop:0 #49C26D, stop:0.5 #E6E6E6, stop:1 #A85AE6);"
        "}"
        "QSlider::handle:horizontal {"
        "  background: #F2F2F2;"
        "  border: 1px solid #2A2A2A;"
        "  width: 14px;"
        "  margin: -4px 0;"
        "  border-radius: 7px;"
        "}");

    color_temp_unsupported_label_ =
        new QLabel(Tr("Color temperature/tint is unavailable for this image."), controls_);
    color_temp_unsupported_label_->setWordWrap(true);
    AppTheme::MarkFontRole(color_temp_unsupported_label_, AppTheme::FontRole::UiHint);
    color_temp_unsupported_label_->setStyleSheet(
        "QLabel {"
        "  color: #FFB454;"
        "  background: rgba(255, 180, 84, 0.12);"
        "  border: 1px solid rgba(255, 180, 84, 0.35);"
        "  border-radius: 8px;"
        "  padding: 6px 8px;"
        "}");
    controls_layout_->insertWidget(controls_layout_->count() - 1, color_temp_unsupported_label_);
    color_temp_unsupported_label_->setVisible(!state_.color_temp_supported_);

    addSection(Tr("Detail"), Tr("Micro-contrast and sharpen controls."));

    sharpen_slider_ = addSlider(
        Tr("Sharpen"), -100, 100, static_cast<int>(std::lround(state_.sharpen_)),
        [&](int v) {
          state_.sharpen_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Sharpen); },
        [this]() {
          ResetFieldToDefault(AdjustmentField::Sharpen, [this](const AdjustmentState& defaults) {
            state_.sharpen_ = defaults.sharpen_;
          });
        },
        [](int v) { return QString::number(v, 'f', 2); });

    clarity_slider_ = addSlider(
        Tr("Clarity"), -100, 100, static_cast<int>(std::lround(state_.clarity_)),
        [&](int v) {
          state_.clarity_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Clarity); },
        [this]() {
          ResetFieldToDefault(AdjustmentField::Clarity, [this](const AdjustmentState& defaults) {
            state_.clarity_ = defaults.clarity_;
          });
        },
        [](int v) { return QString::number(v, 'f', 2); });




}

}  // namespace alcedo::ui
