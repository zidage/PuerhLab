//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/widgets/tone_control_panel_widget.hpp"

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSizePolicy>
#include <QString>
#include <QStringList>
#include <QStringLiteral>
#include <algorithm>
#include <cmath>
#include <utility>

#include "ui/alcedo_main/app_theme.hpp"
#include "ui/alcedo_main/editor_dialog/editor_slider_styling.hpp"
#include "ui/alcedo_main/editor_dialog/modules/color_temp.hpp"
#include "ui/alcedo_main/editor_dialog/modules/curve.hpp"
#include "ui/alcedo_main/editor_dialog/session/editor_adjustment_session.hpp"
#include "ui/alcedo_main/i18n.hpp"

namespace alcedo::ui {
namespace {

constexpr char kLocalizedTextProperty[]      = "puerhlabI18nText";
constexpr char kLocalizedTextUpperProperty[] = "puerhlabI18nTextUpper";
constexpr char kLocalizedToolTipProperty[]   = "puerhlabI18nToolTip";

void SetLocalizedText(QObject* object, const char* source, bool uppercase = false) {
  if (!object || source == nullptr) {
    return;
  }
  object->setProperty(kLocalizedTextProperty, source);
  object->setProperty(kLocalizedTextUpperProperty, uppercase);
  QString text = Tr(source);
  if (uppercase) {
    text = text.toUpper();
  }
  if (auto* label = qobject_cast<QLabel*>(object)) {
    label->setText(text);
  } else if (auto* button = qobject_cast<QPushButton*>(object)) {
    button->setText(text);
  }
}

void SetLocalizedToolTip(QWidget* widget, const char* source) {
  if (!widget || source == nullptr) {
    return;
  }
  widget->setProperty(kLocalizedToolTipProperty, source);
  widget->setToolTip(Tr(source));
  widget->setAccessibleName(Tr(source));
}

auto NewLocalizedLabel(const char* source, QWidget* parent, bool uppercase = false) -> QLabel* {
  auto* label = new QLabel(parent);
  SetLocalizedText(label, source, uppercase);
  return label;
}

auto NewLocalizedButton(const char* source, QWidget* parent) -> QPushButton* {
  auto* button = new QPushButton(parent);
  SetLocalizedText(button, source);
  return button;
}

auto ColorTempSliderPosToCct(int pos) -> float { return color_temp::SliderPosToCct(pos); }
auto ColorTempCctToSliderPos(float cct) -> int { return color_temp::CctToSliderPos(cct); }

}  // namespace

ToneControlPanelWidget::ToneControlPanelWidget(QWidget* parent) : AdjustmentPanelWidget(parent) {}

void ToneControlPanelWidget::Configure(Dependencies deps, Callbacks callbacks) {
  deps_      = std::move(deps);
  callbacks_ = std::move(callbacks);
  PullToneStateFromDialog();
  PullCommittedToneStateFromDialog();
}

void ToneControlPanelWidget::SetSyncing(bool syncing) { local_syncing_ = syncing; }

auto ToneControlPanelWidget::IsSyncing() const -> bool {
  if (local_syncing_) {
    return true;
  }
  if (callbacks_.is_global_syncing && callbacks_.is_global_syncing()) {
    return true;
  }
  return false;
}

void ToneControlPanelWidget::RequestPipelineRender() {
  if (callbacks_.request_render) {
    callbacks_.request_render();
  }
}

void ToneControlPanelWidget::ProjectToneStateToDialog() {
  if (!deps_.dialog_state) {
    return;
  }
  auto& s        = *deps_.dialog_state;
  s.exposure_    = tone_state_.exposure_;
  s.contrast_    = tone_state_.contrast_;
  s.blacks_      = tone_state_.blacks_;
  s.whites_      = tone_state_.whites_;
  s.shadows_     = tone_state_.shadows_;
  s.highlights_  = tone_state_.highlights_;
  s.curve_points_ = tone_state_.curve_points_;
  s.saturation_  = tone_state_.saturation_;
  s.sharpen_     = tone_state_.sharpen_;
  s.clarity_     = tone_state_.clarity_;
}

void ToneControlPanelWidget::PullToneStateFromDialog() {
  if (!deps_.dialog_state) {
    return;
  }
  const auto& s              = *deps_.dialog_state;
  tone_state_.exposure_      = s.exposure_;
  tone_state_.contrast_      = s.contrast_;
  tone_state_.blacks_        = s.blacks_;
  tone_state_.whites_        = s.whites_;
  tone_state_.shadows_       = s.shadows_;
  tone_state_.highlights_    = s.highlights_;
  tone_state_.curve_points_  = s.curve_points_;
  tone_state_.saturation_    = s.saturation_;
  tone_state_.sharpen_       = s.sharpen_;
  tone_state_.clarity_       = s.clarity_;
}

void ToneControlPanelWidget::PullCommittedToneStateFromDialog() {
  if (!deps_.dialog_committed_state) {
    return;
  }
  const auto& s                          = *deps_.dialog_committed_state;
  committed_tone_state_.exposure_        = s.exposure_;
  committed_tone_state_.contrast_        = s.contrast_;
  committed_tone_state_.blacks_          = s.blacks_;
  committed_tone_state_.whites_          = s.whites_;
  committed_tone_state_.shadows_         = s.shadows_;
  committed_tone_state_.highlights_      = s.highlights_;
  committed_tone_state_.curve_points_    = s.curve_points_;
  committed_tone_state_.saturation_      = s.saturation_;
  committed_tone_state_.sharpen_         = s.sharpen_;
  committed_tone_state_.clarity_         = s.clarity_;
}

void ToneControlPanelWidget::PreviewToneField(AdjustmentField field) {
  ProjectToneStateToDialog();
  RequestPipelineRender();
  if (deps_.session) {
    AdjustmentPreview preview{
        .field  = field,
        .params = nlohmann::json{},
        .policy = PreviewPolicy::FastViewport,
    };
    if (deps_.dialog_state) {
      preview.params = deps_.session->ParamsForField(field, *deps_.dialog_state);
    }
    deps_.session->Preview(preview);
  }
}

void ToneControlPanelWidget::CommitToneField(AdjustmentField field) {
  ProjectToneStateToDialog();
  if (deps_.session) {
    deps_.session->Commit(field);
  }
  PullCommittedToneStateFromDialog();
}

void ToneControlPanelWidget::ResetToneFieldToDefault(
    AdjustmentField field,
    const std::function<void(const ToneAdjustmentState&, const AdjustmentState&)>& apply_default) {
  if (!apply_default || !callbacks_.default_adjustment_state) {
    return;
  }
  const AdjustmentState& dialog_defaults = callbacks_.default_adjustment_state();
  ToneAdjustmentState    tone_defaults{};
  tone_defaults.exposure_     = dialog_defaults.exposure_;
  tone_defaults.contrast_     = dialog_defaults.contrast_;
  tone_defaults.blacks_       = dialog_defaults.blacks_;
  tone_defaults.whites_       = dialog_defaults.whites_;
  tone_defaults.shadows_      = dialog_defaults.shadows_;
  tone_defaults.highlights_   = dialog_defaults.highlights_;
  tone_defaults.curve_points_ = dialog_defaults.curve_points_;
  tone_defaults.saturation_   = dialog_defaults.saturation_;
  tone_defaults.sharpen_      = dialog_defaults.sharpen_;
  tone_defaults.clarity_      = dialog_defaults.clarity_;

  apply_default(tone_defaults, dialog_defaults);
  ProjectToneStateToDialog();
  if (callbacks_.sync_controls_from_state) {
    callbacks_.sync_controls_from_state();
  }
  RequestPipelineRender();
  CommitToneField(field);
}

void ToneControlPanelWidget::ResetCurveToDefaultLocal() {
  tone_state_.curve_points_ = curve::DefaultCurveControlPoints();
  ProjectToneStateToDialog();
  if (callbacks_.sync_controls_from_state) {
    callbacks_.sync_controls_from_state();
  }
  RequestPipelineRender();
  CommitToneField(AdjustmentField::Curve);
}

void ToneControlPanelWidget::PromoteColorTempToCustomForEditing() {
  if (!deps_.dialog_state) {
    return;
  }
  auto& s = *deps_.dialog_state;
  if (s.color_temp_mode_ == ColorTempMode::CUSTOM) {
    return;
  }
  s.color_temp_custom_cct_  = DisplayedColorTempCct(s);
  s.color_temp_custom_tint_ = DisplayedColorTempTint(s);
  s.color_temp_mode_        = ColorTempMode::CUSTOM;

  if (color_temp_mode_combo_) {
    const bool prev = local_syncing_;
    local_syncing_  = true;
    color_temp_mode_combo_->setCurrentIndex(ColorTempModeToComboIndex(s.color_temp_mode_));
    local_syncing_  = prev;
  }
}

void ToneControlPanelWidget::Build() {
  if (!deps_.panel_layout) {
    return;
  }

  auto* parent_widget = this;
  auto& layout        = *deps_.panel_layout;

  auto* controls_header = NewLocalizedLabel("Adjustments", parent_widget);
  controls_header->setObjectName("SectionTitle");
  controls_header->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
  AppTheme::MarkFontRole(controls_header, AppTheme::FontRole::UiHeadline);
  layout.addWidget(controls_header, 0);
  layout.addStretch();

  BuildToneSection();
  BuildToneCurveSection();
  BuildColorSection();
  BuildDetailSection();
}

namespace {

void AddSection(QWidget* parent, QVBoxLayout& layout, const char* title_source,
                const char* subtitle_source) {
  auto* frame = new QWidget(parent);
  auto* v     = new QVBoxLayout(frame);
  v->setContentsMargins(0, 8, 0, 2);
  v->setSpacing(4);

  auto* header_row    = new QWidget(frame);
  auto* header_layout = new QHBoxLayout(header_row);
  header_layout->setContentsMargins(0, 0, 0, 0);
  header_layout->setSpacing(6);

  auto* t = NewLocalizedLabel(title_source, header_row, true);
  t->setObjectName("EditorSectionTitle");
  t->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
  AppTheme::MarkFontRole(t, AppTheme::FontRole::UiOverline);
  if (subtitle_source != nullptr && subtitle_source[0] != '\0') {
    SetLocalizedToolTip(t, subtitle_source);
  }
  header_layout->addWidget(t, 0);
  header_layout->addStretch(1);

  auto* divider = new QFrame(frame);
  divider->setFrameShape(QFrame::HLine);
  divider->setFixedHeight(1);
  divider->setStyleSheet(QStringLiteral("QFrame { background: %1; border: none; }")
                             .arg(WithAlpha(AppTheme::Instance().dividerColor(), 110)
                                      .name(QColor::HexArgb)));

  v->addWidget(header_row, 0);
  v->addWidget(divider, 0);
  layout.insertWidget(layout.count() - 1, frame, 0);
}

}  // namespace

void ToneControlPanelWidget::BuildToneSection() {
  auto* parent_widget = this;
  auto& layout        = *deps_.panel_layout;

  const QString value_chip_style =
      QStringLiteral("QLabel {"
                     "  color: %1;"
                     "  background: transparent;"
                     "  border: none;"
                     "  padding: 0;"
                     "}")
          .arg(AppTheme::Instance().textMutedColor().name(QColor::HexRgb));

  auto add_slider = [&, value_chip_style](
                        const char* name_source, int min, int max, int value,
                        std::function<void(int)>           on_change,
                        std::function<void()>              on_release,
                        std::function<void()>              on_reset,
                        std::function<QString(int)>        formatter,
                        SliderVisualStyle visual_style = SliderVisualStyle::Accent) -> QSlider* {
    auto* name_label = NewLocalizedLabel(name_source, parent_widget);
    name_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    AppTheme::MarkFontRole(name_label, AppTheme::FontRole::UiCaption);
    name_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    auto* value_chip = new QLabel(formatter(value), parent_widget);
    value_chip->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    value_chip->setMinimumWidth(40);
    value_chip->setMaximumWidth(72);
    value_chip->setFixedHeight(16);
    value_chip->setStyleSheet(value_chip_style);
    AppTheme::MarkFontRole(value_chip, AppTheme::FontRole::DataCaption);

    QSlider* slider = nullptr;
    if (visual_style == SliderVisualStyle::Accent) {
      slider = new AccentBalanceSlider(kRegularSliderMetrics, parent_widget);
    } else {
      slider = new QSlider(Qt::Horizontal, parent_widget);
    }
    slider->setRange(min, max);
    slider->setValue(value);
    slider->setSingleStep(1);
    slider->setPageStep(std::max(1, (max - min) / 20));
    slider->setMinimumWidth(0);
    slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    slider->setFixedHeight(22);

    QObject::connect(slider, &QSlider::valueChanged, parent_widget,
                     [this, value_chip, formatter, on_change](int v) {
                       value_chip->setText(formatter(v));
                       if (IsSyncing()) {
                         return;
                       }
                       on_change(v);
                     });

    QObject::connect(slider, &QSlider::sliderReleased, parent_widget, [this, on_release]() {
      if (IsSyncing()) {
        return;
      }
      on_release();
    });

    if (callbacks_.register_slider_reset) {
      callbacks_.register_slider_reset(slider, [this, on_reset]() {
        if (IsSyncing()) {
          return;
        }
        on_reset();
      });
    }

    auto* head_row    = new QWidget(parent_widget);
    auto* head_layout = new QHBoxLayout(head_row);
    head_layout->setContentsMargins(0, 0, 0, 0);
    head_layout->setSpacing(8);
    head_layout->addWidget(name_label, 1);
    head_layout->addWidget(value_chip, 0, Qt::AlignRight | Qt::AlignVCenter);

    auto* row        = new QWidget(parent_widget);
    auto* row_layout = new QVBoxLayout(row);
    row_layout->setContentsMargins(0, 0, 0, 0);
    row_layout->setSpacing(2);
    row_layout->addWidget(head_row, 0);
    row_layout->addWidget(slider, 0);

    layout.insertWidget(layout.count() - 1, row);
    return slider;
  };

  AddSection(parent_widget, layout, "Tone", "Primary tonal shaping controls.");

  exposure_slider_ = add_slider(
      "Exposure", -1000, 1000, static_cast<int>(std::lround(tone_state_.exposure_ * 100.0f)),
      [this](int v) {
        tone_state_.exposure_ = static_cast<float>(v) / 100.0f;
        PreviewToneField(AdjustmentField::Exposure);
      },
      [this]() { CommitToneField(AdjustmentField::Exposure); },
      [this]() {
        ResetToneFieldToDefault(
            AdjustmentField::Exposure,
            [this](const ToneAdjustmentState& defaults, const AdjustmentState&) {
              tone_state_.exposure_ = defaults.exposure_;
            });
      },
      [](int v) { return QString::number(v / 100.0, 'f', 2); });

  contrast_slider_ = add_slider(
      "Contrast", -100, 100, static_cast<int>(std::lround(tone_state_.contrast_)),
      [this](int v) {
        tone_state_.contrast_ = static_cast<float>(v);
        PreviewToneField(AdjustmentField::Contrast);
      },
      [this]() { CommitToneField(AdjustmentField::Contrast); },
      [this]() {
        ResetToneFieldToDefault(
            AdjustmentField::Contrast,
            [this](const ToneAdjustmentState& defaults, const AdjustmentState&) {
              tone_state_.contrast_ = defaults.contrast_;
            });
      },
      [](int v) { return QString::number(v, 'f', 2); });

  highlights_slider_ = add_slider(
      "Highlights", -100, 100, static_cast<int>(std::lround(tone_state_.highlights_)),
      [this](int v) {
        tone_state_.highlights_ = static_cast<float>(v);
        PreviewToneField(AdjustmentField::Highlights);
      },
      [this]() { CommitToneField(AdjustmentField::Highlights); },
      [this]() {
        ResetToneFieldToDefault(
            AdjustmentField::Highlights,
            [this](const ToneAdjustmentState& defaults, const AdjustmentState&) {
              tone_state_.highlights_ = defaults.highlights_;
            });
      },
      [](int v) { return QString::number(v, 'f', 2); });

  shadows_slider_ = add_slider(
      "Shadows", -100, 100, static_cast<int>(std::lround(tone_state_.shadows_)),
      [this](int v) {
        tone_state_.shadows_ = static_cast<float>(v);
        PreviewToneField(AdjustmentField::Shadows);
      },
      [this]() { CommitToneField(AdjustmentField::Shadows); },
      [this]() {
        ResetToneFieldToDefault(
            AdjustmentField::Shadows,
            [this](const ToneAdjustmentState& defaults, const AdjustmentState&) {
              tone_state_.shadows_ = defaults.shadows_;
            });
      },
      [](int v) { return QString::number(v, 'f', 2); });

  whites_slider_ = add_slider(
      "Whites", -100, 100, static_cast<int>(std::lround(tone_state_.whites_)),
      [this](int v) {
        tone_state_.whites_ = static_cast<float>(v);
        PreviewToneField(AdjustmentField::Whites);
      },
      [this]() { CommitToneField(AdjustmentField::Whites); },
      [this]() {
        ResetToneFieldToDefault(
            AdjustmentField::Whites,
            [this](const ToneAdjustmentState& defaults, const AdjustmentState&) {
              tone_state_.whites_ = defaults.whites_;
            });
      },
      [](int v) { return QString::number(v, 'f', 2); });

  blacks_slider_ = add_slider(
      "Blacks", -100, 100, static_cast<int>(std::lround(tone_state_.blacks_)),
      [this](int v) {
        tone_state_.blacks_ = static_cast<float>(v);
        PreviewToneField(AdjustmentField::Blacks);
      },
      [this]() { CommitToneField(AdjustmentField::Blacks); },
      [this]() {
        ResetToneFieldToDefault(
            AdjustmentField::Blacks,
            [this](const ToneAdjustmentState& defaults, const AdjustmentState&) {
              tone_state_.blacks_ = defaults.blacks_;
            });
      },
      [](int v) { return QString::number(v, 'f', 2); });
}

void ToneControlPanelWidget::BuildToneCurveSection() {
  auto* parent_widget = this;
  auto& layout        = *deps_.panel_layout;

  AddSection(parent_widget, layout, "Tone Curve",
             "Smooth tone curve mapped from input [0, 1] to output [0, 1].");

  auto* frame      = new QFrame(parent_widget);
  auto* v_layout   = new QVBoxLayout(frame);
  v_layout->setContentsMargins(0, 0, 0, 0);
  v_layout->setSpacing(4);

  curve_widget_ = new ToneCurveWidget(frame);
  curve_widget_->SetControlPoints(tone_state_.curve_points_);
  curve_widget_->SetCurveChangedCallback([this](const std::vector<QPointF>& points) {
    if (IsSyncing()) {
      return;
    }
    tone_state_.curve_points_ = curve::NormalizeCurveControlPoints(points);
    PreviewToneField(AdjustmentField::Curve);
  });
  curve_widget_->SetCurveReleasedCallback([this](const std::vector<QPointF>& points) {
    if (IsSyncing()) {
      return;
    }
    tone_state_.curve_points_ = curve::NormalizeCurveControlPoints(points);
    CommitToneField(AdjustmentField::Curve);
  });
  if (callbacks_.register_curve_reset) {
    callbacks_.register_curve_reset(curve_widget_, [this]() { ResetCurveToDefaultLocal(); });
  }

  auto* actions_row        = new QWidget(frame);
  auto* actions_row_layout = new QHBoxLayout(actions_row);
  actions_row_layout->setContentsMargins(0, 0, 0, 0);
  actions_row_layout->setSpacing(8);

  auto* curve_hint = NewLocalizedLabel(
      "Left click/drag to shape. Right click a point to remove. Double click to reset.",
      actions_row);
  curve_hint->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
  AppTheme::MarkFontRole(curve_hint, AppTheme::FontRole::UiHint);
  curve_hint->setWordWrap(true);

  auto* reset_curve_btn = NewLocalizedButton("Reset Curve", actions_row);
  reset_curve_btn->setFixedHeight(28);
  reset_curve_btn->setStyleSheet(AppTheme::EditorPrimaryButtonStyle());
  QObject::connect(reset_curve_btn, &QPushButton::clicked, this,
                   [this]() { ResetCurveToDefaultLocal(); });

  actions_row_layout->addWidget(curve_hint, 1);
  actions_row_layout->addWidget(reset_curve_btn, 0);

  v_layout->addWidget(curve_widget_, 1);
  v_layout->addWidget(actions_row, 0);

  layout.insertWidget(layout.count() - 1, frame);
}

void ToneControlPanelWidget::BuildColorSection() {
  auto* parent_widget = this;
  auto& layout        = *deps_.panel_layout;

  const QString value_chip_style =
      QStringLiteral("QLabel {"
                     "  color: %1;"
                     "  background: transparent;"
                     "  border: none;"
                     "  padding: 0;"
                     "}")
          .arg(AppTheme::Instance().textMutedColor().name(QColor::HexRgb));

  auto add_combo_box = [&](const char* name_source, const QStringList& items, int initial_index,
                           std::function<void(int)> on_change) -> QComboBox* {
    auto* label = NewLocalizedLabel(name_source, parent_widget);
    label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    AppTheme::MarkFontRole(label, AppTheme::FontRole::UiBody);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    auto* combo = new QComboBox(parent_widget);
    combo->addItems(items);
    combo->setCurrentIndex(initial_index);
    combo->setMinimumWidth(96);
    combo->setMaximumWidth(160);
    combo->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    combo->setFixedHeight(26);
    combo->setStyleSheet(AppTheme::EditorComboBoxStyle());

    QObject::connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), parent_widget,
                     [this, on_change](int idx) {
                       if (IsSyncing()) {
                         return;
                       }
                       on_change(idx);
                     });

    auto* row        = new QWidget(parent_widget);
    auto* row_layout = new QHBoxLayout(row);
    row_layout->setContentsMargins(0, 0, 0, 0);
    row_layout->setSpacing(8);
    row_layout->addWidget(label, 1);
    row_layout->addWidget(combo, 0, Qt::AlignRight | Qt::AlignVCenter);

    layout.insertWidget(layout.count() - 1, row);
    return combo;
  };

  auto add_slider = [&, value_chip_style](
                        const char* name_source, int min, int max, int value,
                        std::function<void(int)>    on_change,
                        std::function<void()>       on_release,
                        std::function<void()>       on_reset,
                        std::function<QString(int)> formatter,
                        SliderVisualStyle visual_style = SliderVisualStyle::Accent) -> QSlider* {
    auto* name_label = NewLocalizedLabel(name_source, parent_widget);
    name_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    AppTheme::MarkFontRole(name_label, AppTheme::FontRole::UiCaption);
    name_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    auto* value_chip = new QLabel(formatter(value), parent_widget);
    value_chip->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    value_chip->setMinimumWidth(40);
    value_chip->setMaximumWidth(72);
    value_chip->setFixedHeight(16);
    value_chip->setStyleSheet(value_chip_style);
    AppTheme::MarkFontRole(value_chip, AppTheme::FontRole::DataCaption);

    QSlider* slider = nullptr;
    if (visual_style == SliderVisualStyle::Accent) {
      slider = new AccentBalanceSlider(kRegularSliderMetrics, parent_widget);
    } else {
      slider = new QSlider(Qt::Horizontal, parent_widget);
    }
    slider->setRange(min, max);
    slider->setValue(value);
    slider->setSingleStep(1);
    slider->setPageStep(std::max(1, (max - min) / 20));
    slider->setMinimumWidth(0);
    slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    slider->setFixedHeight(22);

    QObject::connect(slider, &QSlider::valueChanged, parent_widget,
                     [this, value_chip, formatter, on_change](int v) {
                       value_chip->setText(formatter(v));
                       if (IsSyncing()) {
                         return;
                       }
                       on_change(v);
                     });

    QObject::connect(slider, &QSlider::sliderReleased, parent_widget, [this, on_release]() {
      if (IsSyncing()) {
        return;
      }
      on_release();
    });

    if (callbacks_.register_slider_reset) {
      callbacks_.register_slider_reset(slider, [this, on_reset]() {
        if (IsSyncing()) {
          return;
        }
        on_reset();
      });
    }

    auto* head_row    = new QWidget(parent_widget);
    auto* head_layout = new QHBoxLayout(head_row);
    head_layout->setContentsMargins(0, 0, 0, 0);
    head_layout->setSpacing(8);
    head_layout->addWidget(name_label, 1);
    head_layout->addWidget(value_chip, 0, Qt::AlignRight | Qt::AlignVCenter);

    auto* row        = new QWidget(parent_widget);
    auto* row_layout = new QVBoxLayout(row);
    row_layout->setContentsMargins(0, 0, 0, 0);
    row_layout->setSpacing(2);
    row_layout->addWidget(head_row, 0);
    row_layout->addWidget(slider, 0);

    layout.insertWidget(layout.count() - 1, row);
    return slider;
  };

  AddSection(parent_widget, layout, "Color", "Color balance and saturation.");

  saturation_slider_ = add_slider(
      "Saturation", -100, 100, static_cast<int>(std::lround(tone_state_.saturation_)),
      [this](int v) {
        tone_state_.saturation_ = static_cast<float>(v);
        PreviewToneField(AdjustmentField::Saturation);
      },
      [this]() { CommitToneField(AdjustmentField::Saturation); },
      [this]() {
        ResetToneFieldToDefault(
            AdjustmentField::Saturation,
            [this](const ToneAdjustmentState& defaults, const AdjustmentState&) {
              tone_state_.saturation_ = defaults.saturation_;
            });
      },
      [](int v) { return QString::number(v, 'f', 2); });

  // Color temperature controls (state still lives on the dialog state; ownership
  // moves to ColorTempPipelineAdapter in a later phase).
  auto* dialog_state_ptr = deps_.dialog_state;
  color_temp_mode_combo_ = add_combo_box(
      "White Balance", {Tr("As Shot"), Tr("Custom")},
      ColorTempModeToComboIndex(dialog_state_ptr ? dialog_state_ptr->color_temp_mode_
                                                 : ColorTempMode::AS_SHOT),
      [this](int idx) {
        if (!deps_.dialog_state) {
          return;
        }
        auto&      s        = *deps_.dialog_state;
        const auto new_mode = ComboIndexToColorTempMode(idx);
        if (new_mode == s.color_temp_mode_) {
          return;
        }
        if (s.color_temp_mode_ == ColorTempMode::AS_SHOT && new_mode == ColorTempMode::CUSTOM) {
          s.color_temp_custom_cct_  = DisplayedColorTempCct(s);
          s.color_temp_custom_tint_ = DisplayedColorTempTint(s);
        }
        s.color_temp_mode_ = new_mode;
        if (new_mode == ColorTempMode::AS_SHOT && callbacks_.prime_color_temp_for_as_shot) {
          callbacks_.prime_color_temp_for_as_shot();
        }
        if (callbacks_.sync_controls_from_state) {
          callbacks_.sync_controls_from_state();
        }
        RequestPipelineRender();
        if (deps_.session) {
          deps_.session->Commit(AdjustmentField::ColorTemp);
        }
        PullCommittedToneStateFromDialog();
      });

  color_temp_cct_slider_ = add_slider(
      "Color Temp", color_temp::kSliderUiMin, color_temp::kSliderUiMax,
      ColorTempCctToSliderPos(dialog_state_ptr ? DisplayedColorTempCct(*dialog_state_ptr) : 6500.0f),
      [this](int v) {
        if (!deps_.dialog_state) {
          return;
        }
        PromoteColorTempToCustomForEditing();
        deps_.dialog_state->color_temp_custom_cct_ = ColorTempSliderPosToCct(v);
        RequestPipelineRender();
        if (deps_.session) {
          AdjustmentPreview preview{
              .field  = AdjustmentField::ColorTemp,
              .params = deps_.session->ParamsForField(AdjustmentField::ColorTemp,
                                                      *deps_.dialog_state),
              .policy = PreviewPolicy::FastViewport,
          };
          deps_.session->Preview(preview);
        }
      },
      [this]() {
        if (deps_.session) {
          deps_.session->Commit(AdjustmentField::ColorTemp);
        }
        PullCommittedToneStateFromDialog();
      },
      [this]() {
        if (callbacks_.reset_color_temp_to_as_shot) {
          callbacks_.reset_color_temp_to_as_shot();
        }
      },
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

  color_temp_tint_slider_ = add_slider(
      "Color Tint", color_temp::kTintMin, color_temp::kTintMax,
      static_cast<int>(std::lround(
          dialog_state_ptr ? DisplayedColorTempTint(*dialog_state_ptr) : 0.0f)),
      [this](int v) {
        if (!deps_.dialog_state) {
          return;
        }
        PromoteColorTempToCustomForEditing();
        deps_.dialog_state->color_temp_custom_tint_ = static_cast<float>(v);
        RequestPipelineRender();
        if (deps_.session) {
          AdjustmentPreview preview{
              .field  = AdjustmentField::ColorTemp,
              .params = deps_.session->ParamsForField(AdjustmentField::ColorTemp,
                                                      *deps_.dialog_state),
              .policy = PreviewPolicy::FastViewport,
          };
          deps_.session->Preview(preview);
        }
      },
      [this]() {
        if (deps_.session) {
          deps_.session->Commit(AdjustmentField::ColorTemp);
        }
        PullCommittedToneStateFromDialog();
      },
      [this]() {
        if (callbacks_.reset_color_temp_to_as_shot) {
          callbacks_.reset_color_temp_to_as_shot();
        }
      },
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

  color_temp_unsupported_label_ = NewLocalizedLabel(
      "Color temperature/tint is unavailable for this image.", parent_widget);
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
  layout.insertWidget(layout.count() - 1, color_temp_unsupported_label_);
  color_temp_unsupported_label_->setVisible(
      dialog_state_ptr ? !dialog_state_ptr->color_temp_supported_ : false);
}

void ToneControlPanelWidget::BuildDetailSection() {
  auto* parent_widget = this;
  auto& layout        = *deps_.panel_layout;

  const QString value_chip_style =
      QStringLiteral("QLabel {"
                     "  color: %1;"
                     "  background: transparent;"
                     "  border: none;"
                     "  padding: 0;"
                     "}")
          .arg(AppTheme::Instance().textMutedColor().name(QColor::HexRgb));

  auto add_slider = [&, value_chip_style](
                        const char* name_source, int min, int max, int value,
                        std::function<void(int)>    on_change,
                        std::function<void()>       on_release,
                        std::function<void()>       on_reset,
                        std::function<QString(int)> formatter) -> QSlider* {
    auto* name_label = NewLocalizedLabel(name_source, parent_widget);
    name_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    AppTheme::MarkFontRole(name_label, AppTheme::FontRole::UiCaption);
    name_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    auto* value_chip = new QLabel(formatter(value), parent_widget);
    value_chip->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    value_chip->setMinimumWidth(40);
    value_chip->setMaximumWidth(72);
    value_chip->setFixedHeight(16);
    value_chip->setStyleSheet(value_chip_style);
    AppTheme::MarkFontRole(value_chip, AppTheme::FontRole::DataCaption);

    auto* slider = new AccentBalanceSlider(kRegularSliderMetrics, parent_widget);
    slider->setRange(min, max);
    slider->setValue(value);
    slider->setSingleStep(1);
    slider->setPageStep(std::max(1, (max - min) / 20));
    slider->setMinimumWidth(0);
    slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    slider->setFixedHeight(22);

    QObject::connect(slider, &QSlider::valueChanged, parent_widget,
                     [this, value_chip, formatter, on_change](int v) {
                       value_chip->setText(formatter(v));
                       if (IsSyncing()) {
                         return;
                       }
                       on_change(v);
                     });

    QObject::connect(slider, &QSlider::sliderReleased, parent_widget, [this, on_release]() {
      if (IsSyncing()) {
        return;
      }
      on_release();
    });

    if (callbacks_.register_slider_reset) {
      callbacks_.register_slider_reset(slider, [this, on_reset]() {
        if (IsSyncing()) {
          return;
        }
        on_reset();
      });
    }

    auto* head_row    = new QWidget(parent_widget);
    auto* head_layout = new QHBoxLayout(head_row);
    head_layout->setContentsMargins(0, 0, 0, 0);
    head_layout->setSpacing(8);
    head_layout->addWidget(name_label, 1);
    head_layout->addWidget(value_chip, 0, Qt::AlignRight | Qt::AlignVCenter);

    auto* row        = new QWidget(parent_widget);
    auto* row_layout = new QVBoxLayout(row);
    row_layout->setContentsMargins(0, 0, 0, 0);
    row_layout->setSpacing(2);
    row_layout->addWidget(head_row, 0);
    row_layout->addWidget(slider, 0);

    layout.insertWidget(layout.count() - 1, row);
    return slider;
  };

  AddSection(parent_widget, layout, "Detail", "Micro-contrast and sharpen controls.");

  sharpen_slider_ = add_slider(
      "Sharpen", -100, 100, static_cast<int>(std::lround(tone_state_.sharpen_)),
      [this](int v) {
        tone_state_.sharpen_ = static_cast<float>(v);
        PreviewToneField(AdjustmentField::Sharpen);
      },
      [this]() { CommitToneField(AdjustmentField::Sharpen); },
      [this]() {
        ResetToneFieldToDefault(
            AdjustmentField::Sharpen,
            [this](const ToneAdjustmentState& defaults, const AdjustmentState&) {
              tone_state_.sharpen_ = defaults.sharpen_;
            });
      },
      [](int v) { return QString::number(v, 'f', 2); });

  clarity_slider_ = add_slider(
      "Clarity", -100, 100, static_cast<int>(std::lround(tone_state_.clarity_)),
      [this](int v) {
        tone_state_.clarity_ = static_cast<float>(v);
        PreviewToneField(AdjustmentField::Clarity);
      },
      [this]() { CommitToneField(AdjustmentField::Clarity); },
      [this]() {
        ResetToneFieldToDefault(
            AdjustmentField::Clarity,
            [this](const ToneAdjustmentState& defaults, const AdjustmentState&) {
              tone_state_.clarity_ = defaults.clarity_;
            });
      },
      [](int v) { return QString::number(v, 'f', 2); });
}

void ToneControlPanelWidget::SyncControlsFromDialogState() {
  if (!deps_.dialog_state) {
    return;
  }
  PullToneStateFromDialog();
  PullCommittedToneStateFromDialog();

  const bool prev = local_syncing_;
  local_syncing_  = true;
  if (exposure_slider_) {
    exposure_slider_->setValue(static_cast<int>(std::lround(tone_state_.exposure_ * 100.0f)));
  }
  if (contrast_slider_) {
    contrast_slider_->setValue(static_cast<int>(std::lround(tone_state_.contrast_)));
  }
  if (saturation_slider_) {
    saturation_slider_->setValue(static_cast<int>(std::lround(tone_state_.saturation_)));
  }
  if (blacks_slider_) {
    blacks_slider_->setValue(static_cast<int>(std::lround(tone_state_.blacks_)));
  }
  if (whites_slider_) {
    whites_slider_->setValue(static_cast<int>(std::lround(tone_state_.whites_)));
  }
  if (shadows_slider_) {
    shadows_slider_->setValue(static_cast<int>(std::lround(tone_state_.shadows_)));
  }
  if (highlights_slider_) {
    highlights_slider_->setValue(static_cast<int>(std::lround(tone_state_.highlights_)));
  }
  if (sharpen_slider_) {
    sharpen_slider_->setValue(static_cast<int>(std::lround(tone_state_.sharpen_)));
  }
  if (clarity_slider_) {
    clarity_slider_->setValue(static_cast<int>(std::lround(tone_state_.clarity_)));
  }
  if (curve_widget_) {
    curve_widget_->SetControlPoints(tone_state_.curve_points_);
  }
  SyncColorTempControlsFromDialogState();
  local_syncing_ = prev;
}

void ToneControlPanelWidget::SyncColorTempControlsFromDialogState() {
  if (!deps_.dialog_state) {
    return;
  }
  const auto& s    = *deps_.dialog_state;
  const bool  prev = local_syncing_;
  local_syncing_   = true;
  if (color_temp_mode_combo_) {
    color_temp_mode_combo_->setCurrentIndex(ColorTempModeToComboIndex(s.color_temp_mode_));
  }
  if (color_temp_cct_slider_) {
    color_temp_cct_slider_->setValue(ColorTempCctToSliderPos(DisplayedColorTempCct(s)));
    color_temp_cct_slider_->setEnabled(s.color_temp_supported_);
  }
  if (color_temp_tint_slider_) {
    color_temp_tint_slider_->setValue(static_cast<int>(std::lround(DisplayedColorTempTint(s))));
    color_temp_tint_slider_->setEnabled(s.color_temp_supported_);
  }
  if (color_temp_unsupported_label_) {
    color_temp_unsupported_label_->setVisible(!s.color_temp_supported_);
  }
  local_syncing_ = prev;
}

void ToneControlPanelWidget::RetranslateColorTempModeCombo() {
  if (!color_temp_mode_combo_) {
    return;
  }
  const int  current_value = color_temp_mode_combo_->currentData().toInt();
  const bool prev          = local_syncing_;
  local_syncing_           = true;
  color_temp_mode_combo_->clear();
  color_temp_mode_combo_->addItem(Tr("As Shot"), static_cast<int>(ColorTempMode::AS_SHOT));
  color_temp_mode_combo_->addItem(Tr("Custom"), static_cast<int>(ColorTempMode::CUSTOM));
  const int index = color_temp_mode_combo_->findData(current_value);
  color_temp_mode_combo_->setCurrentIndex(std::max(0, index));
  local_syncing_ = prev;
}

void ToneControlPanelWidget::LoadFromPipeline() {
  if (!deps_.session) {
    return;
  }
  if (deps_.session->LoadFromPipeline()) {
    PullToneStateFromDialog();
    PullCommittedToneStateFromDialog();
  }
}

void ToneControlPanelWidget::ReloadFromCommittedState() { SyncControlsFromDialogState(); }

}  // namespace alcedo::ui
