//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/widgets/display_transform_panel_widget.hpp"

#include <QKeyEvent>

#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"

namespace alcedo::ui {

DisplayTransformPanelWidget::DisplayTransformPanelWidget(QWidget* parent) : QWidget(parent) {}

namespace {

class AccordionHeader final : public QFrame {
 public:
  explicit AccordionHeader(QWidget* parent = nullptr) : QFrame(parent) {
    setCursor(Qt::PointingHandCursor);
    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_StyledBackground, true);
  }

  void SetOnActivated(std::function<void()> on_activated) {
    on_activated_ = std::move(on_activated);
  }

 protected:
  void mouseReleaseEvent(QMouseEvent* event) override {
    if (event->button() == Qt::LeftButton && rect().contains(event->pos())) {
      Activate();
      event->accept();
      return;
    }
    QFrame::mouseReleaseEvent(event);
  }

  void keyPressEvent(QKeyEvent* event) override {
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter ||
        event->key() == Qt::Key_Space) {
      Activate();
      event->accept();
      return;
    }
    QFrame::keyPressEvent(event);
  }

 private:
  void Activate() {
    if (on_activated_) {
      on_activated_();
    }
  }

  std::function<void()> on_activated_{};
};

}  // namespace

void EditorDialog::BuildDisplayTransformPanel() {
  auto* controls_header = NewLocalizedLabel("Display Rendering Transform", drt_controls_);
  controls_header->setObjectName("SectionTitle");
  controls_header->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
  AppTheme::MarkFontRole(controls_header, AppTheme::FontRole::UiHeadline);
  drt_controls_layout_->insertWidget(0, controls_header, 0);

  auto addDrtSection = [&](const char* title_source, const char* subtitle_source) {
    auto* frame = new QFrame(drt_controls_);
    frame->setObjectName("EditorSection");
    auto* v = new QVBoxLayout(frame);
    v->setContentsMargins(12, 10, 12, 10);
    v->setSpacing(2);

    auto* t = NewLocalizedLabel(title_source, frame);
    t->setObjectName("EditorSectionTitle");
    auto* s = NewLocalizedLabel(subtitle_source, frame);
    s->setObjectName("EditorSectionSub");
    s->setWordWrap(true);
    v->addWidget(t, 0);
    v->addWidget(s, 0);
    drt_controls_layout_->insertWidget(drt_controls_layout_->count() - 1, frame);
  };

  auto addDrtComboBox = [&](QWidget* parent, QVBoxLayout* parent_layout, const char* name_source,
                            auto options, auto current_value, auto&& onChange) {
    auto* label = NewLocalizedLabel(name_source, parent);
    label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    label->setWordWrap(true);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    auto* combo = new QComboBox(parent);
    for (const auto& option : options) {
      combo->addItem(Tr(option.label_), static_cast<int>(option.value_));
    }
    const int current_index = combo->findData(static_cast<int>(current_value));
    combo->setCurrentIndex(std::max(0, current_index));
    combo->setMinimumWidth(0);
    combo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    combo->setFixedHeight(32);
    combo->setStyleSheet(AppTheme::EditorComboBoxStyle());

    QObject::connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), parent,
                     [this, combo, onChange = std::forward<decltype(onChange)>(onChange)](int idx) {
                       if (syncing_controls_ || idx < 0) {
                         return;
                       }
                       onChange(combo->itemData(idx).toInt());
                     });

    auto* row       = new QWidget(parent);
    auto* rowLayout = new QVBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->setSpacing(4);
    rowLayout->addWidget(label, 0);
    rowLayout->addWidget(combo, 1);
    if (parent_layout == drt_controls_layout_) {
      parent_layout->insertWidget(parent_layout->count() - 1, row);
    } else {
      parent_layout->addWidget(row, 0);
    }
    return combo;
  };

  auto addDrtSlider = [&](const char* name_source, int min, int max, int value, auto&& onChange,
                          auto&& onRelease, auto&& onReset, const QString& suffix) {
    auto* info = NewLocalizedLabel(name_source, drt_controls_);
    info->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    info->setMinimumWidth(0);
    info->setWordWrap(true);
    info->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    auto* slider = new QSlider(Qt::Horizontal, drt_controls_);
    slider->setRange(min, max);
    slider->setValue(value);
    slider->setSingleStep(1);
    slider->setPageStep(std::max(1, (max - min) / 20));
    slider->setMinimumWidth(0);
    slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    slider->setFixedHeight(32);

    auto* spin = new QSpinBox(drt_controls_);
    spin->setRange(min, max);
    spin->setValue(value);
    spin->setSuffix(suffix);
    spin->setStyleSheet(AppTheme::EditorSpinBoxStyle());
    AppTheme::MarkFontRole(spin, AppTheme::FontRole::DataBody);
    spin->setFixedWidth(80);
    spin->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    QObject::connect(slider, &QSlider::valueChanged, drt_controls_, [this, spin, onChange](int v) {
      const QSignalBlocker blocker(spin);
      spin->setValue(v);
      if (syncing_controls_) {
        return;
      }
      onChange(v);
    });

    QObject::connect(spin, QOverload<int>::of(&QSpinBox::valueChanged), drt_controls_,
                     [this, slider, onChange](int v) {
                       const QSignalBlocker blocker(slider);
                       slider->setValue(v);
                       if (syncing_controls_) {
                         return;
                       }
                       onChange(v);
                     });

    // To ensure that typing + pressing Enter (or losing focus) commits the adjustment.
    QObject::connect(spin, &QSpinBox::editingFinished, drt_controls_, [this, onRelease]() {
      if (syncing_controls_) {
        return;
      }
      onRelease();
    });

    QObject::connect(slider, &QSlider::sliderReleased, drt_controls_, [this, onRelease]() {
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

    auto* row       = new QWidget(drt_controls_);
    auto* rowLayout = new QVBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->setSpacing(4);
    rowLayout->addWidget(info, 0);

    auto* value_row        = new QWidget(row);
    auto* value_row_layout = new QHBoxLayout(value_row);
    value_row_layout->setContentsMargins(0, 0, 0, 0);
    value_row_layout->setSpacing(8);
    value_row_layout->addWidget(slider, 1);
    value_row_layout->addWidget(spin, 0);
    rowLayout->addWidget(value_row, 1);

    drt_controls_layout_->insertWidget(drt_controls_layout_->count() - 1, row);
    return slider;
  };

  auto addOpenDrtFloatSlider = [&](QWidget* parent, QVBoxLayout* parent_layout,
                                   const char* name_source, float min, float max, float step,
                                   float value, auto&& getter, auto&& onChange, auto&& onRelease,
                                   auto&& onReset, const QString& suffix) {
    const float scale = 1.0f / step;
    auto*       info  = NewLocalizedLabel(name_source, parent);
    info->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    AppTheme::MarkFontRole(info, AppTheme::FontRole::UiHint);
    info->setMinimumWidth(0);
    info->setWordWrap(false);
    info->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

    auto* slider = new QSlider(Qt::Horizontal, parent);
    slider->setRange(static_cast<int>(std::lround(min * scale)),
                     static_cast<int>(std::lround(max * scale)));
    slider->setValue(static_cast<int>(std::lround(value * scale)));
    slider->setSingleStep(1);
    slider->setPageStep(std::max(1, (slider->maximum() - slider->minimum()) / 20));
    slider->setMinimumWidth(0);
    slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    slider->setFixedHeight(22);

    auto* spin = new QDoubleSpinBox(parent);
    spin->setRange(min, max);
    spin->setDecimals(step < 0.001f ? 4 : step < 0.01f ? 3 : 2);
    spin->setSingleStep(static_cast<double>(step));
    spin->setValue(value);
    spin->setSuffix(suffix);
    spin->setStyleSheet(AppTheme::EditorSpinBoxStyle());
    AppTheme::MarkFontRole(spin, AppTheme::FontRole::DataCaption);
    spin->setFixedSize(82, 24);
    spin->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    QObject::connect(slider, &QSlider::valueChanged, parent, [this, spin, scale, onChange](int v) {
      const float          value = static_cast<float>(v) / scale;
      const QSignalBlocker blocker(spin);
      spin->setValue(value);
      if (syncing_controls_) {
        return;
      }
      onChange(value);
    });

    QObject::connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), parent,
                     [this, slider, scale, onChange](double v) {
                       const int slider_value =
                           static_cast<int>(std::lround(static_cast<float>(v) * scale));
                       const QSignalBlocker blocker(slider);
                       slider->setValue(slider_value);
                       if (syncing_controls_) {
                         return;
                       }
                       onChange(static_cast<float>(v));
                     });

    QObject::connect(spin, &QDoubleSpinBox::editingFinished, parent, [this, onRelease]() {
      if (syncing_controls_) {
        return;
      }
      onRelease();
    });

    QObject::connect(slider, &QSlider::sliderReleased, parent, [this, onRelease]() {
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

    auto* row       = new QWidget(parent);
    auto* rowLayout = new QVBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->setSpacing(1);
    rowLayout->addWidget(info, 0);

    auto* value_row        = new QWidget(row);
    auto* value_row_layout = new QHBoxLayout(value_row);
    value_row_layout->setContentsMargins(0, 0, 0, 0);
    value_row_layout->setSpacing(6);
    value_row_layout->addWidget(slider, 1);
    value_row_layout->addWidget(spin, 0);
    rowLayout->addWidget(value_row, 1);

    parent_layout->addWidget(row, 0);
    odt_open_drt_detail_sliders_.push_back(
        {slider, spin, min, max, scale, std::forward<decltype(getter)>(getter)});
    return slider;
  };

  odt_encoding_space_combo_ = addDrtComboBox(
      drt_controls_, drt_controls_layout_, "Encoding Space", kDisplayEncodingSpaceOptions,
      state_.odt_.encoding_space_, [this](int value) {
        state_.odt_.encoding_space_ = static_cast<ColorUtils::ColorSpace>(value);
        if (!IsSupportedDisplayEncoding(state_.odt_.encoding_space_, state_.odt_.encoding_eotf_)) {
          state_.odt_.encoding_eotf_ = DefaultDisplayEotfForSpace(state_.odt_.encoding_space_);
        }
        RefreshOdtEncodingEotfComboFromState();
        frame_manager_.SyncViewerDisplayEncoding(state_.odt_.encoding_space_,
                                                 state_.odt_.encoding_eotf_);
        RequestRender();
        CommitAdjustment(AdjustmentField::Odt);
      });

  odt_encoding_eotf_combo_ = new QComboBox(drt_controls_);
  odt_encoding_eotf_combo_->setMinimumWidth(0);
  odt_encoding_eotf_combo_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  odt_encoding_eotf_combo_->setFixedHeight(32);
  odt_encoding_eotf_combo_->setStyleSheet(AppTheme::EditorComboBoxStyle());
  QObject::connect(odt_encoding_eotf_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                   drt_controls_, [this](int idx) {
                     if (syncing_controls_ || idx < 0 || !odt_encoding_eotf_combo_) {
                       return;
                     }
                     state_.odt_.encoding_eotf_ = static_cast<ColorUtils::EOTF>(
                         odt_encoding_eotf_combo_->itemData(idx).toInt());
                     frame_manager_.SyncViewerDisplayEncoding(state_.odt_.encoding_space_,
                                                              state_.odt_.encoding_eotf_);
                     RequestRender();
                     CommitAdjustment(AdjustmentField::Odt);
                   });
  {
    auto* label = NewLocalizedLabel("Encoding EOTF", drt_controls_);
    label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    label->setWordWrap(true);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    auto* row       = new QWidget(drt_controls_);
    auto* rowLayout = new QVBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->setSpacing(4);
    rowLayout->addWidget(label, 0);
    rowLayout->addWidget(odt_encoding_eotf_combo_, 1);
    drt_controls_layout_->insertWidget(drt_controls_layout_->count() - 1, row);
  }
  RefreshOdtEncodingEotfComboFromState();

  odt_peak_luminance_slider_ = addDrtSlider(
      "Peak Luminance", 100, 1000, static_cast<int>(std::lround(state_.odt_.peak_luminance_)),
      [this](int value) {
        state_.odt_.peak_luminance_ = static_cast<float>(value);
        RequestRender();
      },
      [this]() { CommitAdjustment(AdjustmentField::Odt); },
      [this]() {
        ResetFieldToDefault(AdjustmentField::Odt, [this](const AdjustmentState& defaults) {
          state_.odt_.peak_luminance_ = defaults.odt_.peak_luminance_;
        });
      },
      " nits");

  {
    auto* frame = new QFrame(drt_controls_);
    frame->setObjectName("EditorSection");
    auto* layout = new QVBoxLayout(frame);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->setSpacing(10);

    auto* title = NewLocalizedLabel("Rendering Method", frame);
    title->setObjectName("EditorSectionTitle");
    auto* sub = NewLocalizedLabel(
        "Choose the transform family. Shared encoding settings stay above; method-specific "
        "settings stay preserved per method.",
        frame);
    sub->setObjectName("EditorSectionSub");
    sub->setWordWrap(true);
    layout->addWidget(title, 0);
    layout->addWidget(sub, 0);

    auto* cards_row    = new QWidget(frame);
    auto* cards_layout = new QVBoxLayout(cards_row);
    cards_layout->setContentsMargins(0, 0, 0, 0);
    cards_layout->setSpacing(12);

    odt_aces_method_card_     = NewLocalizedButton("ACES 2.0", cards_row);
    odt_open_drt_method_card_ = NewLocalizedButton("OpenDRT", cards_row);
    for (QPushButton* card : {odt_aces_method_card_, odt_open_drt_method_card_}) {
      card->setCheckable(true);
      card->setCursor(Qt::PointingHandCursor);
      card->setMinimumHeight(76);
      card->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
      AppTheme::MarkFontRole(card, AppTheme::FontRole::UiTitle);
    }

    QObject::connect(odt_aces_method_card_, &QPushButton::clicked, frame, [this]() {
      state_.odt_.method_ = ColorUtils::ODTMethod::ACES_2_0;
      RefreshOdtMethodUi();
      RequestRender();
      CommitAdjustment(AdjustmentField::Odt);
    });
    QObject::connect(odt_open_drt_method_card_, &QPushButton::clicked, frame, [this]() {
      state_.odt_.method_ = ColorUtils::ODTMethod::OPEN_DRT;
      RefreshOdtMethodUi();
      RequestRender();
      CommitAdjustment(AdjustmentField::Odt);
    });

    cards_layout->addWidget(odt_aces_method_card_, 1);
    cards_layout->addWidget(odt_open_drt_method_card_, 1);
    layout->addWidget(cards_row, 0);

    odt_method_stack_ = new QStackedWidget(frame);
    odt_method_stack_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    auto* aces_page   = new QWidget(odt_method_stack_);
    auto* aces_layout = new QVBoxLayout(aces_page);
    aces_layout->setContentsMargins(0, 4, 0, 0);
    aces_layout->setSpacing(8);
    odt_aces_limiting_space_combo_ = addDrtComboBox(
        aces_page, aces_layout, "Limiting Space", kAcesLimitingSpaceOptions,
        state_.odt_.aces_.limiting_space_, [this](int value) {
          state_.odt_.aces_.limiting_space_ = static_cast<ColorUtils::ColorSpace>(value);
          RequestRender();
          CommitAdjustment(AdjustmentField::Odt);
        });
    odt_method_stack_->addWidget(aces_page);

    auto* open_drt_page   = new QWidget(odt_method_stack_);
    auto* open_drt_layout = new QVBoxLayout(open_drt_page);
    open_drt_layout->setContentsMargins(0, 4, 0, 0);
    open_drt_layout->setSpacing(8);
    odt_open_drt_look_preset_combo_ = addDrtComboBox(
        open_drt_page, open_drt_layout, "Look Preset", kOpenDrtLookPresetOptions,
        state_.odt_.open_drt_.look_preset_, [this](int value) {
          state_.odt_.open_drt_.look_preset_ = static_cast<odt_cpu::OpenDRTLookPreset>(value);
          if (state_.odt_.open_drt_.look_preset_ != odt_cpu::OpenDRTLookPreset::CUSTOM) {
            odt_cpu::ApplyOpenDRTLookPresetToSettings(state_.odt_.open_drt_.look_preset_,
                                                      &state_.odt_.open_drt_);
            if (state_.odt_.open_drt_.tonescale_preset_ !=
                odt_cpu::OpenDRTTonescalePreset::CUSTOM) {
              odt_cpu::ApplyOpenDRTTonescalePresetToSettings(
                  state_.odt_.open_drt_.tonescale_preset_, &state_.odt_.open_drt_);
            }
            SyncOpenDrtDetailControlsFromState();
          }
          RequestRender();
          CommitAdjustment(AdjustmentField::Odt);
        });
    odt_open_drt_tonescale_preset_combo_ = addDrtComboBox(
        open_drt_page, open_drt_layout, "Tonescale Preset", kOpenDrtTonescaleOptions,
        state_.odt_.open_drt_.tonescale_preset_, [this](int value) {
          state_.odt_.open_drt_.tonescale_preset_ =
              static_cast<odt_cpu::OpenDRTTonescalePreset>(value);
          if (state_.odt_.open_drt_.tonescale_preset_ != odt_cpu::OpenDRTTonescalePreset::CUSTOM) {
            odt_cpu::ApplyOpenDRTTonescalePresetToSettings(state_.odt_.open_drt_.tonescale_preset_,
                                                           &state_.odt_.open_drt_);
            SyncOpenDrtDetailControlsFromState();
          }
          RequestRender();
          CommitAdjustment(AdjustmentField::Odt);
        });
    odt_open_drt_creative_white_combo_ = addDrtComboBox(
        open_drt_page, open_drt_layout, "Creative White", kOpenDrtCreativeWhiteOptions,
        state_.odt_.open_drt_.creative_white_, [this](int value) {
          state_.odt_.open_drt_.creative_white_ =
              static_cast<odt_cpu::OpenDRTCreativeWhitePreset>(value);
          RequestRender();
          CommitAdjustment(AdjustmentField::Odt);
        });

    const auto& theme            = AppTheme::Instance();

    auto*       detail_accordion = new QFrame(open_drt_page);
    detail_accordion->setObjectName("EditorSection");
    auto* accordion_layout = new QVBoxLayout(detail_accordion);
    accordion_layout->setContentsMargins(0, 0, 0, 0);
    accordion_layout->setSpacing(0);

    auto* detail_header = new AccordionHeader(detail_accordion);
    detail_header->setObjectName("OpenDrtDetailAccordionHeader");
    detail_header->setStyleSheet(
        QStringLiteral("QFrame#OpenDrtDetailAccordionHeader {"
                       "  background: transparent;"
                       "  border: none;"
                       "  border-radius: 10px;"
                       "}"
                       "QFrame#OpenDrtDetailAccordionHeader:hover {"
                       "  background: %1;"
                       "}")
            .arg(QColor(theme.bgPanelColor().red(), theme.bgPanelColor().green(),
                        theme.bgPanelColor().blue(), 170)
                     .name(QColor::HexArgb)));
    SetLocalizedToolTip(detail_header, "Advanced Parameters");

    auto* detail_header_layout = new QHBoxLayout(detail_header);
    detail_header_layout->setContentsMargins(10, 8, 10, 8);
    detail_header_layout->setSpacing(8);

    auto* detail_title = NewLocalizedLabel("Advanced Parameters", detail_header);
    detail_title->setObjectName("EditorSectionTitle");
    detail_title->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    AppTheme::MarkFontRole(detail_title, AppTheme::FontRole::UiCaptionStrong);

    auto* detail_chevron = new QLabel(QStringLiteral(">"), detail_header);
    detail_chevron->setObjectName("OpenDrtDetailAccordionChevron");
    detail_chevron->setAlignment(Qt::AlignCenter);
    detail_chevron->setFixedWidth(16);
    detail_chevron->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    detail_chevron->setStyleSheet(AppTheme::EditorLabelStyle(theme.textMutedColor()));
    AppTheme::MarkFontRole(detail_chevron, AppTheme::FontRole::UiCaptionStrong);

    detail_header_layout->addWidget(detail_title, 1);
    detail_header_layout->addWidget(detail_chevron, 0);
    accordion_layout->addWidget(detail_header, 0);

    odt_open_drt_detail_panel_ = new QFrame(detail_accordion);
    auto* detail_layout        = new QVBoxLayout(odt_open_drt_detail_panel_);
    detail_layout->setContentsMargins(8, 0, 8, 10);
    detail_layout->setSpacing(8);
    odt_open_drt_detail_panel_->setVisible(false);
    accordion_layout->addWidget(odt_open_drt_detail_panel_, 0);

    auto set_detail_expanded = [this, detail_chevron](bool expanded) {
      if (odt_open_drt_detail_panel_) {
        odt_open_drt_detail_panel_->setVisible(expanded);
      }
      detail_chevron->setText(expanded ? QStringLiteral("v") : QStringLiteral(">"));
      RefreshOdtMethodUi();
    };
    detail_header->SetOnActivated([this, set_detail_expanded]() {
      set_detail_expanded(!odt_open_drt_detail_panel_->isVisible());
    });

    auto addDetailSection = [&](const char* title_source) {
      auto* section        = new QWidget(odt_open_drt_detail_panel_);
      auto* section_layout = new QVBoxLayout(section);
      section_layout->setContentsMargins(0, 0, 0, 0);
      section_layout->setSpacing(5);

      auto* title = NewLocalizedLabel(title_source, section);
      title->setObjectName("EditorSectionTitle");
      AppTheme::MarkFontRole(title, AppTheme::FontRole::UiCaptionStrong);
      section_layout->addWidget(title, 0);
      detail_layout->addWidget(section, 0);
      return section_layout;
    };

    auto commitOdt = [this]() { CommitAdjustment(AdjustmentField::Odt); };
    auto resetOdt  = [this]() {
      ResetFieldToDefault(AdjustmentField::Odt, [this](const AdjustmentState& defaults) {
        state_.odt_.open_drt_ = defaults.odt_.open_drt_;
      });
    };
    auto toneChanged = [this](float& target, float value) {
      MarkOpenDrtTonescalePresetCustomForEditing();
      target = value;
      RequestRender();
    };
    auto lookChanged = [this](float& target, float value) {
      MarkOpenDrtLookPresetCustomForEditing();
      target = value;
      RequestRender();
    };
    auto addLookDetail = [&](QVBoxLayout* section, const char* label, float min, float max,
                             float step, float odt_cpu::OpenDRTDetailedSettings::* member) {
      addOpenDrtFloatSlider(
          open_drt_page, section, label, min, max, step, state_.odt_.open_drt_.detailed_.*member,
          [member](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.*member; },
          [this, lookChanged, member](float v) {
            lookChanged(state_.odt_.open_drt_.detailed_.*member, v);
          },
          commitOdt, resetOdt, QString());
    };

    auto* tonescale_section = addDetailSection("Tonescale");
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Contrast", 1.0f, 2.0f, 0.01f,
        state_.odt_.open_drt_.detailed_.tn_con_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.tn_con_; },
        [this, toneChanged](float v) { toneChanged(state_.odt_.open_drt_.detailed_.tn_con_, v); },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Shoulder Clip", 0.0f, 1.0f, 0.01f,
        state_.odt_.open_drt_.detailed_.tn_sh_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.tn_sh_; },
        [this, toneChanged](float v) { toneChanged(state_.odt_.open_drt_.detailed_.tn_sh_, v); },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Toe", 0.0f, 0.1f, 0.001f,
        state_.odt_.open_drt_.detailed_.tn_toe_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.tn_toe_; },
        [this, toneChanged](float v) { toneChanged(state_.odt_.open_drt_.detailed_.tn_toe_, v); },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Offset", 0.0f, 0.02f, 0.0002f,
        state_.odt_.open_drt_.detailed_.tn_off_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.tn_off_; },
        [this, toneChanged](float v) { toneChanged(state_.odt_.open_drt_.detailed_.tn_off_, v); },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Contrast High", -1.0f, 1.0f, 0.02f,
        state_.odt_.open_drt_.detailed_.tn_hcon_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.tn_hcon_; },
        [this, toneChanged](float v) { toneChanged(state_.odt_.open_drt_.detailed_.tn_hcon_, v); },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Contrast High Pivot", 0.0f, 4.0f, 0.04f,
        state_.odt_.open_drt_.detailed_.tn_hcon_pv_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.tn_hcon_pv_; },
        [this, toneChanged](float v) {
          toneChanged(state_.odt_.open_drt_.detailed_.tn_hcon_pv_, v);
        },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Contrast High Strength", 0.0f, 4.0f, 0.04f,
        state_.odt_.open_drt_.detailed_.tn_hcon_st_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.tn_hcon_st_; },
        [this, toneChanged](float v) {
          toneChanged(state_.odt_.open_drt_.detailed_.tn_hcon_st_, v);
        },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Contrast Low", 0.0f, 3.0f, 0.03f,
        state_.odt_.open_drt_.detailed_.tn_lcon_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.tn_lcon_; },
        [this, toneChanged](float v) { toneChanged(state_.odt_.open_drt_.detailed_.tn_lcon_, v); },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Contrast Low Width", 0.0f, 2.0f, 0.02f,
        state_.odt_.open_drt_.detailed_.tn_lcon_w_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.tn_lcon_w_; },
        [this, toneChanged](float v) {
          toneChanged(state_.odt_.open_drt_.detailed_.tn_lcon_w_, v);
        },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "Display Grey Luminance", 3.0f, 25.0f, 0.1f,
        state_.odt_.open_drt_.display_grey_luminance_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.display_grey_luminance_; },
        [this](float v) {
          state_.odt_.open_drt_.display_grey_luminance_ = v;
          RequestRender();
        },
        commitOdt, resetOdt, " nits");
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "HDR Grey Boost", 0.0f, 1.0f, 0.001f,
        state_.odt_.open_drt_.hdr_grey_boost_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.hdr_grey_boost_; },
        [this](float v) {
          state_.odt_.open_drt_.hdr_grey_boost_ = v;
          RequestRender();
        },
        commitOdt, resetOdt, QString());
    addOpenDrtFloatSlider(
        open_drt_page, tonescale_section, "HDR Purity", 0.0f, 1.0f, 0.01f,
        state_.odt_.open_drt_.hdr_purity_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.hdr_purity_; },
        [this](float v) {
          state_.odt_.open_drt_.hdr_purity_ = v;
          RequestRender();
        },
        commitOdt, resetOdt, QString());

    auto* purity_section = addDetailSection("Purity");
    addOpenDrtFloatSlider(
        open_drt_page, purity_section, "Creative White Limit", 0.0f, 1.0f, 0.01f,
        state_.odt_.open_drt_.detailed_.cwp_lm_,
        [](const odt_cpu::OpenDRTSettings& s) { return s.detailed_.cwp_lm_; },
        [this](float v) {
          MarkOpenDrtLookPresetCustomForEditing();
          state_.odt_.open_drt_.detailed_.cwp_lm_     = v;
          state_.odt_.open_drt_.creative_white_limit_ = v;
          RequestRender();
        },
        commitOdt, resetOdt, QString());
    addLookDetail(purity_section, "Render Space Strength", 0.0f, 0.6f, 0.006f,
                  &odt_cpu::OpenDRTDetailedSettings::rs_sa_);
    addLookDetail(purity_section, "Render Space Weight R", 0.0f, 0.8f, 0.008f,
                  &odt_cpu::OpenDRTDetailedSettings::rs_rw_);
    addLookDetail(purity_section, "Render Space Weight B", 0.0f, 0.8f, 0.008f,
                  &odt_cpu::OpenDRTDetailedSettings::rs_bw_);
    addLookDetail(purity_section, "Purity Limit Low", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::pt_lml_);
    addLookDetail(purity_section, "Purity Limit Low R", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::pt_lml_r_);
    addLookDetail(purity_section, "Purity Limit Low G", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::pt_lml_g_);
    addLookDetail(purity_section, "Purity Limit Low B", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::pt_lml_b_);
    addLookDetail(purity_section, "Purity Limit High", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::pt_lmh_);
    addLookDetail(purity_section, "Purity Limit High R", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::pt_lmh_r_);
    addLookDetail(purity_section, "Purity Limit High B", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::pt_lmh_b_);
    addLookDetail(purity_section, "Purity Softclip C", 0.0f, 0.25f, 0.0025f,
                  &odt_cpu::OpenDRTDetailedSettings::ptl_c_);
    addLookDetail(purity_section, "Purity Softclip M", 0.0f, 0.25f, 0.0025f,
                  &odt_cpu::OpenDRTDetailedSettings::ptl_m_);
    addLookDetail(purity_section, "Purity Softclip Y", 0.0f, 0.25f, 0.0025f,
                  &odt_cpu::OpenDRTDetailedSettings::ptl_y_);
    addLookDetail(purity_section, "Mid Purity Low", 0.0f, 2.0f, 0.02f,
                  &odt_cpu::OpenDRTDetailedSettings::ptm_low_);
    addLookDetail(purity_section, "Mid Purity Low Range", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::ptm_low_rng_);
    addLookDetail(purity_section, "Mid Purity Low Strength", 0.1f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::ptm_low_st_);
    addLookDetail(purity_section, "Mid Purity High", -0.9f, 0.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::ptm_high_);
    addLookDetail(purity_section, "Mid Purity High Range", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::ptm_high_rng_);
    addLookDetail(purity_section, "Mid Purity High Strength", 0.1f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::ptm_high_st_);

    auto* brilliance_section = addDetailSection("Brilliance");
    addLookDetail(brilliance_section, "Brilliance", -6.0f, 2.0f, 0.08f,
                  &odt_cpu::OpenDRTDetailedSettings::brl_);
    addLookDetail(brilliance_section, "Brilliance R", -6.0f, 2.0f, 0.08f,
                  &odt_cpu::OpenDRTDetailedSettings::brl_r_);
    addLookDetail(brilliance_section, "Brilliance G", -6.0f, 2.0f, 0.08f,
                  &odt_cpu::OpenDRTDetailedSettings::brl_g_);
    addLookDetail(brilliance_section, "Brilliance B", -6.0f, 2.0f, 0.08f,
                  &odt_cpu::OpenDRTDetailedSettings::brl_b_);
    addLookDetail(brilliance_section, "Brilliance Range", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::brl_rng_);
    addLookDetail(brilliance_section, "Brilliance Strength", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::brl_st_);
    addLookDetail(brilliance_section, "Brilliance Post", -1.0f, 0.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::brlp_);
    addLookDetail(brilliance_section, "Post Brilliance R", -3.0f, 0.0f, 0.03f,
                  &odt_cpu::OpenDRTDetailedSettings::brlp_r_);
    addLookDetail(brilliance_section, "Post Brilliance G", -3.0f, 0.0f, 0.03f,
                  &odt_cpu::OpenDRTDetailedSettings::brlp_g_);
    addLookDetail(brilliance_section, "Post Brilliance B", -3.0f, 0.0f, 0.03f,
                  &odt_cpu::OpenDRTDetailedSettings::brlp_b_);

    auto* hue_section = addDetailSection("Hue Shift");
    addLookDetail(hue_section, "Hue Contrast R", 0.0f, 2.0f, 0.02f,
                  &odt_cpu::OpenDRTDetailedSettings::hc_r_);
    addLookDetail(hue_section, "Hue Contrast R Range", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hc_r_rng_);
    addLookDetail(hue_section, "Hueshift R", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_r_);
    addLookDetail(hue_section, "Hueshift R Range", 0.0f, 2.0f, 0.02f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_r_rng_);
    addLookDetail(hue_section, "Hueshift G", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_g_);
    addLookDetail(hue_section, "Hueshift G Range", 0.0f, 2.0f, 0.02f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_g_rng_);
    addLookDetail(hue_section, "Hueshift B", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_b_);
    addLookDetail(hue_section, "Hueshift B Range", 0.0f, 4.0f, 0.02f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_b_rng_);
    addLookDetail(hue_section, "Hueshift C", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_c_);
    addLookDetail(hue_section, "Hueshift C Range", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_c_rng_);
    addLookDetail(hue_section, "Hueshift M", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_m_);
    addLookDetail(hue_section, "Hueshift M Range", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_m_rng_);
    addLookDetail(hue_section, "Hueshift Y", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_y_);
    addLookDetail(hue_section, "Hueshift Y Range", 0.0f, 1.0f, 0.01f,
                  &odt_cpu::OpenDRTDetailedSettings::hs_y_rng_);

    open_drt_layout->addWidget(detail_accordion, 0);
    odt_method_stack_->addWidget(open_drt_page);

    layout->addWidget(odt_method_stack_, 0);
    drt_controls_layout_->insertWidget(drt_controls_layout_->count() - 1, frame);
  }

  RefreshOdtMethodUi();
}

}  // namespace alcedo::ui
