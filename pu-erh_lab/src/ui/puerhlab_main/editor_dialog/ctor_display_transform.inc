// ctor_display_transform.inc -- Constructor section for EditorDialog.
// This file is #included from the constructor body in dialog.cpp.
// It shares the constructor scope and has access to all local variables.

    auto addDrtSection = [&](const QString& title, const QString& subtitle) {
      auto* frame = new QFrame(drt_controls_);
      frame->setObjectName("EditorSection");
      auto* v = new QVBoxLayout(frame);
      v->setContentsMargins(12, 10, 12, 10);
      v->setSpacing(2);

      auto* t = new QLabel(title, frame);
      t->setObjectName("EditorSectionTitle");
      auto* s = new QLabel(subtitle, frame);
      s->setObjectName("EditorSectionSub");
      s->setWordWrap(true);
      v->addWidget(t, 0);
      v->addWidget(s, 0);
      drt_controls_layout->insertWidget(drt_controls_layout->count() - 1, frame);
    };

    auto addDrtComboBox = [&](QWidget* parent, QVBoxLayout* parent_layout, const QString& name,
                              auto options, auto current_value, auto&& onChange) {
      auto* label = new QLabel(name, parent);
      label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));

      auto* combo = new QComboBox(parent);
      for (const auto& option : options) {
        combo->addItem(Tr(option.label_), static_cast<int>(option.value_));
      }
      const int current_index = combo->findData(static_cast<int>(current_value));
      combo->setCurrentIndex(std::max(0, current_index));
      combo->setMinimumWidth(240);
      combo->setFixedHeight(32);
      combo->setStyleSheet(AppTheme::EditorComboBoxStyle());

      QObject::connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), parent,
                       [this, combo,
                        onChange = std::forward<decltype(onChange)>(onChange)](int idx) {
                         if (syncing_controls_ || idx < 0) {
                           return;
                         }
                         onChange(combo->itemData(idx).toInt());
                       });

      auto* row       = new QWidget(parent);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(label, 1);
      rowLayout->addWidget(combo);
      if (parent_layout == drt_controls_layout) {
        parent_layout->insertWidget(parent_layout->count() - 1, row);
      } else {
        parent_layout->addWidget(row, 0);
      }
      return combo;
    };

    auto addDrtSlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                            auto&& onRelease, auto&& onReset, const QString& suffix) {
      auto* info = new QLabel(name, drt_controls_);
      info->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
      info->setMinimumWidth(100);

      auto* slider = new QSlider(Qt::Horizontal, drt_controls_);
      slider->setRange(min, max);
      slider->setValue(value);
      slider->setSingleStep(1);
      slider->setPageStep(std::max(1, (max - min) / 20));
      slider->setMinimumWidth(200);
      slider->setFixedHeight(32);

      auto* spin = new QSpinBox(drt_controls_);
      spin->setRange(min, max);
      spin->setValue(value);
      spin->setSuffix(suffix);
      spin->setStyleSheet(AppTheme::EditorSpinBoxStyle());
      AppTheme::MarkFontRole(spin, AppTheme::FontRole::DataBody);
      spin->setFixedWidth(80);
      spin->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

      QObject::connect(slider, &QSlider::valueChanged, drt_controls_,
                       [this, spin, onChange](int v) {
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
      QObject::connect(spin, &QSpinBox::editingFinished, drt_controls_,
                       [this, onRelease]() {
                         if (syncing_controls_) {
                           return;
                         }
                         onRelease();
                       });

      QObject::connect(slider, &QSlider::sliderReleased, drt_controls_,
                       [this, onRelease]() {
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

      auto* row       = new QWidget(drt_controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(8);
      rowLayout->addWidget(info, 0);
      rowLayout->addWidget(slider, 1);
      rowLayout->addWidget(spin, 0);

      drt_controls_layout->insertWidget(drt_controls_layout->count() - 1, row);
      return slider;
    };

    addDrtSection(Tr("Display Rendering Transform"),
                  Tr("Shared display encoding controls with switchable ACES 2.0 and OpenDRT "
                     "method presets."));

    odt_encoding_space_combo_ = addDrtComboBox(
        drt_controls_, drt_controls_layout, Tr("Encoding Space"), kDisplayEncodingSpaceOptions,
        state_.odt_.encoding_space_, [this](int value) {
          state_.odt_.encoding_space_ = static_cast<ColorUtils::ColorSpace>(value);
          if (!IsSupportedDisplayEncoding(state_.odt_.encoding_space_,
                                          state_.odt_.encoding_eotf_)) {
            state_.odt_.encoding_eotf_ = DefaultDisplayEotfForSpace(state_.odt_.encoding_space_);
          }
          RefreshOdtEncodingEotfComboFromState();
          RequestRender();
          CommitAdjustment(AdjustmentField::Odt);
        });

    odt_encoding_eotf_combo_ = new QComboBox(drt_controls_);
    odt_encoding_eotf_combo_->setMinimumWidth(240);
    odt_encoding_eotf_combo_->setFixedHeight(32);
    odt_encoding_eotf_combo_->setStyleSheet(AppTheme::EditorComboBoxStyle());
    QObject::connect(odt_encoding_eotf_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                     drt_controls_, [this](int idx) {
                       if (syncing_controls_ || idx < 0 || !odt_encoding_eotf_combo_) {
                         return;
                       }
                       state_.odt_.encoding_eotf_ = static_cast<ColorUtils::EOTF>(
                           odt_encoding_eotf_combo_->itemData(idx).toInt());
                       RequestRender();
                       CommitAdjustment(AdjustmentField::Odt);
                     });
    {
      auto* label = new QLabel(Tr("Encoding EOTF"), drt_controls_);
      label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
      auto* row       = new QWidget(drt_controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(label, 1);
      rowLayout->addWidget(odt_encoding_eotf_combo_);
      drt_controls_layout->insertWidget(drt_controls_layout->count() - 1, row);
    }
    RefreshOdtEncodingEotfComboFromState();

    odt_peak_luminance_slider_ = addDrtSlider(
        Tr("Peak Luminance"), 100, 1000,
        static_cast<int>(std::lround(state_.odt_.peak_luminance_)),
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

      auto* title = new QLabel(Tr("Rendering Method"), frame);
      title->setObjectName("EditorSectionTitle");
      auto* sub = new QLabel(
          Tr("Choose the transform family. Shared encoding settings stay above; method-specific "
             "settings stay preserved per method."),
          frame);
      sub->setObjectName("EditorSectionSub");
      sub->setWordWrap(true);
      layout->addWidget(title, 0);
      layout->addWidget(sub, 0);

      auto* cards_row = new QWidget(frame);
      auto* cards_layout = new QVBoxLayout(cards_row);
      cards_layout->setContentsMargins(0, 0, 0, 0);
      cards_layout->setSpacing(12);

      odt_aces_method_card_ = new QPushButton(
          Tr("ACES 2.0"),
          cards_row);
      odt_open_drt_method_card_ = new QPushButton(
          Tr("OpenDRT"),
          cards_row);
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

      auto* aces_page = new QWidget(odt_method_stack_);
      auto* aces_layout = new QVBoxLayout(aces_page);
      aces_layout->setContentsMargins(0, 4, 0, 0);
      aces_layout->setSpacing(8);
      odt_aces_limiting_space_combo_ = addDrtComboBox(
          aces_page, aces_layout, Tr("Limiting Space"), kAcesLimitingSpaceOptions,
          state_.odt_.aces_.limiting_space_, [this](int value) {
            state_.odt_.aces_.limiting_space_ = static_cast<ColorUtils::ColorSpace>(value);
            RequestRender();
            CommitAdjustment(AdjustmentField::Odt);
          });
      aces_layout->addStretch();
      odt_method_stack_->addWidget(aces_page);

      auto* open_drt_page = new QWidget(odt_method_stack_);
      auto* open_drt_layout = new QVBoxLayout(open_drt_page);
      open_drt_layout->setContentsMargins(0, 4, 0, 0);
      open_drt_layout->setSpacing(8);
      odt_open_drt_look_preset_combo_ = addDrtComboBox(
          open_drt_page, open_drt_layout, Tr("Look Preset"), kOpenDrtLookPresetOptions,
          state_.odt_.open_drt_.look_preset_, [this](int value) {
            state_.odt_.open_drt_.look_preset_ =
                static_cast<odt_cpu::OpenDRTLookPreset>(value);
            RequestRender();
            CommitAdjustment(AdjustmentField::Odt);
          });
      odt_open_drt_tonescale_preset_combo_ = addDrtComboBox(
          open_drt_page, open_drt_layout, Tr("Tonescale Preset"), kOpenDrtTonescaleOptions,
          state_.odt_.open_drt_.tonescale_preset_, [this](int value) {
            state_.odt_.open_drt_.tonescale_preset_ =
                static_cast<odt_cpu::OpenDRTTonescalePreset>(value);
            RequestRender();
            CommitAdjustment(AdjustmentField::Odt);
          });
      odt_open_drt_creative_white_combo_ = addDrtComboBox(
          open_drt_page, open_drt_layout, Tr("Creative White"), kOpenDrtCreativeWhiteOptions,
          state_.odt_.open_drt_.creative_white_, [this](int value) {
            state_.odt_.open_drt_.creative_white_ =
                static_cast<odt_cpu::OpenDRTCreativeWhitePreset>(value);
            RequestRender();
            CommitAdjustment(AdjustmentField::Odt);
          });
      open_drt_layout->addStretch();
      odt_method_stack_->addWidget(open_drt_page);

      layout->addWidget(odt_method_stack_, 0);
      drt_controls_layout->insertWidget(drt_controls_layout->count() - 1, frame);
    }

    RefreshOdtMethodUi();
