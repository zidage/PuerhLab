// ctor_tone_color.inc -- Constructor section for EditorDialog.
// This file is #included from the constructor body in dialog.cpp.
// It shares the constructor scope and has access to all local variables.

    auto* controls_header = new QLabel("Adjustments", controls_);
    controls_header->setObjectName("SectionTitle");
    controls_layout->addWidget(controls_header, 0);

    // Prefer LUTs next to the executable (installed layout), fall back to source tree.
    const auto app_luts_dir =
        std::filesystem::path(QCoreApplication::applicationDirPath().toStdWString()) / "LUTs";
    const auto src_luts_dir = std::filesystem::path(CONFIG_PATH) / "LUTs";
    const auto luts_dir = std::filesystem::is_directory(app_luts_dir) ? app_luts_dir : src_luts_dir;
    const auto lut_files = ListCubeLutsInDir(luts_dir);

    lut_paths_.push_back("");  // index 0 => None
    lut_names_.push_back("None");
    for (const auto& p : lut_files) {
      lut_paths_.push_back(p.generic_string());
      lut_names_.push_back(QString::fromStdString(p.filename().string()));
    }

    auto addSection = [&](const QString& title, const QString& subtitle) {
      auto* frame = new QFrame(controls_);
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
      controls_layout->addWidget(frame, 0);
    };

    controls_layout->addStretch();

    int default_lut_index = 0;
    for (int i = 1; i < static_cast<int>(lut_paths_.size()); ++i) {
      if (std::filesystem::path(lut_paths_[i]).filename() == "5207.cube") {
        default_lut_index = i;
        break;
      }
    }

    // If the pipeline already has operator params (loaded from PipelineService/storage),
    // initialize UI state from those params rather than overwriting them.
    const bool loaded_state_from_pipeline = LoadStateFromPipelineIfPresent();
    if (!loaded_state_from_pipeline) {
      // Demo-friendly default: apply a LUT only for brand-new pipelines with no saved params.
      state_.lut_path_ = lut_paths_[default_lut_index];
      UpdateAllCdlWheelDerivedColors(state_);
    }
    committed_state_ = state_;

    // Seed a working version from the latest committed one (if any).
    working_version_ =
        controllers::SeedWorkingVersionFromLatest(element_id_, history_guard_);
    working_version_.SetBasePipelineExecutor(pipeline_guard_->pipeline_);

    int initial_lut_index = 0;
    if (!state_.lut_path_.empty()) {
      auto it = std::find(lut_paths_.begin(), lut_paths_.end(), state_.lut_path_);
      if (it != lut_paths_.end()) {
        initial_lut_index = static_cast<int>(std::distance(lut_paths_.begin(), it));
      } else {
        // Keep UI consistent even if LUT path comes from an external/custom location.
        lut_paths_.push_back(state_.lut_path_);
        lut_names_.push_back(
            QString::fromStdString(std::filesystem::path(state_.lut_path_).filename().string()));
        initial_lut_index = static_cast<int>(lut_paths_.size() - 1);
      }
    }

    auto addComboBox = [&](const QString& name, const QStringList& items, int initial_index,
                           auto&& onChange) {
      auto* label = new QLabel(name, controls_);
      label->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 14px;"
          "  font-weight: 400;"
          "}");

      auto* combo = new QComboBox(controls_);
      combo->addItems(items);
      combo->setCurrentIndex(initial_index);
      combo->setMinimumWidth(240);
      combo->setFixedHeight(32);
      combo->setStyleSheet(
          "QComboBox {"
          "  background: #1A1A1A;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 4px 8px;"
          "}"
          "QComboBox::drop-down {"
          "  border: 0px;"
          "  width: 24px;"
          "}"
          "QComboBox QAbstractItemView {"
          "  background: #1A1A1A;"
          "  border: none;"
          "  selection-background-color: #FCC704;"
          "  selection-color: #121212;"
          "}"
          "QComboBox QAbstractItemView::item:hover {"
          "  background: #202020;"
          "  color: #E6E6E6;"
          "}"
          "QComboBox QAbstractItemView::item:selected {"
          "  background: #FCC704;"
          "  color: #121212;"
          "}");

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
      rowLayout->addWidget(label, /*stretch*/ 1);
      rowLayout->addWidget(combo);

      controls_layout->insertWidget(controls_layout->count() - 1, row);
      return combo;
    };

    auto addSlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                         auto&& onRelease, auto&& formatter) {
      auto* info = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), controls_);
      info->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 14px;"
          "  font-weight: 400;"
          "}");

      auto* slider = new QSlider(Qt::Horizontal, controls_);
      slider->setRange(min, max);
      slider->setValue(value);
      slider->setSingleStep(1);
      slider->setPageStep(std::max(1, (max - min) / 20));
      slider->setMinimumWidth(240);
      slider->setFixedHeight(32);

      QObject::connect(slider, &QSlider::valueChanged, controls_,
                       [this, info, name, formatter,
                        onChange = std::forward<decltype(onChange)>(onChange)](int v) {
                         info->setText(QString("%1: %2").arg(name).arg(formatter(v)));
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

      auto* row       = new QWidget(controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(info, /*stretch*/ 1);
      rowLayout->addWidget(slider);

      controls_layout->insertWidget(controls_layout->count() - 1, row);
      return slider;
    };

    lut_combo_ = addComboBox("LUT", lut_names_, initial_lut_index, [&](int idx) {
      if (idx < 0 || idx >= static_cast<int>(lut_paths_.size())) {
        return;
      }
      state_.lut_path_ = lut_paths_[idx];
      CommitAdjustment(AdjustmentField::Lut);
    });

    addSection("Tone", "Primary tonal shaping controls.");

    exposure_slider_ = addSlider(
        "Exposure", -1000, 1000, static_cast<int>(std::lround(state_.exposure_ * 100.0f)),
        [&](int v) {
          state_.exposure_ = static_cast<float>(v) / 100.0f;
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Exposure); },
        [](int v) { return QString::number(v / 100.0, 'f', 2); });

    contrast_slider_ = addSlider(
        "Contrast", -100, 100, static_cast<int>(std::lround(state_.contrast_)),
        [&](int v) {
          state_.contrast_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Contrast); },
        [](int v) { return QString::number(v, 'f', 2); });

    highlights_slider_ = addSlider(
        "Highlights", -100, 100, static_cast<int>(std::lround(state_.highlights_)),
        [&](int v) {
          state_.highlights_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Highlights); },
        [](int v) { return QString::number(v, 'f', 2); });

    shadows_slider_ = addSlider(
        "Shadows", -100, 100, static_cast<int>(std::lround(state_.shadows_)),
        [&](int v) {
          state_.shadows_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Shadows); },
        [](int v) { return QString::number(v, 'f', 2); });

    whites_slider_ = addSlider(
        "Whites", -100, 100, static_cast<int>(std::lround(state_.whites_)),
        [&](int v) {
          state_.whites_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Whites); },
        [](int v) { return QString::number(v, 'f', 2); });

    blacks_slider_ = addSlider(
        "Blacks", -100, 100, static_cast<int>(std::lround(state_.blacks_)),
        [&](int v) {
          state_.blacks_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Blacks); },
        [](int v) { return QString::number(v, 'f', 2); });

    addSection("Curve", "Smooth tone curve mapped from input [0, 1] to output [0, 1].");
    {
      auto* frame  = new QFrame(controls_);
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(0, 0, 0, 0);
      layout->setSpacing(8);

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

      auto* actions_row        = new QWidget(frame);
      auto* actions_row_layout = new QHBoxLayout(actions_row);
      actions_row_layout->setContentsMargins(0, 0, 0, 0);
      actions_row_layout->setSpacing(8);

      auto* curve_hint =
          new QLabel("Left click/drag to shape. Right click a point to remove.", actions_row);
      curve_hint->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 11px;"
          "}");
      curve_hint->setWordWrap(true);

      auto* reset_curve_btn = new QPushButton("Reset Curve", actions_row);
      reset_curve_btn->setFixedHeight(28);
      reset_curve_btn->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 4px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #FCC704;"
          "}");
      QObject::connect(reset_curve_btn, &QPushButton::clicked, this, [this]() {
        if (!curve_widget_) {
          return;
        }
        state_.curve_points_ = DefaultCurveControlPoints();
        curve_widget_->SetControlPoints(state_.curve_points_);
        RequestRender();
        CommitAdjustment(AdjustmentField::Curve);
      });

      actions_row_layout->addWidget(curve_hint, 1);
      actions_row_layout->addWidget(reset_curve_btn, 0);

      layout->addWidget(curve_widget_, 1);
      layout->addWidget(actions_row, 0);

      controls_layout->insertWidget(controls_layout->count() - 1, frame);
    }

    addSection("Color", "Color balance and saturation.");

    saturation_slider_ = addSlider(
        "Saturation", -100, 100, static_cast<int>(std::lround(state_.saturation_)),
        [&](int v) {
          state_.saturation_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Saturation); },
        [](int v) { return QString::number(v, 'f', 2); });

    color_temp_mode_combo_ = addComboBox(
        "White Balance", {"As Shot", "Custom"},
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
          SyncControlsFromState();
          RequestRender();
          CommitAdjustment(AdjustmentField::ColorTemp);
        });

    color_temp_cct_slider_ = addSlider(
        "Color Temp", kColorTempSliderUiMin, kColorTempSliderUiMax,
        ColorTempCctToSliderPos(DisplayedColorTempCct(state_)),
        [&](int v) {
          PromoteColorTempToCustomForEditing();
          state_.color_temp_custom_cct_ = ColorTempSliderPosToCct(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::ColorTemp); },
        [](int v) { return QString("%1 K").arg(static_cast<int>(std::lround(ColorTempSliderPosToCct(v)))); });
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
        "Color Tint", kColorTempTintMin, kColorTempTintMax,
        static_cast<int>(std::lround(DisplayedColorTempTint(state_))),
        [&](int v) {
          PromoteColorTempToCustomForEditing();
          state_.color_temp_custom_tint_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::ColorTemp); },
        [](int v) { return QString::number(v, 'f', 0); });
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
        new QLabel("Color temperature/tint is unavailable for this image.", controls_);
    color_temp_unsupported_label_->setWordWrap(true);
    color_temp_unsupported_label_->setStyleSheet(
        "QLabel {"
        "  color: #FFB454;"
        "  background: rgba(255, 180, 84, 0.12);"
        "  border: 1px solid rgba(255, 180, 84, 0.35);"
        "  border-radius: 8px;"
        "  padding: 6px 8px;"
        "  font-size: 12px;"
        "}");
    controls_layout->insertWidget(controls_layout->count() - 1, color_temp_unsupported_label_);
    color_temp_unsupported_label_->setVisible(!state_.color_temp_supported_);

    {
      auto* frame = new QFrame(controls_);
      frame->setObjectName("EditorSection");
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(12, 10, 12, 10);
      layout->setSpacing(8);

      hls_target_label_ = new QLabel(frame);
      hls_target_label_->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 13px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(hls_target_label_, 0);

      auto* swatch_row        = new QWidget(frame);
      auto* swatch_row_layout = new QHBoxLayout(swatch_row);
      swatch_row_layout->setContentsMargins(0, 0, 0, 0);
      swatch_row_layout->setSpacing(6);

      hls_candidate_buttons_.clear();
      hls_candidate_buttons_.reserve(kHlsCandidateHues.size());
      for (int i = 0; i < static_cast<int>(kHlsCandidateHues.size()); ++i) {
        auto* btn = new QPushButton(swatch_row);
        btn->setFixedSize(22, 22);
        btn->setCursor(Qt::PointingHandCursor);
        btn->setToolTip(
            QString("Hue %1 deg").arg(kHlsCandidateHues[static_cast<size_t>(i)], 0, 'f', 0));
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

      controls_layout->insertWidget(controls_layout->count() - 1, frame);
      RefreshHlsTargetUi();
    }

    hls_hue_adjust_slider_ = addSlider(
        "Hue Shift", -15, 15, static_cast<int>(std::lround(state_.hls_hue_adjust_)),
        [&](int v) {
          state_.hls_hue_adjust_ =
              std::clamp(static_cast<float>(v), -kHlsMaxHueShiftDegrees, kHlsMaxHueShiftDegrees);
          SaveActiveHlsProfile(state_);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Hls); },
        [](int v) { return QString("%1 deg").arg(v); });

    hls_lightness_adjust_slider_ = addSlider(
        "Lightness", -100, 100, static_cast<int>(std::lround(state_.hls_lightness_adjust_)),
        [&](int v) {
          state_.hls_lightness_adjust_ =
              std::clamp(static_cast<float>(v), kHlsAdjUiMin, kHlsAdjUiMax);
          SaveActiveHlsProfile(state_);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Hls); },
        [](int v) { return QString::number(v, 'f', 0); });

    hls_saturation_adjust_slider_ = addSlider(
        "HSL Saturation", -100, 100, static_cast<int>(std::lround(state_.hls_saturation_adjust_)),
        [&](int v) {
          state_.hls_saturation_adjust_ =
              std::clamp(static_cast<float>(v), kHlsAdjUiMin, kHlsAdjUiMax);
          SaveActiveHlsProfile(state_);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Hls); },
        [](int v) { return QString::number(v, 'f', 0); });

    hls_hue_range_slider_ = addSlider(
        "Hue Range", 1, 180, static_cast<int>(std::lround(state_.hls_hue_range_)),
        [&](int v) {
          state_.hls_hue_range_ = static_cast<float>(v);
          SaveActiveHlsProfile(state_);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Hls); },
        [](int v) { return QString("%1 deg").arg(v); });

    {
      auto* wheel_frame = new QFrame(controls_);
      wheel_frame->setObjectName("EditorSection");
      wheel_frame->setMaximumWidth(480);
      wheel_frame->setMaximumHeight(440);

      auto* wheel_layout = new QVBoxLayout(wheel_frame);
      wheel_layout->setContentsMargins(10, 8, 10, 8);
      wheel_layout->setSpacing(2);

      auto* wheel_title = new QLabel("CDL Wheels", wheel_frame);
      wheel_title->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 12px;"
          "  font-weight: 600;"
          "}");
      wheel_layout->addWidget(wheel_title, 0, Qt::AlignHCenter);

      auto makeWheelUnit = [&](const QString& title, CdlWheelState& wheel_state, bool add_unity,
                               bool invert_delta, CdlTrackballDiscWidget*& disc_widget,
                               QLabel*& offset_label, QSlider*& slider_widget) -> QWidget* {
        auto* unit = new QWidget(wheel_frame);
        unit->setFixedWidth(180);
        auto* unit_layout = new QVBoxLayout(unit);
        unit_layout->setContentsMargins(0, 0, 0, 0);
        unit_layout->setSpacing(2);

        auto* title_label = new QLabel(title, unit);
        title_label->setStyleSheet(
            "QLabel {"
            "  color: #CFCFCF;"
            "  font-size: 10px;"
            "  font-weight: 500;"
            "}");
        unit_layout->addWidget(title_label, 0, Qt::AlignHCenter);

        disc_widget = new CdlTrackballDiscWidget(unit);
        disc_widget->setFixedSize(120, 120);
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
        offset_label->setStyleSheet(
            "QLabel {"
            "  color: #A9A9A9;"
            "  font-size: 9px;"
            "}");
        offset_label->setAlignment(Qt::AlignHCenter);
        unit_layout->addWidget(offset_label, 0);

        slider_widget = new QSlider(Qt::Horizontal, unit);
        slider_widget->setRange(kCdlWheelSliderUiMin, kCdlWheelSliderUiMax);
        const float sign = invert_delta ? -1.0f : 1.0f;
        slider_widget->setValue(CdlMasterToSliderUi(wheel_state.master_offset_ * sign));
        slider_widget->setSingleStep(1);
        slider_widget->setPageStep(100);
        slider_widget->setFixedHeight(14);
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
        unit_layout->addWidget(slider_widget, 0);

        return unit;
      };

      auto* gamma_unit = makeWheelUnit("Gamma", state_.gamma_wheel_, true, true,
                                       gamma_disc_widget_, gamma_offset_label_,
                                       gamma_master_slider_);
      auto* lift_unit  = makeWheelUnit("Lift", state_.lift_wheel_, false, false,
                                       lift_disc_widget_, lift_offset_label_,
                                       lift_master_slider_);
      auto* gain_unit  = makeWheelUnit("Gain", state_.gain_wheel_, true, false,
                                       gain_disc_widget_, gain_offset_label_,
                                       gain_master_slider_);

      auto* top_row = new QWidget(wheel_frame);
      auto* top_row_layout = new QHBoxLayout(top_row);
      top_row_layout->setContentsMargins(0, 0, 0, 0);
      top_row_layout->setSpacing(0);
      top_row_layout->addStretch();
      top_row_layout->addWidget(gamma_unit, 0);
      top_row_layout->addStretch();
      wheel_layout->addWidget(top_row, 0);

      auto* bottom_row = new QWidget(wheel_frame);
      auto* bottom_row_layout = new QHBoxLayout(bottom_row);
      bottom_row_layout->setContentsMargins(0, 0, 0, 0);
      bottom_row_layout->setSpacing(12);
      bottom_row_layout->addStretch();
      bottom_row_layout->addWidget(lift_unit, 0);
      bottom_row_layout->addWidget(gain_unit, 0);
      bottom_row_layout->addStretch();
      wheel_layout->addWidget(bottom_row, 0);

      controls_layout->insertWidget(controls_layout->count() - 1, wheel_frame, 0, Qt::AlignHCenter);
      RefreshCdlOffsetLabels();
    }

    addSection("Detail", "Micro-contrast and sharpen controls.");

    sharpen_slider_ = addSlider(
        "Sharpen", -100, 100, static_cast<int>(std::lround(state_.sharpen_)),
        [&](int v) {
          state_.sharpen_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Sharpen); },
        [](int v) { return QString::number(v, 'f', 2); });

    clarity_slider_ = addSlider(
        "Clarity", -100, 100, static_cast<int>(std::lround(state_.clarity_)),
        [&](int v) {
          state_.clarity_ = static_cast<float>(v);
          RequestRender();
        },
        [this]() { CommitAdjustment(AdjustmentField::Clarity); },
        [](int v) { return QString::number(v, 'f', 2); });

