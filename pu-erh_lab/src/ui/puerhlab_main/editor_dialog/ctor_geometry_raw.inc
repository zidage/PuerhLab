// ctor_geometry_raw.inc -- Constructor section for EditorDialog.
// This file is #included from the constructor body in dialog.cpp.
// It shares the constructor scope and has access to all local variables.

    auto addGeometrySection = [&](const QString& title, const QString& subtitle) {
      auto* frame = new QFrame(geometry_controls_);
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
      geometry_controls_layout->insertWidget(geometry_controls_layout->count() - 1, frame);
    };

    auto addGeometrySlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                                 auto&& formatter) {
      auto* info = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), geometry_controls_);
      info->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 14px;"
          "  font-weight: 400;"
          "}");

      auto* slider = new QSlider(Qt::Horizontal, geometry_controls_);
      slider->setRange(min, max);
      slider->setValue(value);
      slider->setSingleStep(1);
      slider->setPageStep(std::max(1, (max - min) / 20));
      slider->setMinimumWidth(240);
      slider->setFixedHeight(32);

      QObject::connect(slider, &QSlider::valueChanged, geometry_controls_,
                       [this, info, name, formatter,
                        onChange = std::forward<decltype(onChange)>(onChange)](int v) {
                         info->setText(QString("%1: %2").arg(name).arg(formatter(v)));
                         if (syncing_controls_) {
                           return;
                         }
                         onChange(v);
                       });

      auto* row       = new QWidget(geometry_controls_);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->addWidget(info, 1);
      rowLayout->addWidget(slider);
      geometry_controls_layout->insertWidget(geometry_controls_layout->count() - 1, row);
      return slider;
    };

    addGeometrySection("Geometry", "Rotate and crop workflow. Changes apply only when committed.");
    rotate_slider_ = addGeometrySlider(
        "Rotate", -18000, 18000, static_cast<int>(std::lround(state_.rotate_degrees_ * kRotationSliderScale)),
        [&](int v) {
          state_.rotate_degrees_ = static_cast<float>(v) / kRotationSliderScale;
          if (viewer_) {
            viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
          }
        },
        [](int v) { return QString("%1 deg").arg(static_cast<double>(v) / kRotationSliderScale, 0, 'f', 2); });
    // Double-click rotate slider to reset to 0.
    rotate_slider_->installEventFilter(this);

    geometry_crop_x_slider_ = addGeometrySlider(
        "Crop X", 0, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_x_ * kCropRectSliderScale)),
        [&](int v) {
          const auto clamped = ClampCropRect(static_cast<float>(v) / kCropRectSliderScale, state_.crop_y_,
                                             state_.crop_w_, state_.crop_h_);
          state_.crop_x_      = clamped[0];
          state_.crop_y_      = clamped[1];
          state_.crop_w_      = clamped[2];
          state_.crop_h_      = clamped[3];
          state_.crop_enabled_ = true;
          UpdateGeometryCropRectLabel();
          if (viewer_) {
            viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                                  state_.crop_h_);
          }
        },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    geometry_crop_y_slider_ = addGeometrySlider(
        "Crop Y", 0, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_y_ * kCropRectSliderScale)),
        [&](int v) {
          const auto clamped = ClampCropRect(state_.crop_x_, static_cast<float>(v) / kCropRectSliderScale,
                                             state_.crop_w_, state_.crop_h_);
          state_.crop_x_      = clamped[0];
          state_.crop_y_      = clamped[1];
          state_.crop_w_      = clamped[2];
          state_.crop_h_      = clamped[3];
          state_.crop_enabled_ = true;
          UpdateGeometryCropRectLabel();
          if (viewer_) {
            viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                                  state_.crop_h_);
          }
        },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    geometry_crop_w_slider_ = addGeometrySlider(
        "Crop W", 1, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_w_ * kCropRectSliderScale)),
        [&](int v) {
          const auto clamped = ClampCropRect(state_.crop_x_, state_.crop_y_,
                                             static_cast<float>(v) / kCropRectSliderScale, state_.crop_h_);
          state_.crop_x_      = clamped[0];
          state_.crop_y_      = clamped[1];
          state_.crop_w_      = clamped[2];
          state_.crop_h_      = clamped[3];
          state_.crop_enabled_ = true;
          UpdateGeometryCropRectLabel();
          if (viewer_) {
            viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                                  state_.crop_h_);
          }
        },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    geometry_crop_h_slider_ = addGeometrySlider(
        "Crop H", 1, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_h_ * kCropRectSliderScale)),
        [&](int v) {
          const auto clamped = ClampCropRect(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                             static_cast<float>(v) / kCropRectSliderScale);
          state_.crop_x_      = clamped[0];
          state_.crop_y_      = clamped[1];
          state_.crop_w_      = clamped[2];
          state_.crop_h_      = clamped[3];
          state_.crop_enabled_ = true;
          UpdateGeometryCropRectLabel();
          if (viewer_) {
            viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                                  state_.crop_h_);
          }
        },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    {
      auto* frame = new QFrame(geometry_controls_);
      frame->setObjectName("EditorSection");
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(12, 10, 12, 10);
      layout->setSpacing(8);

      geometry_crop_rect_label_ = new QLabel(frame);
      geometry_crop_rect_label_->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 12px;"
          "}");
      layout->addWidget(geometry_crop_rect_label_, 0);

      auto* row       = new QWidget(frame);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(8);

      geometry_apply_btn_ = new QPushButton("Apply Crop", row);
      geometry_apply_btn_->setFixedHeight(30);
      geometry_apply_btn_->setCursor(Qt::PointingHandCursor);
      geometry_apply_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  font-weight: 600;"
          "}"
          "QPushButton:hover {"
          "  background: #F5C200;"
          "}");
      QObject::connect(geometry_apply_btn_, &QPushButton::clicked, this, [this]() {
        state_.crop_enabled_ = true;
        CommitAdjustment(AdjustmentField::CropRotate);
      });
      rowLayout->addWidget(geometry_apply_btn_, 1);

      geometry_reset_btn_ = new QPushButton("Reset", row);
      geometry_reset_btn_->setFixedHeight(30);
      geometry_reset_btn_->setCursor(Qt::PointingHandCursor);
      geometry_reset_btn_->setToolTip("Reset crop & rotation (Ctrl+R)");
      geometry_reset_btn_->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_R));
      geometry_reset_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #E6E6E6;"
          "  background: #3A3A3A;"
          "  border: none;"
          "  border-radius: 8px;"
          "  font-weight: 600;"
          "}"
          "QPushButton:hover {"
          "  background: #505050;"
          "}");
      QObject::connect(geometry_reset_btn_, &QPushButton::clicked, this, [this]() {
        ResetCropAndRotation();
      });
      rowLayout->addWidget(geometry_reset_btn_, 0);
      layout->addWidget(row, 0);

      auto* hint = new QLabel(
          "Geometry panel edits only the crop frame overlay. Image pixels update only after Apply Crop. "
          "Double click viewer to restore full crop frame. "
          "Double click rotate slider to reset angle. Ctrl+R to reset all geometry.",
          frame);
      hint->setWordWrap(true);
      hint->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 11px;"
          "}");
      layout->addWidget(hint, 0);

      geometry_controls_layout->insertWidget(geometry_controls_layout->count() - 1, frame);
    }

    if (viewer_) {
      QObject::connect(viewer_, &QtEditViewer::CropOverlayRectChanged, this,
                       [this](float x, float y, float w, float h, bool /*is_final*/) {
                         if (syncing_controls_) {
                           return;
                         }
                         const auto clamped = ClampCropRect(x, y, w, h);
                         state_.crop_x_     = clamped[0];
                         state_.crop_y_     = clamped[1];
                         state_.crop_w_     = clamped[2];
                         state_.crop_h_     = clamped[3];
                         state_.crop_enabled_ = true;
                         UpdateGeometryCropRectLabel();
                         const bool prev_sync = syncing_controls_;
                         syncing_controls_     = true;
                         if (geometry_crop_x_slider_) {
                           geometry_crop_x_slider_->setValue(
                               static_cast<int>(std::lround(state_.crop_x_ * kCropRectSliderScale)));
                         }
                         if (geometry_crop_y_slider_) {
                           geometry_crop_y_slider_->setValue(
                               static_cast<int>(std::lround(state_.crop_y_ * kCropRectSliderScale)));
                         }
                         if (geometry_crop_w_slider_) {
                           geometry_crop_w_slider_->setValue(
                               static_cast<int>(std::lround(state_.crop_w_ * kCropRectSliderScale)));
                         }
                         if (geometry_crop_h_slider_) {
                           geometry_crop_h_slider_->setValue(
                               static_cast<int>(std::lround(state_.crop_h_ * kCropRectSliderScale)));
                         }
                          syncing_controls_ = prev_sync;
                        });
      QObject::connect(viewer_, &QtEditViewer::CropOverlayRotationChanged, this,
                       [this](float angle_degrees, bool /*is_final*/) {
                         if (syncing_controls_) {
                           return;
                         }
                         state_.rotate_degrees_ = angle_degrees;
                         const bool prev_sync = syncing_controls_;
                         syncing_controls_     = true;
                         if (rotate_slider_) {
                           rotate_slider_->setValue(
                               static_cast<int>(std::lround(state_.rotate_degrees_ *
                                                            kRotationSliderScale)));
                         }
                         syncing_controls_ = prev_sync;
                       });
    }
    UpdateGeometryCropRectLabel();
    RefreshGeometryModeUi();

    lens_catalog_ = LoadLensCatalog();
    {
      auto* raw_layout = qobject_cast<QVBoxLayout*>(raw_controls_ ? raw_controls_->layout() : nullptr);
      if (raw_layout) {
        auto addRawSection = [&](const QString& title, const QString& subtitle) {
          auto* frame = new QFrame(raw_controls_);
          frame->setObjectName("EditorSection");
          auto* v = new QVBoxLayout(frame);
          v->setContentsMargins(12, 10, 12, 10);
          v->setSpacing(8);

          auto* t = new QLabel(title, frame);
          t->setObjectName("EditorSectionTitle");
          auto* s = new QLabel(subtitle, frame);
          s->setObjectName("EditorSectionSub");
          s->setWordWrap(true);
          v->addWidget(t, 0);
          v->addWidget(s, 0);
          raw_layout->insertWidget(raw_layout->count() - 1, frame);
          return v;
        };

        auto* decode_layout = addRawSection(
            "RAW Decode",
            "Configure RAW decode options. These settings are shared with thumbnail rendering.");

        raw_highlights_reconstruct_checkbox_ =
            new QCheckBox("Enable Highlight Reconstruction", raw_controls_);
        raw_highlights_reconstruct_checkbox_->setChecked(state_.raw_highlights_reconstruct_);
        raw_highlights_reconstruct_checkbox_->setStyleSheet(
            "QCheckBox {"
            "  color: #E6E6E6;"
            "  spacing: 8px;"
            "  font-size: 13px;"
            "}"
            "QCheckBox::indicator {"
            "  width: 16px;"
            "  height: 16px;"
            "}"
            "QCheckBox::indicator:unchecked {"
            "  background: #121212;"
            "  border: 1px solid #2A2A2A;"
            "  border-radius: 3px;"
            "}"
            "QCheckBox::indicator:checked {"
            "  background: #FCC704;"
            "  border: 1px solid #FCC704;"
            "  border-radius: 3px;"
            "}");
        QObject::connect(raw_highlights_reconstruct_checkbox_, &QCheckBox::toggled, this,
                         [this](bool checked) {
                           if (syncing_controls_) {
                             return;
                           }
                           state_.raw_highlights_reconstruct_ = checked;
                           RequestRender();
                           CommitAdjustment(AdjustmentField::RawDecode);
                         });
        decode_layout->addWidget(raw_highlights_reconstruct_checkbox_, 0);

        auto* lens_layout = addRawSection(
            "Lens Calibration",
            "Enable correction and optionally override lens metadata with catalog entries.");

        lens_calib_enabled_checkbox_ = new QCheckBox("Enable Lens Calibration", raw_controls_);
        lens_calib_enabled_checkbox_->setChecked(state_.lens_calib_enabled_);
        lens_calib_enabled_checkbox_->setStyleSheet(raw_highlights_reconstruct_checkbox_->styleSheet());
        QObject::connect(lens_calib_enabled_checkbox_, &QCheckBox::toggled, this,
                         [this](bool checked) {
                           if (syncing_controls_) {
                             return;
                           }
                           state_.lens_calib_enabled_ = checked;
                           RequestRender();
                           CommitAdjustment(AdjustmentField::LensCalib);
                         });
        lens_layout->addWidget(lens_calib_enabled_checkbox_, 0);

        auto* brand_row = new QWidget(raw_controls_);
        auto* brand_row_layout = new QHBoxLayout(brand_row);
        brand_row_layout->setContentsMargins(0, 0, 0, 0);
        brand_row_layout->setSpacing(8);
        auto* brand_label = new QLabel("Lens Brand", brand_row);
        brand_label->setStyleSheet(
            "QLabel {"
            "  color: #E6E6E6;"
            "  font-size: 9px;"
            "}");
        lens_brand_combo_ = new QComboBox(brand_row);
        lens_brand_combo_->setMinimumWidth(220);
        lens_brand_combo_->setStyleSheet(
            "QComboBox {"
            "  background: #1A1A1A;"
            "  border: none;"
            "  border-radius: 8px;"
            "  padding: 4px 8px;"
            "}"
            "QComboBox::drop-down { border: 0px; width: 24px; }"
            "QComboBox QAbstractItemView {"
            "  background: #1A1A1A;"
            "  border: none;"
            "  selection-background-color: #FCC704;"
            "  selection-color: #121212;"
            "}");
        QObject::connect(lens_brand_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                         this, [this](int) {
                           if (syncing_controls_ || !lens_brand_combo_) {
                             return;
                           }
                           state_.lens_override_make_ =
                               lens_brand_combo_->currentData().toString().toStdString();
                           if (state_.lens_override_make_.empty()) {
                             state_.lens_override_model_.clear();
                           }
                           RefreshLensModelComboFromState();
                           RequestRender();
                           CommitAdjustment(AdjustmentField::LensCalib);
                         });
        brand_row_layout->addWidget(brand_label, 0);
        brand_row_layout->addWidget(lens_brand_combo_, 1);
        lens_layout->addWidget(brand_row, 0);

        auto* model_row = new QWidget(raw_controls_);
        auto* model_row_layout = new QHBoxLayout(model_row);
        model_row_layout->setContentsMargins(0, 0, 0, 0);
        model_row_layout->setSpacing(8);
        auto* model_label = new QLabel("Lens Model", model_row);
        model_label->setStyleSheet(brand_label->styleSheet());
        lens_model_combo_ = new QComboBox(model_row);
        lens_model_combo_->setMinimumWidth(220);
        lens_model_combo_->setStyleSheet(lens_brand_combo_->styleSheet());
        QObject::connect(lens_model_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                         this, [this](int) {
                           if (syncing_controls_ || !lens_model_combo_ ||
                               state_.lens_override_make_.empty()) {
                             return;
                           }
                           state_.lens_override_model_ =
                               lens_model_combo_->currentData().toString().toStdString();
                           RequestRender();
                           CommitAdjustment(AdjustmentField::LensCalib);
                         });
        model_row_layout->addWidget(model_label, 0);
        model_row_layout->addWidget(lens_model_combo_, 1);
        lens_layout->addWidget(model_row, 0);

        lens_catalog_status_label_ = new QLabel(raw_controls_);
        lens_catalog_status_label_->setWordWrap(true);
        lens_catalog_status_label_->setStyleSheet(
            "QLabel {"
            "  color: #A3A3A3;"
            "  font-size: 11px;"
            "}");
        lens_layout->addWidget(lens_catalog_status_label_, 0);

        RefreshLensComboFromState();
      }
    }

    {
      auto* section = new QFrame(shared_versioning_root);
      section->setObjectName("EditorSection");
      auto* v = new QVBoxLayout(section);
      v->setContentsMargins(12, 10, 12, 10);
      v->setSpacing(2);
      auto* t = new QLabel("Versioning", section);
      t->setObjectName("EditorSectionTitle");
      auto* s = new QLabel("Commit and inspect edit history.", section);
      s->setObjectName("EditorSectionSub");
      s->setWordWrap(true);
      v->addWidget(t, 0);
      v->addWidget(s, 0);
      shared_versioning_layout->addWidget(section, 0);
    }
