//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/dialog_internal.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/geometry_panel_widget.hpp"

namespace puerhlab::ui {

GeometryPanelWidget::GeometryPanelWidget(QWidget* parent) : QWidget(parent) {}

void EditorDialog::BuildGeometryRawPanels() {
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
      geometry_controls_layout_->insertWidget(geometry_controls_layout_->count() - 1, frame);
    };

    auto addGeometrySlider = [&](const QString& name, int min, int max, int value, auto&& onChange,
                                 auto&& onReset, auto&& formatter) {
      auto* info = new QLabel(QString("%1: %2").arg(name).arg(formatter(value)), geometry_controls_);
      info->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
      AppTheme::MarkFontRole(info, AppTheme::FontRole::DataBody);
      info->setWordWrap(true);
      info->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

      auto* slider = new QSlider(Qt::Horizontal, geometry_controls_);
      slider->setRange(min, max);
      slider->setValue(value);
      slider->setSingleStep(1);
      slider->setPageStep(std::max(1, (max - min) / 20));
      slider->setMinimumWidth(0);
      slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
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

      RegisterSliderReset(
          slider, [this, onReset = std::forward<decltype(onReset)>(onReset)]() mutable {
            if (syncing_controls_) {
              return;
            }
            onReset();
          });

      auto* row       = new QWidget(geometry_controls_);
      auto* rowLayout = new QVBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(4);
      rowLayout->addWidget(info, 0);
      rowLayout->addWidget(slider, 1);
      geometry_controls_layout_->insertWidget(geometry_controls_layout_->count() - 1, row);
      return slider;
    };

    addGeometrySection(Tr("Geometry"),
                       Tr("Rotate and crop workflow. Changes apply only when committed."));
    rotate_slider_ = addGeometrySlider(
        Tr("Rotate"), -18000, 18000,
        static_cast<int>(std::lround(state_.rotate_degrees_ * kRotationSliderScale)),
        [&](int v) {
          state_.rotate_degrees_ = static_cast<float>(v) / kRotationSliderScale;
          if (viewer_) {
            viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
          }
        },
        [this]() { ResetCropAndRotation(); },
        [](int v) { return QString("%1 deg").arg(static_cast<double>(v) / kRotationSliderScale, 0, 'f', 2); });

    {
      auto* frame = new QFrame(geometry_controls_);
      frame->setObjectName("EditorSection");
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(12, 10, 12, 10);
      layout->setSpacing(8);

      auto* preset_row = new QWidget(frame);
      auto* preset_row_layout = new QVBoxLayout(preset_row);
      preset_row_layout->setContentsMargins(0, 0, 0, 0);
      preset_row_layout->setSpacing(4);

      auto* preset_label = new QLabel(Tr("Aspect Preset"), preset_row);
      preset_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
      AppTheme::MarkFontRole(preset_label, AppTheme::FontRole::UiCaption);
      preset_label->setWordWrap(true);
      preset_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
      preset_row_layout->addWidget(preset_label, 0);

      geometry_crop_aspect_preset_combo_ = new QComboBox(preset_row);
      geometry_crop_aspect_preset_combo_->setStyleSheet(AppTheme::EditorComboBoxStyle());
      geometry_crop_aspect_preset_combo_->setSizeAdjustPolicy(
          QComboBox::AdjustToMinimumContentsLengthWithIcon);
      geometry_crop_aspect_preset_combo_->setMinimumContentsLength(1);
      geometry_crop_aspect_preset_combo_->setMinimumWidth(0);
      geometry_crop_aspect_preset_combo_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
      for (const auto& option : geometry::CropAspectPresetOptions()) {
        geometry_crop_aspect_preset_combo_->addItem(option.label_, static_cast<int>(option.value_));
      }
      geometry_crop_aspect_preset_combo_->setCurrentIndex(
          std::max(0, geometry_crop_aspect_preset_combo_->findData(
                            static_cast<int>(state_.crop_aspect_preset_))));
      QObject::connect(geometry_crop_aspect_preset_combo_,
                       QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
                         if (syncing_controls_ || !geometry_crop_aspect_preset_combo_) {
                           return;
                         }
                         const auto preset = static_cast<CropAspectPreset>(
                             geometry_crop_aspect_preset_combo_->currentData().toInt());
                         SetCropAspectPresetState(preset);
                         RefreshGeometryModeUi();
                       });
      preset_row_layout->addWidget(geometry_crop_aspect_preset_combo_, 0);
      layout->addWidget(preset_row, 0);

      auto* ratio_row = new QWidget(frame);
      auto* ratio_row_layout = new QHBoxLayout(ratio_row);
      ratio_row_layout->setContentsMargins(0, 0, 0, 0);
      ratio_row_layout->setSpacing(8);

      auto* width_label = new QLabel(Tr("Width"), ratio_row);
      width_label->setStyleSheet(preset_label->styleSheet());
      AppTheme::MarkFontRole(width_label, AppTheme::FontRole::UiCaption);
      width_label->setWordWrap(true);
      width_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
      ratio_row_layout->addWidget(width_label, 0);

      geometry_crop_aspect_width_spin_ = new QDoubleSpinBox(ratio_row);
      geometry_crop_aspect_width_spin_->setRange(kCropAspectSpinMin, kCropAspectSpinMax);
      geometry_crop_aspect_width_spin_->setDecimals(3);
      geometry_crop_aspect_width_spin_->setSingleStep(0.01);
      geometry_crop_aspect_width_spin_->setValue(state_.crop_aspect_width_);
      geometry_crop_aspect_width_spin_->setStyleSheet(AppTheme::EditorSpinBoxStyle());
      geometry_crop_aspect_width_spin_->setMinimumWidth(0);
      geometry_crop_aspect_width_spin_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
      QObject::connect(geometry_crop_aspect_width_spin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                       this, [this](double value) {
                         if (syncing_controls_) {
                           return;
                         }
                         state_.crop_aspect_preset_ = CropAspectPreset::Custom;
                         state_.crop_aspect_width_  = static_cast<float>(value);
                         if (geometry_crop_aspect_preset_combo_) {
                           const bool prev_sync = syncing_controls_;
                           syncing_controls_    = true;
                           const int custom_index = geometry_crop_aspect_preset_combo_->findData(
                               static_cast<int>(CropAspectPreset::Custom));
                           geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, custom_index));
                           syncing_controls_ = prev_sync;
                         }
                         ApplyAspectPresetToCurrentCrop();
                         RefreshGeometryModeUi();
                       });
      ratio_row_layout->addWidget(geometry_crop_aspect_width_spin_, 1);

      auto* height_label = new QLabel(Tr("Height"), ratio_row);
      height_label->setStyleSheet(width_label->styleSheet());
      AppTheme::MarkFontRole(height_label, AppTheme::FontRole::UiCaption);
      height_label->setWordWrap(true);
      height_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
      ratio_row_layout->addWidget(height_label, 0);

      geometry_crop_aspect_height_spin_ = new QDoubleSpinBox(ratio_row);
      geometry_crop_aspect_height_spin_->setRange(kCropAspectSpinMin, kCropAspectSpinMax);
      geometry_crop_aspect_height_spin_->setDecimals(3);
      geometry_crop_aspect_height_spin_->setSingleStep(0.01);
      geometry_crop_aspect_height_spin_->setValue(state_.crop_aspect_height_);
      geometry_crop_aspect_height_spin_->setStyleSheet(AppTheme::EditorSpinBoxStyle());
      geometry_crop_aspect_height_spin_->setMinimumWidth(0);
      geometry_crop_aspect_height_spin_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
      QObject::connect(geometry_crop_aspect_height_spin_,
                       QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
                         if (syncing_controls_) {
                           return;
                         }
                         state_.crop_aspect_preset_ = CropAspectPreset::Custom;
                         state_.crop_aspect_height_ = static_cast<float>(value);
                         if (geometry_crop_aspect_preset_combo_) {
                           const bool prev_sync = syncing_controls_;
                           syncing_controls_    = true;
                           const int custom_index = geometry_crop_aspect_preset_combo_->findData(
                               static_cast<int>(CropAspectPreset::Custom));
                           geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, custom_index));
                           syncing_controls_ = prev_sync;
                         }
                         ApplyAspectPresetToCurrentCrop();
                         RefreshGeometryModeUi();
                       });
      ratio_row_layout->addWidget(geometry_crop_aspect_height_spin_, 1);

      layout->addWidget(ratio_row, 0);

      geometry_controls_layout_->insertWidget(geometry_controls_layout_->count() - 1, frame);
    }

    geometry_crop_x_slider_ = addGeometrySlider(
        Tr("Crop X"), 0, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_x_ * kCropRectSliderScale)),
        [&](int v) {
          SetCropRectState(static_cast<float>(v) / kCropRectSliderScale, state_.crop_y_, state_.crop_w_,
                           state_.crop_h_, false, true);
        },
        [this]() { ResetCropAndRotation(); },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    geometry_crop_y_slider_ = addGeometrySlider(
        Tr("Crop Y"), 0, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_y_ * kCropRectSliderScale)),
        [&](int v) {
          SetCropRectState(state_.crop_x_, static_cast<float>(v) / kCropRectSliderScale, state_.crop_w_,
                           state_.crop_h_, false, true);
        },
        [this]() { ResetCropAndRotation(); },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    geometry_crop_w_slider_ = addGeometrySlider(
        Tr("Crop W"), 1, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_w_ * kCropRectSliderScale)),
        [&](int v) {
          ResizeCropRectWithAspect(static_cast<float>(v) / kCropRectSliderScale, true);
        },
        [this]() { ResetCropAndRotation(); },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    geometry_crop_h_slider_ = addGeometrySlider(
        Tr("Crop H"), 1, static_cast<int>(kCropRectSliderScale),
        static_cast<int>(std::lround(state_.crop_h_ * kCropRectSliderScale)),
        [&](int v) {
          ResizeCropRectWithAspect(static_cast<float>(v) / kCropRectSliderScale, false);
        },
        [this]() { ResetCropAndRotation(); },
        [](int v) { return QString::number(static_cast<double>(v) / kCropRectSliderScale, 'f', 3); });

    {
      auto* frame = new QFrame(geometry_controls_);
      frame->setObjectName("EditorSection");
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(12, 10, 12, 10);
      layout->setSpacing(8);

      geometry_crop_rect_label_ = new QLabel(frame);
      geometry_crop_rect_label_->setStyleSheet(
          AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
      AppTheme::MarkFontRole(geometry_crop_rect_label_, AppTheme::FontRole::DataBody);
      layout->addWidget(geometry_crop_rect_label_, 0);

      auto* row       = new QWidget(frame);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(8);

      geometry_apply_btn_ = new QPushButton(Tr("Apply Crop"), row);
      geometry_apply_btn_->setFixedHeight(30);
      geometry_apply_btn_->setCursor(Qt::PointingHandCursor);
      geometry_apply_btn_->setStyleSheet(AppTheme::EditorPrimaryButtonStyle());
      QObject::connect(geometry_apply_btn_, &QPushButton::clicked, this, [this]() {
        state_.crop_enabled_ = true;
        CommitAdjustment(AdjustmentField::CropRotate);
      });
      rowLayout->addWidget(geometry_apply_btn_, 1);

      geometry_reset_btn_ = new QPushButton(Tr("Reset"), row);
      geometry_reset_btn_->setFixedHeight(30);
      geometry_reset_btn_->setCursor(Qt::PointingHandCursor);
      geometry_reset_btn_->setStyleSheet(AppTheme::EditorSecondaryButtonStyle());
      QObject::connect(geometry_reset_btn_, &QPushButton::clicked, this, [this]() {
        ResetCropAndRotation();
      });
      rowLayout->addWidget(geometry_reset_btn_, 0);
      layout->addWidget(row, 0);

      auto* hint = new QLabel(
          Tr("Geometry panel edits only the crop frame overlay. Image pixels update only after Apply Crop. "
             "Aspect presets lock crop resizing until switched back to Free. "
             "Double click viewer to restore full crop frame. "
             "Double click any geometry slider to restore the full-frame crop and zero rotation. Ctrl+R to reset all geometry."),
          frame);
      hint->setWordWrap(true);
      hint->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
      AppTheme::MarkFontRole(hint, AppTheme::FontRole::UiHint);
      layout->addWidget(hint, 0);

      geometry_controls_layout_->insertWidget(geometry_controls_layout_->count() - 1, frame);
    }

    if (viewer_) {
      QObject::connect(viewer_, &QtEditViewer::CropOverlayRectChanged, this,
                       [this](float x, float y, float w, float h, bool /*is_final*/) {
                         if (syncing_controls_) {
                           return;
                         }
                         SetCropRectState(x, y, w, h, true, false);
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
    BuildRawDecodePanel();
}

}  // namespace puerhlab::ui
