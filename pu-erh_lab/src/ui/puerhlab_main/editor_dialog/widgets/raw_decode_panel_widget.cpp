//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/dialog_internal.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/raw_decode_panel_widget.hpp"

namespace puerhlab::ui {

RawDecodePanelWidget::RawDecodePanelWidget(QWidget* parent) : QWidget(parent) {}

void EditorDialog::BuildRawDecodePanel() {
  lens_catalog_ = LoadLensCatalog();

  auto* raw_layout = qobject_cast<QVBoxLayout*>(raw_controls_ ? raw_controls_->layout() : nullptr);
  if (!raw_layout) {
    return;
  }

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
      Tr("RAW Decode"),
      Tr("Configure RAW decode options. These settings are shared with thumbnail rendering."));

  raw_highlights_reconstruct_checkbox_ =
      new QCheckBox(Tr("Enable Highlight Reconstruction"), raw_controls_);
  raw_highlights_reconstruct_checkbox_->setChecked(state_.raw_highlights_reconstruct_);
  raw_highlights_reconstruct_checkbox_->setStyleSheet(AppTheme::EditorCheckBoxStyle());
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
      Tr("Lens Calibration"),
      Tr("Enable correction and optionally override lens metadata with catalog entries."));

  lens_calib_enabled_checkbox_ = new QCheckBox(Tr("Enable Lens Calibration"), raw_controls_);
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

  auto* brand_label = new QLabel(Tr("Lens Brand"), brand_row);
  brand_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
  AppTheme::MarkFontRole(brand_label, AppTheme::FontRole::UiCaption);

  lens_brand_combo_ = new QComboBox(brand_row);
  lens_brand_combo_->setMinimumWidth(0);
  lens_brand_combo_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  lens_brand_combo_->setStyleSheet(AppTheme::EditorComboBoxStyle());
  QObject::connect(lens_brand_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                   [this](int) {
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

  auto* model_label = new QLabel(Tr("Lens Model"), model_row);
  model_label->setStyleSheet(brand_label->styleSheet());
  AppTheme::MarkFontRole(model_label, AppTheme::FontRole::UiCaption);

  lens_model_combo_ = new QComboBox(model_row);
  lens_model_combo_->setMinimumWidth(0);
  lens_model_combo_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  lens_model_combo_->setStyleSheet(lens_brand_combo_->styleSheet());
  QObject::connect(lens_model_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                   [this](int) {
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
      AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
  AppTheme::MarkFontRole(lens_catalog_status_label_, AppTheme::FontRole::UiHint);
  lens_layout->addWidget(lens_catalog_status_label_, 0);

  RefreshLensComboFromState();
}

}  // namespace puerhlab::ui
