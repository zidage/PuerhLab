//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/geometry_panel_widget.hpp"

namespace alcedo::ui {

GeometryPanelWidget::GeometryPanelWidget(QWidget* parent) : QWidget(parent) {}

namespace {

constexpr int kSectionContentMarginH = 18;
constexpr int kSectionContentMarginV = 16;
constexpr int kSectionInnerSpacing   = 14;
constexpr int kRowInnerSpacing       = 8;
constexpr int kControlHeight         = 32;
constexpr int kSliderHeight          = 26;

}  // namespace

void EditorDialog::BuildGeometryRawPanels() {
    const auto& theme = AppTheme::Instance();

    auto* controls_header = new QLabel(Tr("Geometry"), geometry_controls_);
    controls_header->setObjectName("SectionTitle");
    controls_header->setStyleSheet(AppTheme::EditorLabelStyle(theme.textColor()));
    AppTheme::MarkFontRole(controls_header, AppTheme::FontRole::UiHeadline);
    geometry_controls_layout_->insertWidget(0, controls_header, 0);

    // --- helpers -----------------------------------------------------------

    auto addSection = [&](const QString& title) -> std::pair<QFrame*, QVBoxLayout*> {
      auto* frame = new QFrame(geometry_controls_);
      frame->setObjectName("EditorSection");
      auto* v = new QVBoxLayout(frame);
      v->setContentsMargins(kSectionContentMarginH, kSectionContentMarginV,
                            kSectionContentMarginH, kSectionContentMarginV);
      v->setSpacing(kSectionInnerSpacing);

      auto* t = new QLabel(title.toUpper(), frame);
      t->setObjectName("EditorSectionTitle");
      AppTheme::MarkFontRole(t, AppTheme::FontRole::UiOverline);
      QFont title_font = t->font();
      title_font.setBold(true);
      title_font.setLetterSpacing(QFont::AbsoluteSpacing, 1.4);
      t->setFont(title_font);
      v->addWidget(t, 0);

      geometry_controls_layout_->insertWidget(geometry_controls_layout_->count() - 1, frame);
      return {frame, v};
    };

    auto addSlider = [&](QWidget* parent, QVBoxLayout* layout, const QString& name, int min, int max,
                         int value, auto&& onChange, auto&& onReset, auto&& formatter) -> QSlider* {
      auto* row = new QWidget(parent);
      auto* row_v = new QVBoxLayout(row);
      row_v->setContentsMargins(0, 0, 0, 0);
      row_v->setSpacing(6);

      auto* header = new QWidget(row);
      auto* header_h = new QHBoxLayout(header);
      header_h->setContentsMargins(0, 0, 0, 0);
      header_h->setSpacing(kRowInnerSpacing);

      auto* name_label = new QLabel(name, header);
      name_label->setStyleSheet(AppTheme::EditorLabelStyle(theme.textColor()));
      AppTheme::MarkFontRole(name_label, AppTheme::FontRole::UiBody);
      name_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

      auto* value_label = new QLabel(formatter(value), header);
      value_label->setStyleSheet(AppTheme::EditorLabelStyle(theme.accentColor()));
      AppTheme::MarkFontRole(value_label, AppTheme::FontRole::DataNumeric);
      value_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
      value_label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

      header_h->addWidget(name_label, 1);
      header_h->addWidget(value_label, 0);

      auto* slider = new QSlider(Qt::Horizontal, row);
      slider->setRange(min, max);
      slider->setValue(value);
      slider->setSingleStep(1);
      slider->setPageStep(std::max(1, (max - min) / 20));
      slider->setMinimumWidth(0);
      slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
      slider->setFixedHeight(kSliderHeight);

      QObject::connect(slider, &QSlider::valueChanged, parent,
                       [this, value_label, formatter,
                        onChange = std::forward<decltype(onChange)>(onChange)](int v) {
                         value_label->setText(formatter(v));
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

      row_v->addWidget(header, 0);
      row_v->addWidget(slider, 0);
      layout->addWidget(row, 0);
      return slider;
    };

    // --- SECTION 1: CROP & ASPECT RATIO ------------------------------------
    {
      auto [frame, v] = addSection(Tr("Crop & Aspect Ratio"));

      auto* aspect_row = new QWidget(frame);
      auto* aspect_h   = new QHBoxLayout(aspect_row);
      aspect_h->setContentsMargins(0, 0, 0, 0);
      aspect_h->setSpacing(kRowInnerSpacing);

      auto* aspect_label = new QLabel(Tr("Aspect"), aspect_row);
      aspect_label->setStyleSheet(AppTheme::EditorLabelStyle(theme.textColor()));
      AppTheme::MarkFontRole(aspect_label, AppTheme::FontRole::UiBody);
      aspect_label->setMinimumWidth(64);
      aspect_label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
      aspect_h->addWidget(aspect_label, 0);

      geometry_crop_aspect_preset_combo_ = new QComboBox(aspect_row);
      geometry_crop_aspect_preset_combo_->setStyleSheet(AppTheme::EditorComboBoxStyle());
      geometry_crop_aspect_preset_combo_->setSizeAdjustPolicy(
          QComboBox::AdjustToMinimumContentsLengthWithIcon);
      geometry_crop_aspect_preset_combo_->setMinimumContentsLength(1);
      geometry_crop_aspect_preset_combo_->setMinimumWidth(0);
      geometry_crop_aspect_preset_combo_->setFixedHeight(kControlHeight);
      geometry_crop_aspect_preset_combo_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
      for (const auto& option : geometry::CropAspectPresetOptions()) {
        geometry_crop_aspect_preset_combo_->addItem(Tr(option.label_),
                                                    static_cast<int>(option.value_));
      }
      geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(
          0, geometry_crop_aspect_preset_combo_->findData(
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
      aspect_h->addWidget(geometry_crop_aspect_preset_combo_, 1);
      v->addWidget(aspect_row, 0);

      auto* wh_row = new QWidget(frame);
      auto* wh_h   = new QHBoxLayout(wh_row);
      wh_h->setContentsMargins(0, 0, 0, 0);
      wh_h->setSpacing(kRowInnerSpacing);

      auto configureRatioSpin = [&](QDoubleSpinBox* spin, const QString& prefix, double value) {
        spin->setRange(kCropAspectSpinMin, kCropAspectSpinMax);
        spin->setDecimals(2);
        spin->setSingleStep(0.01);
        spin->setValue(value);
        spin->setPrefix(prefix);
        spin->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        spin->setStyleSheet(AppTheme::EditorSpinBoxStyle());
        spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
        spin->setFixedHeight(kControlHeight);
        spin->setMinimumWidth(0);
        spin->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        AppTheme::MarkFontRole(spin, AppTheme::FontRole::DataNumeric);
      };

      auto setPresetToCustom = [this]() {
        if (!geometry_crop_aspect_preset_combo_) {
          return;
        }
        const bool prev_sync    = syncing_controls_;
        syncing_controls_       = true;
        const int custom_index  = geometry_crop_aspect_preset_combo_->findData(
            static_cast<int>(CropAspectPreset::Custom));
        geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, custom_index));
        syncing_controls_ = prev_sync;
      };

      geometry_crop_aspect_width_spin_ = new QDoubleSpinBox(wh_row);
      configureRatioSpin(geometry_crop_aspect_width_spin_, QStringLiteral("W   "),
                         state_.crop_aspect_width_);
      QObject::connect(geometry_crop_aspect_width_spin_,
                       QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                       [this, setPresetToCustom](double value) {
                         if (syncing_controls_) {
                           return;
                         }
                         state_.crop_aspect_preset_ = CropAspectPreset::Custom;
                         state_.crop_aspect_width_  = static_cast<float>(value);
                         setPresetToCustom();
                         ApplyAspectPresetToCurrentCrop();
                         RefreshGeometryModeUi();
                       });
      wh_h->addWidget(geometry_crop_aspect_width_spin_, 1);

      geometry_crop_aspect_height_spin_ = new QDoubleSpinBox(wh_row);
      configureRatioSpin(geometry_crop_aspect_height_spin_, QStringLiteral("H   "),
                         state_.crop_aspect_height_);
      QObject::connect(geometry_crop_aspect_height_spin_,
                       QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                       [this, setPresetToCustom](double value) {
                         if (syncing_controls_) {
                           return;
                         }
                         state_.crop_aspect_preset_ = CropAspectPreset::Custom;
                         state_.crop_aspect_height_ = static_cast<float>(value);
                         setPresetToCustom();
                         ApplyAspectPresetToCurrentCrop();
                         RefreshGeometryModeUi();
                       });
      wh_h->addWidget(geometry_crop_aspect_height_spin_, 1);

      v->addWidget(wh_row, 0);
    }

    // --- SECTION 2: ROTATE & FLIP -----------------------------------------
    {
      auto [frame, v] = addSection(Tr("Rotate & Flip"));

      rotate_slider_ = addSlider(
          frame, v, Tr("Angle"), -18000, 18000,
          static_cast<int>(std::lround(state_.rotate_degrees_ * kRotationSliderScale)),
          [&](int vv) {
            state_.rotate_degrees_ = static_cast<float>(vv) / kRotationSliderScale;
            if (viewer_) {
              viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
            }
          },
          [this]() { ResetCropAndRotation(); },
          [](int vv) {
            return QString::fromUtf8("%1°")
                .arg(static_cast<double>(vv) / kRotationSliderScale, 0, 'f', 1);
          });

      auto* btn_row = new QWidget(frame);
      auto* btn_h   = new QHBoxLayout(btn_row);
      btn_h->setContentsMargins(0, 0, 0, 0);
      btn_h->setSpacing(kRowInnerSpacing);

      auto rotateBy = [this](float delta) {
        if (!rotate_slider_) {
          return;
        }
        float a = state_.rotate_degrees_ + delta;
        while (a > 180.0f) a -= 360.0f;
        while (a < -180.0f) a += 360.0f;
        rotate_slider_->setValue(static_cast<int>(std::lround(a * kRotationSliderScale)));
      };

      auto makeToolButton = [&](const QString& glyph, const QString& tip,
                                std::function<void()> onClick, bool enabled) {
        auto* b = new QPushButton(glyph, btn_row);
        b->setFixedHeight(36);
        b->setCursor(Qt::PointingHandCursor);
        b->setStyleSheet(AppTheme::EditorSecondaryButtonStyle());
        b->setToolTip(tip);
        b->setEnabled(enabled);
        AppTheme::MarkFontRole(b, AppTheme::FontRole::UiBodyStrong);
        QFont gf = b->font();
        gf.setPointSizeF(gf.pointSizeF() + 2.0);
        b->setFont(gf);
        if (enabled && onClick) {
          QObject::connect(b, &QPushButton::clicked, this, std::move(onClick));
        }
        return b;
      };

      auto* rotate_l = makeToolButton(QString::fromUtf8("\xE2\x86\xBA"),
                                      Tr("Rotate 90° left"), [rotateBy]() { rotateBy(-90.0f); }, true);
      auto* rotate_r = makeToolButton(QString::fromUtf8("\xE2\x86\xBB"),
                                      Tr("Rotate 90° right"), [rotateBy]() { rotateBy(90.0f); }, true);
      auto* flip     = makeToolButton(QString::fromUtf8("\xE2\x87\x84"),
                                      Tr("Flip horizontal (coming soon)"), {}, false);

      btn_h->addWidget(rotate_l, 1);
      btn_h->addWidget(rotate_r, 1);
      btn_h->addWidget(flip, 1);
      v->addWidget(btn_row, 0);
    }

    // --- SECTION 3: CROP OFFSET -------------------------------------------
    {
      auto [frame, v] = addSection(Tr("Crop Offset"));

      auto formatUnit = [](int vv) {
        return QString::number(static_cast<double>(vv) / kCropRectSliderScale, 'f', 3);
      };

      geometry_crop_x_slider_ = addSlider(
          frame, v, Tr("X"), 0, static_cast<int>(kCropRectSliderScale),
          static_cast<int>(std::lround(state_.crop_x_ * kCropRectSliderScale)),
          [&](int vv) {
            SetCropRectState(static_cast<float>(vv) / kCropRectSliderScale, state_.crop_y_,
                             state_.crop_w_, state_.crop_h_, false, true);
          },
          [this]() { ResetCropAndRotation(); }, formatUnit);

      geometry_crop_y_slider_ = addSlider(
          frame, v, Tr("Y"), 0, static_cast<int>(kCropRectSliderScale),
          static_cast<int>(std::lround(state_.crop_y_ * kCropRectSliderScale)),
          [&](int vv) {
            SetCropRectState(state_.crop_x_, static_cast<float>(vv) / kCropRectSliderScale,
                             state_.crop_w_, state_.crop_h_, false, true);
          },
          [this]() { ResetCropAndRotation(); }, formatUnit);

      geometry_crop_w_slider_ = addSlider(
          frame, v, Tr("Width"), 1, static_cast<int>(kCropRectSliderScale),
          static_cast<int>(std::lround(state_.crop_w_ * kCropRectSliderScale)),
          [&](int vv) {
            ResizeCropRectWithAspect(static_cast<float>(vv) / kCropRectSliderScale, true);
          },
          [this]() { ResetCropAndRotation(); }, formatUnit);

      geometry_crop_h_slider_ = addSlider(
          frame, v, Tr("Height"), 1, static_cast<int>(kCropRectSliderScale),
          static_cast<int>(std::lround(state_.crop_h_ * kCropRectSliderScale)),
          [&](int vv) {
            ResizeCropRectWithAspect(static_cast<float>(vv) / kCropRectSliderScale, false);
          },
          [this]() { ResetCropAndRotation(); }, formatUnit);

      geometry_crop_rect_label_ = new QLabel(frame);
      geometry_crop_rect_label_->setStyleSheet(
          AppTheme::EditorLabelStyle(theme.textMutedColor()));
      AppTheme::MarkFontRole(geometry_crop_rect_label_, AppTheme::FontRole::DataCaption);
      geometry_crop_rect_label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
      v->addWidget(geometry_crop_rect_label_, 0);
    }

    // --- SECTION 4: APPLY / RESET -----------------------------------------
    {
      auto* frame = new QFrame(geometry_controls_);
      frame->setObjectName("EditorSection");
      auto* v = new QVBoxLayout(frame);
      v->setContentsMargins(kSectionContentMarginH, kSectionContentMarginV,
                            kSectionContentMarginH, kSectionContentMarginV);
      v->setSpacing(kSectionInnerSpacing);

      auto* btn_row   = new QWidget(frame);
      auto* btn_h     = new QHBoxLayout(btn_row);
      btn_h->setContentsMargins(0, 0, 0, 0);
      btn_h->setSpacing(kRowInnerSpacing);

      geometry_apply_btn_ = new QPushButton(Tr("Apply Crop"), btn_row);
      geometry_apply_btn_->setFixedHeight(36);
      geometry_apply_btn_->setCursor(Qt::PointingHandCursor);
      geometry_apply_btn_->setStyleSheet(AppTheme::EditorPrimaryButtonStyle());
      QObject::connect(geometry_apply_btn_, &QPushButton::clicked, this, [this]() {
        state_.crop_enabled_ = true;
        CommitAdjustment(AdjustmentField::CropRotate);
      });
      btn_h->addWidget(geometry_apply_btn_, 2);

      geometry_reset_btn_ = new QPushButton(Tr("Reset"), btn_row);
      geometry_reset_btn_->setFixedHeight(36);
      geometry_reset_btn_->setCursor(Qt::PointingHandCursor);
      geometry_reset_btn_->setStyleSheet(AppTheme::EditorSecondaryButtonStyle());
      QObject::connect(geometry_reset_btn_, &QPushButton::clicked, this,
                       [this]() { ResetCropAndRotation(); });
      btn_h->addWidget(geometry_reset_btn_, 1);
      v->addWidget(btn_row, 0);

      auto* hint = new QLabel(
          Tr("Pixels update on Apply. Double click any slider or the viewer to reset. "
             "Ctrl+R resets all geometry."),
          frame);
      hint->setWordWrap(true);
      hint->setStyleSheet(AppTheme::EditorLabelStyle(theme.textMutedColor()));
      AppTheme::MarkFontRole(hint, AppTheme::FontRole::UiHint);
      v->addWidget(hint, 0);

      geometry_controls_layout_->insertWidget(geometry_controls_layout_->count() - 1, frame);
    }

    // --- viewer wiring -----------------------------------------------------
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
                         syncing_controls_    = true;
                         if (rotate_slider_) {
                           rotate_slider_->setValue(static_cast<int>(
                               std::lround(state_.rotate_degrees_ * kRotationSliderScale)));
                         }
                         syncing_controls_ = prev_sync;
                       });
    }
    UpdateGeometryCropRectLabel();
    RefreshGeometryModeUi();
    BuildRawDecodePanel();
}

}  // namespace alcedo::ui
