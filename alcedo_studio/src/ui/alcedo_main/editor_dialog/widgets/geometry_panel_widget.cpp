//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/widgets/geometry_panel_widget.hpp"

#include <QCheckBox>
#include <QFrame>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSizePolicy>
#include <QString>
#include <algorithm>
#include <utility>

#include "ui/alcedo_main/app_theme.hpp"
#include "ui/alcedo_main/editor_dialog/pipeline/geometry_pipeline_adapter.hpp"
#include "ui/alcedo_main/editor_dialog/session/editor_adjustment_session.hpp"
#include "ui/alcedo_main/i18n.hpp"

namespace alcedo::ui {
namespace {

constexpr int kSectionContentMarginH = 18;
constexpr int kSectionContentMarginV = 16;
constexpr int kSectionInnerSpacing   = 14;
constexpr int kRowInnerSpacing       = 8;
constexpr int kControlHeight         = 32;
constexpr int kSliderHeight          = 26;
constexpr double kCropAspectSpinMin  = 0.01;
constexpr double kCropAspectSpinMax  = 100.0;

constexpr char kLocalizedTextProperty[]      = "puerhlabI18nText";
constexpr char kLocalizedTextUpperProperty[] = "puerhlabI18nTextUpper";

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
  } else if (auto* checkbox = qobject_cast<QCheckBox*>(object)) {
    checkbox->setText(text);
  }
}

void SetLocalizedToolTip(QWidget* widget, const char* source) {
  if (!widget || source == nullptr) {
    return;
  }
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

}  // namespace

GeometryPanelWidget::GeometryPanelWidget(QWidget* parent) : AdjustmentPanelWidget(parent) {}

void GeometryPanelWidget::Configure(Dependencies deps, Callbacks callbacks) {
  deps_      = std::move(deps);
  callbacks_ = std::move(callbacks);
  PullGeometryStateFromDialog();
  PullCommittedGeometryStateFromDialog();
}

void GeometryPanelWidget::SetSyncing(bool syncing) { local_syncing_ = syncing; }

auto GeometryPanelWidget::IsSyncing() const -> bool {
  return local_syncing_ || (callbacks_.is_global_syncing && callbacks_.is_global_syncing());
}

void GeometryPanelWidget::RequestPipelineRender() {
  if (callbacks_.request_render) {
    callbacks_.request_render();
  }
}

void GeometryPanelWidget::ProjectGeometryStateToDialog() {
  if (!deps_.dialog_state) {
    return;
  }
  auto& s                     = *deps_.dialog_state;
  s.rotate_degrees_           = geometry_state_.rotate_degrees_;
  s.crop_enabled_             = geometry_state_.crop_enabled_;
  s.crop_x_                   = geometry_state_.crop_x_;
  s.crop_y_                   = geometry_state_.crop_y_;
  s.crop_w_                   = geometry_state_.crop_w_;
  s.crop_h_                   = geometry_state_.crop_h_;
  s.crop_expand_to_fit_       = geometry_state_.crop_expand_to_fit_;
  s.crop_aspect_preset_       = geometry_state_.crop_aspect_preset_;
  s.crop_aspect_width_        = geometry_state_.crop_aspect_width_;
  s.crop_aspect_height_       = geometry_state_.crop_aspect_height_;
}

void GeometryPanelWidget::PullGeometryStateFromDialog() {
  if (!deps_.dialog_state) {
    return;
  }
  const auto& s               = *deps_.dialog_state;
  geometry_state_.rotate_degrees_     = s.rotate_degrees_;
  geometry_state_.crop_enabled_       = s.crop_enabled_;
  geometry_state_.crop_x_             = s.crop_x_;
  geometry_state_.crop_y_             = s.crop_y_;
  geometry_state_.crop_w_             = s.crop_w_;
  geometry_state_.crop_h_             = s.crop_h_;
  geometry_state_.crop_expand_to_fit_ = s.crop_expand_to_fit_;
  geometry_state_.crop_aspect_preset_ = s.crop_aspect_preset_;
  geometry_state_.crop_aspect_width_  = s.crop_aspect_width_;
  geometry_state_.crop_aspect_height_ = s.crop_aspect_height_;
}

void GeometryPanelWidget::PullCommittedGeometryStateFromDialog() {
  if (!deps_.dialog_committed_state) {
    return;
  }
  const auto& s                         = *deps_.dialog_committed_state;
  committed_geometry_state_.rotate_degrees_     = s.rotate_degrees_;
  committed_geometry_state_.crop_enabled_       = s.crop_enabled_;
  committed_geometry_state_.crop_x_             = s.crop_x_;
  committed_geometry_state_.crop_y_             = s.crop_y_;
  committed_geometry_state_.crop_w_             = s.crop_w_;
  committed_geometry_state_.crop_h_             = s.crop_h_;
  committed_geometry_state_.crop_expand_to_fit_ = s.crop_expand_to_fit_;
  committed_geometry_state_.crop_aspect_preset_ = s.crop_aspect_preset_;
  committed_geometry_state_.crop_aspect_width_  = s.crop_aspect_width_;
  committed_geometry_state_.crop_aspect_height_ = s.crop_aspect_height_;
}

void GeometryPanelWidget::PreviewGeometryField(AdjustmentField field) {
  ProjectGeometryStateToDialog();
  RequestPipelineRender();
  if (!deps_.session) {
    return;
  }
  deps_.session->Preview(AdjustmentPreview{
      .field  = field,
      .params = GeometryPipelineAdapter::ParamsFor(field, geometry_state_),
      .policy = PreviewPolicy::FastViewport,
  });
}

void GeometryPanelWidget::CommitGeometryField(AdjustmentField field) {
  ProjectGeometryStateToDialog();
  if (!deps_.session) {
    PullCommittedGeometryStateFromDialog();
    return;
  }

  if (!GeometryPipelineAdapter::FieldChanged(field, geometry_state_, committed_geometry_state_)) {
    deps_.session->Commit(field);
    PullCommittedGeometryStateFromDialog();
    return;
  }

  deps_.session->Commit(AdjustmentCommit{
      .field      = field,
      .old_params = GeometryPipelineAdapter::ParamsFor(field, committed_geometry_state_),
      .new_params = GeometryPipelineAdapter::ParamsFor(field, geometry_state_),
  });
  PullCommittedGeometryStateFromDialog();
}

void GeometryPanelWidget::Build() {
  if (!deps_.panel_layout) {
    return;
  }

  auto* controls_header = NewLocalizedLabel("Geometry", this);
  controls_header->setObjectName("SectionTitle");
  controls_header->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
  AppTheme::MarkFontRole(controls_header, AppTheme::FontRole::UiHeadline);
  deps_.panel_layout->insertWidget(0, controls_header, 0);

  BuildCropAspectSection();
  BuildRotateSection();
  BuildCropOffsetSection();
  BuildApplyResetSection();

  UpdateGeometryCropRectLabel();
  RefreshGeometryModeUi();
  PushGeometryStateToOverlay();
}

namespace {

auto AddGeometrySection(QWidget* parent, QVBoxLayout& layout, const char* title_source)
    -> QVBoxLayout* {
  auto* frame = new QFrame(parent);
  frame->setObjectName("EditorSection");
  auto* v = new QVBoxLayout(frame);
  v->setContentsMargins(kSectionContentMarginH, kSectionContentMarginV, kSectionContentMarginH,
                        kSectionContentMarginV);
  v->setSpacing(kSectionInnerSpacing);

  auto* t = NewLocalizedLabel(title_source, frame, true);
  t->setObjectName("EditorSectionTitle");
  QFont title_font = t->font();
  title_font.setBold(true);
  title_font.setLetterSpacing(QFont::AbsoluteSpacing, 1.4);
  t->setFont(title_font);
  v->addWidget(t, 0);

  layout.insertWidget(layout.count() - 1, frame);
  return v;
}

}  // namespace

void GeometryPanelWidget::BuildCropAspectSection() {
  const auto& theme = AppTheme::Instance();

  auto addSlider = [&](QWidget* parent, QVBoxLayout* layout, const char* name_source, int min, int max,
                       int value, auto&& onChange, auto&& onReset, auto&& formatter) -> QSlider* {
    auto* row = new QWidget(parent);
    auto* row_v = new QVBoxLayout(row);
    row_v->setContentsMargins(0, 0, 0, 0);
    row_v->setSpacing(6);

    auto* header = new QWidget(row);
    auto* header_h = new QHBoxLayout(header);
    header_h->setContentsMargins(0, 0, 0, 0);
    header_h->setSpacing(kRowInnerSpacing);

    auto* name_label = NewLocalizedLabel(name_source, header);
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
                       if (IsSyncing()) {
                         return;
                       }
                       onChange(v);
                     });

    if (callbacks_.register_slider_reset) {
      callbacks_.register_slider_reset(
          slider, [this, onReset = std::forward<decltype(onReset)>(onReset)]() mutable {
            if (IsSyncing()) {
              return;
            }
            onReset();
          });
    }

    row_v->addWidget(header, 0);
    row_v->addWidget(slider, 0);
    layout->addWidget(row, 0);
    return slider;
  };

  auto [frame, v] = [this]() -> std::pair<QFrame*, QVBoxLayout*> {
    auto* f = new QFrame(this);
    f->setObjectName("EditorSection");
    auto* vl = new QVBoxLayout(f);
    vl->setContentsMargins(kSectionContentMarginH, kSectionContentMarginV, kSectionContentMarginH,
                           kSectionContentMarginV);
    vl->setSpacing(kSectionInnerSpacing);

    auto* t = NewLocalizedLabel("Crop & Aspect Ratio", f, true);
    t->setObjectName("EditorSectionTitle");
    QFont title_font = t->font();
    title_font.setBold(true);
    title_font.setLetterSpacing(QFont::AbsoluteSpacing, 1.4);
    t->setFont(title_font);
    vl->addWidget(t, 0);

    deps_.panel_layout->insertWidget(deps_.panel_layout->count() - 1, f);
    return {f, vl};
  }();

  auto* aspect_row = new QWidget(frame);
  auto* aspect_h   = new QHBoxLayout(aspect_row);
  aspect_h->setContentsMargins(0, 0, 0, 0);
  aspect_h->setSpacing(kRowInnerSpacing);

  auto* aspect_label = NewLocalizedLabel("Aspect", aspect_row);
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
             static_cast<int>(geometry_state_.crop_aspect_preset_))));
  QObject::connect(geometry_crop_aspect_preset_combo_,
                   QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
                     if (IsSyncing() || !geometry_crop_aspect_preset_combo_) {
                       return;
                     }
                     const auto preset = static_cast<geometry::CropAspectPreset>(
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
    const bool prev_sync   = local_syncing_;
    local_syncing_         = true;
    const int custom_index = geometry_crop_aspect_preset_combo_->findData(
        static_cast<int>(geometry::CropAspectPreset::Custom));
    geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, custom_index));
    local_syncing_ = prev_sync;
  };

  geometry_crop_aspect_width_spin_ = new QDoubleSpinBox(wh_row);
  configureRatioSpin(geometry_crop_aspect_width_spin_, QStringLiteral("W   "),
                     geometry_state_.crop_aspect_width_);
  QObject::connect(geometry_crop_aspect_width_spin_,
                   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                   [this, setPresetToCustom](double value) {
                     if (IsSyncing()) {
                       return;
                     }
                     geometry_state_.crop_aspect_preset_ = geometry::CropAspectPreset::Custom;
                     geometry_state_.crop_aspect_width_  = static_cast<float>(value);
                     setPresetToCustom();
                     ApplyAspectPresetToCurrentCrop();
                     RefreshGeometryModeUi();
                   });
  wh_h->addWidget(geometry_crop_aspect_width_spin_, 1);

  geometry_crop_aspect_height_spin_ = new QDoubleSpinBox(wh_row);
  configureRatioSpin(geometry_crop_aspect_height_spin_, QStringLiteral("H   "),
                     geometry_state_.crop_aspect_height_);
  QObject::connect(geometry_crop_aspect_height_spin_,
                   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                   [this, setPresetToCustom](double value) {
                     if (IsSyncing()) {
                       return;
                     }
                     geometry_state_.crop_aspect_preset_ = geometry::CropAspectPreset::Custom;
                     geometry_state_.crop_aspect_height_ = static_cast<float>(value);
                     setPresetToCustom();
                     ApplyAspectPresetToCurrentCrop();
                     RefreshGeometryModeUi();
                   });
  wh_h->addWidget(geometry_crop_aspect_height_spin_, 1);

  v->addWidget(wh_row, 0);
}

void GeometryPanelWidget::BuildRotateSection() {
  const auto& theme = AppTheme::Instance();

  auto addSlider = [&](QWidget* parent, QVBoxLayout* layout, const char* name_source, int min, int max,
                       int value, auto&& onChange, auto&& onReset, auto&& formatter) -> QSlider* {
    auto* row = new QWidget(parent);
    auto* row_v = new QVBoxLayout(row);
    row_v->setContentsMargins(0, 0, 0, 0);
    row_v->setSpacing(6);

    auto* header = new QWidget(row);
    auto* header_h = new QHBoxLayout(header);
    header_h->setContentsMargins(0, 0, 0, 0);
    header_h->setSpacing(kRowInnerSpacing);

    auto* name_label = NewLocalizedLabel(name_source, header);
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
                       if (IsSyncing()) {
                         return;
                       }
                       onChange(v);
                     });

    if (callbacks_.register_slider_reset) {
      callbacks_.register_slider_reset(
          slider, [this, onReset = std::forward<decltype(onReset)>(onReset)]() mutable {
            if (IsSyncing()) {
              return;
            }
            onReset();
          });
    }

    row_v->addWidget(header, 0);
    row_v->addWidget(slider, 0);
    layout->addWidget(row, 0);
    return slider;
  };

  auto* v = AddGeometrySection(this, *deps_.panel_layout, "Rotate & Flip");

  rotate_slider_ = addSlider(
      this, v, "Angle", -18000, 18000,
      static_cast<int>(std::lround(geometry_state_.rotate_degrees_ * geometry::kRotationSliderScale)),
      [&](int vv) {
        geometry_state_.rotate_degrees_ = static_cast<float>(vv) / geometry::kRotationSliderScale;
        if (callbacks_.set_crop_overlay_rotation) {
          callbacks_.set_crop_overlay_rotation(geometry_state_.rotate_degrees_);
        }
      },
      [this]() { ResetCropAndRotation(); },
      [](int vv) {
        return QString::fromUtf8("%1°")
            .arg(static_cast<double>(vv) / geometry::kRotationSliderScale, 0, 'f', 1);
      });

  auto* btn_row = new QWidget(this);
  auto* btn_h   = new QHBoxLayout(btn_row);
  btn_h->setContentsMargins(0, 0, 0, 0);
  btn_h->setSpacing(kRowInnerSpacing);

  auto rotateBy = [this](float delta) {
    if (!rotate_slider_) {
      return;
    }
    float a = geometry_state_.rotate_degrees_ + delta;
    while (a > 180.0f) a -= 360.0f;
    while (a < -180.0f) a += 360.0f;
    rotate_slider_->setValue(static_cast<int>(std::lround(a * geometry::kRotationSliderScale)));
  };

  auto makeToolButton = [&](const QString& glyph, const char* tip_source,
                            std::function<void()> onClick, bool enabled) {
    auto* b = new QPushButton(glyph, btn_row);
    b->setFixedHeight(36);
    b->setCursor(Qt::PointingHandCursor);
    b->setStyleSheet(AppTheme::EditorSecondaryButtonStyle());
    SetLocalizedToolTip(b, tip_source);
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

  auto* rotate_l = makeToolButton(QString::fromUtf8("\xE2\x86\xBA"), "Rotate 90° left",
                                  [rotateBy]() { rotateBy(-90.0f); }, true);
  auto* rotate_r = makeToolButton(QString::fromUtf8("\xE2\x86\xBB"), "Rotate 90° right",
                                  [rotateBy]() { rotateBy(90.0f); }, true);
  auto* flip = makeToolButton(QString::fromUtf8("\xE2\x87\x84"),
                              "Flip horizontal (coming soon)", {}, false);

  btn_h->addWidget(rotate_l, 1);
  btn_h->addWidget(rotate_r, 1);
  btn_h->addWidget(flip, 1);
  v->addWidget(btn_row, 0);
}

void GeometryPanelWidget::BuildCropOffsetSection() {
  const auto& theme = AppTheme::Instance();

  auto addSlider = [&](QWidget* parent, QVBoxLayout* layout, const char* name_source, int min, int max,
                       int value, auto&& onChange, auto&& onReset, auto&& formatter) -> QSlider* {
    auto* row = new QWidget(parent);
    auto* row_v = new QVBoxLayout(row);
    row_v->setContentsMargins(0, 0, 0, 0);
    row_v->setSpacing(6);

    auto* header = new QWidget(row);
    auto* header_h = new QHBoxLayout(header);
    header_h->setContentsMargins(0, 0, 0, 0);
    header_h->setSpacing(kRowInnerSpacing);

    auto* name_label = NewLocalizedLabel(name_source, header);
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
                       if (IsSyncing()) {
                         return;
                       }
                       onChange(v);
                     });

    if (callbacks_.register_slider_reset) {
      callbacks_.register_slider_reset(
          slider, [this, onReset = std::forward<decltype(onReset)>(onReset)]() mutable {
            if (IsSyncing()) {
              return;
            }
            onReset();
          });
    }

    row_v->addWidget(header, 0);
    row_v->addWidget(slider, 0);
    layout->addWidget(row, 0);
    return slider;
  };

  auto* v = AddGeometrySection(this, *deps_.panel_layout, "Crop Offset");

  auto formatUnit = [](int vv) {
    return QString::number(static_cast<double>(vv) / geometry::kCropRectSliderScale, 'f', 3);
  };

  geometry_crop_x_slider_ = addSlider(
      this, v, "X", 0, static_cast<int>(geometry::kCropRectSliderScale),
      static_cast<int>(std::lround(geometry_state_.crop_x_ * geometry::kCropRectSliderScale)),
      [&](int vv) {
        SetCropRectState(static_cast<float>(vv) / geometry::kCropRectSliderScale, geometry_state_.crop_y_,
                         geometry_state_.crop_w_, geometry_state_.crop_h_, false, true);
      },
      [this]() { ResetCropAndRotation(); }, formatUnit);

  geometry_crop_y_slider_ = addSlider(
      this, v, "Y", 0, static_cast<int>(geometry::kCropRectSliderScale),
      static_cast<int>(std::lround(geometry_state_.crop_y_ * geometry::kCropRectSliderScale)),
      [&](int vv) {
        SetCropRectState(geometry_state_.crop_x_, static_cast<float>(vv) / geometry::kCropRectSliderScale,
                         geometry_state_.crop_w_, geometry_state_.crop_h_, false, true);
      },
      [this]() { ResetCropAndRotation(); }, formatUnit);

  geometry_crop_w_slider_ = addSlider(
      this, v, "Width", 1, static_cast<int>(geometry::kCropRectSliderScale),
      static_cast<int>(std::lround(geometry_state_.crop_w_ * geometry::kCropRectSliderScale)),
      [&](int vv) {
        ResizeCropRectWithAspect(static_cast<float>(vv) / geometry::kCropRectSliderScale, true);
      },
      [this]() { ResetCropAndRotation(); }, formatUnit);

  geometry_crop_h_slider_ = addSlider(
      this, v, "Height", 1, static_cast<int>(geometry::kCropRectSliderScale),
      static_cast<int>(std::lround(geometry_state_.crop_h_ * geometry::kCropRectSliderScale)),
      [&](int vv) {
        ResizeCropRectWithAspect(static_cast<float>(vv) / geometry::kCropRectSliderScale, false);
      },
      [this]() { ResetCropAndRotation(); }, formatUnit);

  geometry_crop_rect_label_ = new QLabel(this);
  geometry_crop_rect_label_->setStyleSheet(AppTheme::EditorLabelStyle(theme.textMutedColor()));
  AppTheme::MarkFontRole(geometry_crop_rect_label_, AppTheme::FontRole::DataCaption);
  geometry_crop_rect_label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  v->addWidget(geometry_crop_rect_label_, 0);
}

void GeometryPanelWidget::BuildApplyResetSection() {
  auto* frame = new QFrame(this);
  frame->setObjectName("EditorSection");
  auto* v = new QVBoxLayout(frame);
  v->setContentsMargins(kSectionContentMarginH, kSectionContentMarginV, kSectionContentMarginH,
                        kSectionContentMarginV);
  v->setSpacing(kSectionInnerSpacing);

  auto* btn_row = new QWidget(frame);
  auto* btn_h   = new QHBoxLayout(btn_row);
  btn_h->setContentsMargins(0, 0, 0, 0);
  btn_h->setSpacing(kRowInnerSpacing);

  geometry_apply_btn_ = NewLocalizedButton("Apply Crop", btn_row);
  geometry_apply_btn_->setFixedHeight(36);
  geometry_apply_btn_->setCursor(Qt::PointingHandCursor);
  geometry_apply_btn_->setStyleSheet(AppTheme::EditorPrimaryButtonStyle());
  QObject::connect(geometry_apply_btn_, &QPushButton::clicked, this, [this]() {
    geometry_state_.crop_enabled_ = true;
    CommitGeometryField(AdjustmentField::CropRotate);
  });
  btn_h->addWidget(geometry_apply_btn_, 2);

  geometry_reset_btn_ = NewLocalizedButton("Reset", btn_row);
  geometry_reset_btn_->setFixedHeight(36);
  geometry_reset_btn_->setCursor(Qt::PointingHandCursor);
  geometry_reset_btn_->setStyleSheet(AppTheme::EditorSecondaryButtonStyle());
  QObject::connect(geometry_reset_btn_, &QPushButton::clicked, this,
                   [this]() { ResetCropAndRotation(); });
  btn_h->addWidget(geometry_reset_btn_, 1);

  auto* hint = NewLocalizedLabel(
      "Pixels update on Apply. Double click any slider or the viewer to reset. "
      "Ctrl+R resets all geometry.",
      frame);
  hint->setWordWrap(true);
  hint->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
  AppTheme::MarkFontRole(hint, AppTheme::FontRole::UiHint);
  v->addWidget(btn_row, 0);
  v->addWidget(hint, 0);

  deps_.panel_layout->insertWidget(deps_.panel_layout->count() - 1, frame);
}

void GeometryPanelWidget::UpdateGeometryCropRectLabel() {
  if (!geometry_crop_rect_label_) {
    return;
  }
  float source_aspect = 1.0f;
  if (callbacks_.source_aspect_ratio) {
    source_aspect = callbacks_.source_aspect_ratio();
  }
  int source_width  = 0;
  int source_height = 0;
  if (source_aspect > 0) {
    source_width  = 1000;
    source_height = static_cast<int>(std::lround(1000.0f / source_aspect));
  }

  const auto resolved = geometry::ResolveCropRect(
      geometry_state_.crop_x_, geometry_state_.crop_y_, geometry_state_.crop_w_,
      geometry_state_.crop_h_, source_aspect, geometry_state_.crop_aspect_preset_,
      geometry_state_.crop_aspect_width_, geometry_state_.crop_aspect_height_, source_width,
      source_height);

  QString text = Tr("Crop Rect: x=%1 y=%2 w=%3 h=%4")
                     .arg(geometry_state_.crop_x_, 0, 'f', 3)
                     .arg(geometry_state_.crop_y_, 0, 'f', 3)
                     .arg(geometry_state_.crop_w_, 0, 'f', 3)
                     .arg(geometry_state_.crop_h_, 0, 'f', 3);
  if (resolved.pixel_width_ > 0 && resolved.pixel_height_ > 0) {
    text += Tr(" | Output: %1x%2").arg(resolved.pixel_width_).arg(resolved.pixel_height_);
  }
  text += Tr(" | Aspect: %1:1").arg(resolved.aspect_ratio_, 0, 'f', 3);
  geometry_crop_rect_label_->setText(text);
}

auto GeometryPanelWidget::CurrentGeometrySourceAspect() const -> float {
  if (callbacks_.source_aspect_ratio) {
    return callbacks_.source_aspect_ratio();
  }
  return 1.0f;
}

auto GeometryPanelWidget::CurrentGeometryAspectRatio() const -> std::optional<float> {
  return geometry::AspectRatioFromSize(geometry_state_.crop_aspect_width_,
                                       geometry_state_.crop_aspect_height_);
}

void GeometryPanelWidget::SyncGeometryCropSlidersFromState() {
  const bool prev_sync = local_syncing_;
  local_syncing_       = true;
  if (geometry_crop_x_slider_) {
    geometry_crop_x_slider_->setValue(
        static_cast<int>(std::lround(geometry_state_.crop_x_ * geometry::kCropRectSliderScale)));
  }
  if (geometry_crop_y_slider_) {
    geometry_crop_y_slider_->setValue(
        static_cast<int>(std::lround(geometry_state_.crop_y_ * geometry::kCropRectSliderScale)));
  }
  if (geometry_crop_w_slider_) {
    geometry_crop_w_slider_->setValue(
        static_cast<int>(std::lround(geometry_state_.crop_w_ * geometry::kCropRectSliderScale)));
  }
  if (geometry_crop_h_slider_) {
    geometry_crop_h_slider_->setValue(
        static_cast<int>(std::lround(geometry_state_.crop_h_ * geometry::kCropRectSliderScale)));
  }
  local_syncing_ = prev_sync;
}

void GeometryPanelWidget::SyncCropAspectControlsFromState() {
  const bool prev_sync = local_syncing_;
  local_syncing_       = true;
  if (geometry_crop_aspect_preset_combo_) {
    const int preset_index = geometry_crop_aspect_preset_combo_->findData(
        static_cast<int>(geometry_state_.crop_aspect_preset_));
    geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, preset_index));
  }
  if (geometry_crop_aspect_width_spin_) {
    geometry_crop_aspect_width_spin_->setValue(geometry_state_.crop_aspect_width_);
  }
  if (geometry_crop_aspect_height_spin_) {
    geometry_crop_aspect_height_spin_->setValue(geometry_state_.crop_aspect_height_);
  }
  local_syncing_ = prev_sync;
}

void GeometryPanelWidget::PushGeometryStateToOverlay() {
  if (callbacks_.set_crop_overlay_aspect_lock) {
    const bool locked_aspect = geometry::HasLockedAspect(
        geometry_state_.crop_aspect_preset_, geometry_state_.crop_aspect_width_,
        geometry_state_.crop_aspect_height_);
    callbacks_.set_crop_overlay_aspect_lock(locked_aspect,
                                            CurrentGeometryAspectRatio().value_or(1.0f));
  }
  if (callbacks_.set_crop_overlay_rect) {
    callbacks_.set_crop_overlay_rect(geometry_state_.crop_x_, geometry_state_.crop_y_,
                                     geometry_state_.crop_w_, geometry_state_.crop_h_);
  }
  if (callbacks_.set_crop_overlay_rotation) {
    callbacks_.set_crop_overlay_rotation(geometry_state_.rotate_degrees_);
  }
}

void GeometryPanelWidget::SetCropRectState(float x, float y, float w, float h, bool sync_controls,
                                           bool sync_overlay) {
  const auto clamped         = geometry::ClampCropRect(x, y, w, h);
  geometry_state_.crop_x_    = clamped[0];
  geometry_state_.crop_y_    = clamped[1];
  geometry_state_.crop_w_    = clamped[2];
  geometry_state_.crop_h_    = clamped[3];
  geometry_state_.crop_enabled_ = true;
  if (sync_controls) {
    SyncGeometryCropSlidersFromState();
  }
  UpdateGeometryCropRectLabel();
  if (sync_overlay) {
    PushGeometryStateToOverlay();
  }
}

void GeometryPanelWidget::ApplyAspectPresetToCurrentCrop() {
  if (!geometry::HasLockedAspect(geometry_state_.crop_aspect_preset_,
                                 geometry_state_.crop_aspect_width_,
                                 geometry_state_.crop_aspect_height_)) {
    UpdateGeometryCropRectLabel();
    PushGeometryStateToOverlay();
    return;
  }

  const auto aspect_ratio = CurrentGeometryAspectRatio();
  if (!aspect_ratio.has_value()) {
    UpdateGeometryCropRectLabel();
    PushGeometryStateToOverlay();
    return;
  }

  const auto max_rect =
      geometry::MakeMaxAspectCropRect(CurrentGeometrySourceAspect(), *aspect_ratio);
  SetCropRectState(max_rect[0], max_rect[1], max_rect[2], max_rect[3], true, true);
}

void GeometryPanelWidget::ResizeCropRectWithAspect(float proposed_value, bool use_width_driver) {
  if (!geometry::HasLockedAspect(geometry_state_.crop_aspect_preset_,
                                 geometry_state_.crop_aspect_width_,
                                 geometry_state_.crop_aspect_height_)) {
    if (use_width_driver) {
      SetCropRectState(geometry_state_.crop_x_, geometry_state_.crop_y_, proposed_value,
                       geometry_state_.crop_h_);
    } else {
      SetCropRectState(geometry_state_.crop_x_, geometry_state_.crop_y_, geometry_state_.crop_w_,
                       proposed_value);
    }
    return;
  }

  const auto aspect_ratio = CurrentGeometryAspectRatio();
  if (!aspect_ratio.has_value()) {
    return;
  }

  const auto resized = geometry::ResizeAspectRectAroundCenter(
      geometry_state_.crop_x_, geometry_state_.crop_y_,
      use_width_driver ? proposed_value : geometry_state_.crop_w_,
      use_width_driver ? geometry_state_.crop_h_ : proposed_value, CurrentGeometrySourceAspect(),
      *aspect_ratio, use_width_driver);
  SetCropRectState(resized[0], resized[1], resized[2], resized[3], true, true);
}

void GeometryPanelWidget::SetCropAspectPresetState(geometry::CropAspectPreset preset) {
  geometry_state_.crop_aspect_preset_ = preset;
  if (const auto preset_ratio = geometry::CropAspectPresetRatio(preset); preset_ratio.has_value()) {
    geometry_state_.crop_aspect_width_  = preset_ratio->at(0);
    geometry_state_.crop_aspect_height_ = preset_ratio->at(1);
  } else if (!geometry::NormalizeCropAspect(geometry_state_.crop_aspect_width_,
                                            geometry_state_.crop_aspect_height_)
                  .has_value()) {
    geometry_state_.crop_aspect_width_  = 1.0f;
    geometry_state_.crop_aspect_height_ = 1.0f;
  }
  SyncCropAspectControlsFromState();
  ApplyAspectPresetToCurrentCrop();
}

void GeometryPanelWidget::ResetCropAndRotation() {
  geometry_state_.crop_x_             = 0.0f;
  geometry_state_.crop_y_             = 0.0f;
  geometry_state_.crop_w_             = 1.0f;
  geometry_state_.crop_h_             = 1.0f;
  geometry_state_.crop_enabled_       = true;
  geometry_state_.rotate_degrees_     = 0.0f;
  geometry_state_.crop_aspect_preset_ = geometry::CropAspectPreset::Free;
  geometry_state_.crop_aspect_width_  = 1.0f;
  geometry_state_.crop_aspect_height_ = 1.0f;

  ProjectGeometryStateToDialog();

  const bool prev_sync = local_syncing_;
  local_syncing_       = true;
  if (geometry_crop_x_slider_) {
    geometry_crop_x_slider_->setValue(0);
  }
  if (geometry_crop_y_slider_) {
    geometry_crop_y_slider_->setValue(0);
  }
  if (geometry_crop_w_slider_) {
    geometry_crop_w_slider_->setValue(static_cast<int>(geometry::kCropRectSliderScale));
  }
  if (geometry_crop_h_slider_) {
    geometry_crop_h_slider_->setValue(static_cast<int>(geometry::kCropRectSliderScale));
  }
  if (rotate_slider_) {
    rotate_slider_->setValue(0);
  }
  if (geometry_crop_aspect_preset_combo_) {
    const int free_index =
        geometry_crop_aspect_preset_combo_->findData(static_cast<int>(geometry::CropAspectPreset::Free));
    geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, free_index));
  }
  if (geometry_crop_aspect_width_spin_) {
    geometry_crop_aspect_width_spin_->setValue(1.0);
  }
  if (geometry_crop_aspect_height_spin_) {
    geometry_crop_aspect_height_spin_->setValue(1.0);
  }
  local_syncing_ = prev_sync;

  UpdateGeometryCropRectLabel();
  if (callbacks_.set_crop_overlay_aspect_lock) {
    callbacks_.set_crop_overlay_aspect_lock(false, 1.0f);
  }
  if (callbacks_.set_crop_overlay_rect) {
    callbacks_.set_crop_overlay_rect(0.0f, 0.0f, 1.0f, 1.0f);
  }
  if (callbacks_.set_crop_overlay_rotation) {
    callbacks_.set_crop_overlay_rotation(0.0f);
  }
}

void GeometryPanelWidget::SetCropRectFromViewer(float x, float y, float w, float h) {
  if (IsSyncing()) {
    return;
  }
  SetCropRectState(x, y, w, h, true, false);
}

void GeometryPanelWidget::SetRotationFromViewer(float degrees) {
  if (IsSyncing()) {
    return;
  }
  geometry_state_.rotate_degrees_ = degrees;
  const bool prev_sync            = local_syncing_;
  local_syncing_                  = true;
  if (rotate_slider_) {
    rotate_slider_->setValue(
        static_cast<int>(std::lround(geometry_state_.rotate_degrees_ * geometry::kRotationSliderScale)));
  }
  local_syncing_ = prev_sync;
}

void GeometryPanelWidget::SyncControlsFromDialogState() {
  if (!deps_.dialog_state) {
    return;
  }
  PullGeometryStateFromDialog();
  PullCommittedGeometryStateFromDialog();

  const bool prev_sync = local_syncing_;
  local_syncing_       = true;
  if (rotate_slider_) {
    rotate_slider_->setValue(
        static_cast<int>(std::lround(geometry_state_.rotate_degrees_ * geometry::kRotationSliderScale)));
  }
  SyncGeometryCropSlidersFromState();
  SyncCropAspectControlsFromState();
  UpdateGeometryCropRectLabel();
  PushGeometryStateToOverlay();
  local_syncing_ = prev_sync;
}

void GeometryPanelWidget::RetranslateUi() {
  if (geometry_apply_btn_) {
    SetLocalizedText(geometry_apply_btn_, "Apply Crop");
  }
  if (geometry_reset_btn_) {
    SetLocalizedText(geometry_reset_btn_, "Reset");
  }
  UpdateGeometryCropRectLabel();
  if (geometry_crop_aspect_preset_combo_) {
    const int  current_value = geometry_crop_aspect_preset_combo_->currentData().toInt();
    const bool prev_sync     = local_syncing_;
    local_syncing_           = true;
    geometry_crop_aspect_preset_combo_->clear();
    for (const auto& option : geometry::CropAspectPresetOptions()) {
      geometry_crop_aspect_preset_combo_->addItem(Tr(option.label_),
                                                  static_cast<int>(option.value_));
    }
    const int index = geometry_crop_aspect_preset_combo_->findData(current_value);
    geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, index));
    local_syncing_ = prev_sync;
  }
}

void GeometryPanelWidget::LoadFromPipeline() {
  if (deps_.session && deps_.session->LoadFromPipeline()) {
    SyncControlsFromDialogState();
  }
}

void GeometryPanelWidget::ReloadFromCommittedState() { SyncControlsFromDialogState(); }

void GeometryPanelWidget::RefreshGeometryModeUi() {
  if (geometry_crop_aspect_width_spin_) {
    geometry_crop_aspect_width_spin_->setEnabled(true);
  }
  if (geometry_crop_aspect_height_spin_) {
    geometry_crop_aspect_height_spin_->setEnabled(true);
  }
}

}  // namespace alcedo::ui
