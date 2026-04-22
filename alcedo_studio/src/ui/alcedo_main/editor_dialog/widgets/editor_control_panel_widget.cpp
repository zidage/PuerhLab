//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/editor_control_panel_widget.hpp"

namespace alcedo::ui {

EditorControlPanelWidget::EditorControlPanelWidget(QWidget* parent) : QWidget(parent) {}

auto EditorDialog::BuildControlPanelShell(const QString& panel_style) -> EditorControlPanelWidget* {
  auto* controls_panel = new EditorControlPanelWidget(this);
  controls_panel->setMinimumWidth(kControlsPanelMinWidth);
  controls_panel->setMaximumWidth(900);
  controls_panel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  controls_panel->setObjectName("EditorControlsPanel");
  controls_panel->setAttribute(Qt::WA_StyledBackground, true);
  controls_panel->setStyleSheet(panel_style);

  auto* controls_panel_layout = new QVBoxLayout(controls_panel);
  controls_panel_layout->setContentsMargins(12, 12, 12, 12);
  controls_panel_layout->setSpacing(8);

  const QString scroll_style = AppTheme::EditorScrollAreaStyle();

  tone_controls_scroll_ = new QScrollArea(controls_panel);
  tone_controls_scroll_->setFrameShape(QFrame::NoFrame);
  tone_controls_scroll_->setWidgetResizable(true);
  tone_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  tone_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  tone_controls_scroll_->setStyleSheet(scroll_style);

  tone_controls_ = new ToneControlPanelWidget(tone_controls_scroll_);
  controls_      = tone_controls_;
  tone_controls_->setMinimumWidth(0);
  tone_controls_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  controls_layout_ = new QVBoxLayout(tone_controls_);
  controls_layout_->setContentsMargins(10, 8, 10, 10);
  controls_layout_->setSpacing(8);
  tone_controls_scroll_->setWidget(tone_controls_);
  controls_scroll_ = tone_controls_scroll_;

  BuildLookControlPanel(controls_panel, scroll_style);

  drt_controls_scroll_ = new QScrollArea(controls_panel);
  drt_controls_scroll_->setFrameShape(QFrame::NoFrame);
  drt_controls_scroll_->setWidgetResizable(true);
  drt_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  drt_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  drt_controls_scroll_->setStyleSheet(scroll_style);

  drt_controls_ = new DisplayTransformPanelWidget(drt_controls_scroll_);
  drt_controls_->setMinimumWidth(0);
  drt_controls_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  drt_controls_layout_ = new QVBoxLayout(drt_controls_);
  drt_controls_layout_->setContentsMargins(14, 14, 14, 14);
  drt_controls_layout_->setSpacing(12);
  drt_controls_layout_->addStretch();
  drt_controls_scroll_->setWidget(drt_controls_);

  geometry_controls_scroll_ = new QScrollArea(controls_panel);
  geometry_controls_scroll_->setFrameShape(QFrame::NoFrame);
  geometry_controls_scroll_->setWidgetResizable(true);
  geometry_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  geometry_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  geometry_controls_scroll_->setStyleSheet(scroll_style);

  geometry_controls_ = new GeometryPanelWidget(geometry_controls_scroll_);
  geometry_controls_->setMinimumWidth(0);
  geometry_controls_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  geometry_controls_layout_ = new QVBoxLayout(geometry_controls_);
  geometry_controls_layout_->setContentsMargins(14, 14, 14, 14);
  geometry_controls_layout_->setSpacing(12);
  geometry_controls_layout_->addStretch();
  geometry_controls_scroll_->setWidget(geometry_controls_);

  raw_controls_scroll_ = new QScrollArea(controls_panel);
  raw_controls_scroll_->setFrameShape(QFrame::NoFrame);
  raw_controls_scroll_->setWidgetResizable(true);
  raw_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  raw_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  raw_controls_scroll_->setStyleSheet(scroll_style);

  raw_controls_ = new RawDecodePanelWidget(raw_controls_scroll_);
  raw_controls_->setMinimumWidth(0);
  raw_controls_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  raw_controls_layout_ = new QVBoxLayout(raw_controls_);
  raw_controls_layout_->setContentsMargins(14, 14, 14, 14);
  raw_controls_layout_->setSpacing(12);
  raw_controls_layout_->addStretch();
  raw_controls_scroll_->setWidget(raw_controls_);

  control_panels_stack_ = new QStackedWidget(controls_panel);
  control_panels_stack_->addWidget(tone_controls_scroll_);
  control_panels_stack_->addWidget(look_controls_scroll_);
  control_panels_stack_->addWidget(drt_controls_scroll_);
  control_panels_stack_->addWidget(geometry_controls_scroll_);
  control_panels_stack_->addWidget(raw_controls_scroll_);
  control_panels_stack_->setCurrentIndex(0);

  auto* panel_switch_row = new QWidget(controls_panel);
  panel_switch_row->setObjectName("EditorPanelSwitchRow");
  panel_switch_row->setAttribute(Qt::WA_StyledBackground, true);
  panel_switch_row->setStyleSheet(
      QStringLiteral("#EditorPanelSwitchRow {"
                     "  background: %1;"
                     "  border: none;"
                     "  border-radius: 10px;"
                     "}")
          .arg(AppTheme::Instance().bgBaseColor().name(QColor::HexArgb)));

  auto* panel_switch_layout = new QHBoxLayout(panel_switch_row);
  panel_switch_layout->setContentsMargins(0, 0, 0, 0);
  panel_switch_layout->setSpacing(0);

  tone_panel_btn_     = new QPushButton(panel_switch_row);
  look_panel_btn_     = new QPushButton(panel_switch_row);
  drt_panel_btn_      = new QPushButton(panel_switch_row);
  geometry_panel_btn_ = new QPushButton(panel_switch_row);
  raw_panel_btn_      = new QPushButton(panel_switch_row);

  ConfigurePanelToggleButton(tone_panel_btn_, Tr("Tone"),
                             QStringLiteral(":/panel_icons/adjustments.svg"));
  ConfigurePanelToggleButton(look_panel_btn_, Tr("Color"),
                             QStringLiteral(":/panel_icons/palette.svg"));
  ConfigurePanelToggleButton(drt_panel_btn_, Tr("Display Rendering Transform"),
                             QStringLiteral(":/panel_icons/color-filter.svg"));
  ConfigurePanelToggleButton(geometry_panel_btn_, Tr("Geometry"),
                             QStringLiteral(":/panel_icons/crop.svg"));
  ConfigurePanelToggleButton(raw_panel_btn_, Tr("RAW Decode"),
                             QStringLiteral(":/panel_icons/aperture.svg"));

  panel_switch_layout->addWidget(tone_panel_btn_, 1);
  panel_switch_layout->addWidget(look_panel_btn_, 1);
  panel_switch_layout->addWidget(drt_panel_btn_, 1);
  panel_switch_layout->addWidget(geometry_panel_btn_, 1);
  panel_switch_layout->addWidget(raw_panel_btn_, 1);

  QObject::connect(tone_panel_btn_, &QPushButton::clicked, this,
                   [this]() { SetActiveControlPanel(ControlPanelKind::Tone); });
  QObject::connect(look_panel_btn_, &QPushButton::clicked, this,
                   [this]() { SetActiveControlPanel(ControlPanelKind::Look); });
  QObject::connect(drt_panel_btn_, &QPushButton::clicked, this,
                   [this]() { SetActiveControlPanel(ControlPanelKind::DisplayRenderingTransform); });
  QObject::connect(geometry_panel_btn_, &QPushButton::clicked, this,
                   [this]() { SetActiveControlPanel(ControlPanelKind::Geometry); });
  QObject::connect(raw_panel_btn_, &QPushButton::clicked, this,
                   [this]() { SetActiveControlPanel(ControlPanelKind::RawDecode); });
  RefreshPanelSwitchUi();

  auto* scope_frame = new QFrame(controls_panel);
  scope_frame->setObjectName("EditorSection");
  scope_frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  auto* scope_layout = new QVBoxLayout(scope_frame);
  scope_layout->setContentsMargins(0, 0, 0, 0);
  scope_layout->setSpacing(0);
  scope_panel_ = new ScopePanel(scope_frame);
  scope_panel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  scope_layout->addWidget(scope_panel_, 0);

  controls_panel_layout->addWidget(scope_frame, 0);
  controls_panel_layout->addWidget(panel_switch_row, 0);
  controls_panel_layout->addWidget(control_panels_stack_, 1);

  return controls_panel;
}

}  // namespace alcedo::ui
