//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/dialog_internal.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/look_control_panel_widget.hpp"

namespace puerhlab::ui {

LookControlPanelWidget::LookControlPanelWidget(QWidget* parent) : QWidget(parent) {}

void EditorDialog::BuildLookControlPanel(EditorControlPanelWidget* controls_panel,
                                         const QString&             scroll_style) {
  look_controls_scroll_ = new QScrollArea(controls_panel);
  look_controls_scroll_->setFrameShape(QFrame::NoFrame);
  look_controls_scroll_->setWidgetResizable(true);
  look_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  look_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  look_controls_scroll_->setStyleSheet(scroll_style);

  look_controls_ = new LookControlPanelWidget(look_controls_scroll_);
  look_controls_->setMinimumWidth(0);
  look_controls_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  look_controls_layout_ = new QVBoxLayout(look_controls_);
  look_controls_layout_->setContentsMargins(14, 14, 14, 14);
  look_controls_layout_->setSpacing(12);

  lut_browser_widget_ = new LutBrowserWidget(look_controls_);
  look_controls_layout_->addWidget(lut_browser_widget_, 1);
  look_controls_scroll_->setWidget(look_controls_);
}

void EditorDialog::WireLookControlPanel() {
  if (!lut_browser_widget_) {
    return;
  }

  const auto lut_view_model = lut_controller_.Refresh(state_.lut_path_);
  lut_browser_widget_->SetDirectoryInfo(lut_view_model.directory_text_, lut_view_model.status_text_,
                                        lut_view_model.can_open_directory_);
  lut_browser_widget_->SetEntries(lut_view_model.entries_, lut_view_model.selected_path_);

  QObject::connect(lut_browser_widget_, &LutBrowserWidget::RefreshRequested, this,
                   [this]() { ForceRefreshLutBrowserUi(); });
  QObject::connect(lut_browser_widget_, &LutBrowserWidget::OpenFolderRequested, this,
                   [this]() { OpenLutFolder(); });
  QObject::connect(lut_browser_widget_, &LutBrowserWidget::LutPathActivated, this,
                   [this](const QString& entry_path) {
                     const auto resolved_path = lut_controller_.TryResolveSelection(entry_path);
                     if (!resolved_path.has_value() || *resolved_path == state_.lut_path_) {
                       return;
                     }
                     state_.lut_path_ = *resolved_path;
                     CommitAdjustment(AdjustmentField::Lut);
                   });
  RefreshLutBrowserUi();
}

}  // namespace puerhlab::ui
