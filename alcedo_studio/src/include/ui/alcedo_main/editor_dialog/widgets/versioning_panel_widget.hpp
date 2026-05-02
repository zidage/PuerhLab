//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QComboBox>
#include <QGraphicsOpacityEffect>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QStackedWidget>
#include <QString>
#include <QVBoxLayout>
#include <QVariantAnimation>
#include <QWidget>
#include <functional>

#include "ui/alcedo_main/editor_dialog/modules/versioning.hpp"

namespace alcedo::ui {

class VersioningPanelWidget final : public QWidget {
 public:
  enum class WorkingMode : int { Incremental = 0, Plain = 1 };
  enum class FlyoutPage : int { History = 0, Versions = 1 };

  static constexpr int kCollapsedWidth = 64;

  struct Callbacks {
    std::function<void()>              undo_last_transaction;
    std::function<void()>              commit_working_version;
    std::function<void()>              start_new_working_version;
    std::function<void(const QString&)> checkout_version_by_id;
    std::function<void()>              on_working_mode_changed;
    std::function<QRect()>             viewer_geometry;
  };

  explicit VersioningPanelWidget(QWidget* parent = nullptr);

  void Configure(QWidget* flyout_parent, Callbacks callbacks);
  void Build();
  void RetranslateUi();

  auto MakeUiContext() const -> versioning::VersionUiContext;

  auto CurrentWorkingMode() const -> WorkingMode;
  auto IsPlainWorkingMode() const -> bool;

  auto UndoButton() const -> QPushButton* { return undo_tx_btn_; }
  auto IsCollapsed() const -> bool { return collapsed_; }
  auto IsFlyoutVisible() const -> bool;

  void SetCollapsed(bool collapsed, bool animate = true);
  void OnDialogResized();
  void RefreshVersionLogSelectionStyles();

 protected:
  bool eventFilter(QObject* obj, QEvent* event) override;

 private:
  void BuildRail();
  void BuildFlyout();
  void RefreshCollapseUi();
  void RepositionFlyout();
  void HandleHistoryButtonClicked();
  void HandleVersionsButtonClicked();

  Callbacks callbacks_{};
  QWidget*  flyout_parent_ = nullptr;

  // Rail (this widget hosts the rail directly).
  QWidget*     rail_         = nullptr;
  QPushButton* history_btn_  = nullptr;
  QPushButton* versions_btn_ = nullptr;

  // Floating flyout overlay.
  QWidget*                flyout_                = nullptr;
  QWidget*                flyout_root_           = nullptr;
  QGraphicsOpacityEffect* flyout_opacity_effect_ = nullptr;
  QVariantAnimation*      flyout_anim_           = nullptr;
  QStackedWidget*         pages_stack_           = nullptr;
  QVBoxLayout*            shared_layout_         = nullptr;

  // Page widgets.
  QLabel*      version_status_     = nullptr;
  QPushButton* undo_tx_btn_        = nullptr;
  QPushButton* commit_version_btn_ = nullptr;
  QListWidget* tx_stack_           = nullptr;
  QComboBox*   working_mode_combo_ = nullptr;
  QPushButton* new_working_btn_    = nullptr;
  QListWidget* version_log_        = nullptr;

  bool       collapsed_   = true;
  qreal      progress_    = 0.0;
  FlyoutPage active_page_ = FlyoutPage::History;
  bool       built_       = false;
};

}  // namespace alcedo::ui
