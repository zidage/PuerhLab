//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/versioning_panel_widget.hpp"

namespace alcedo::ui {

VersioningPanelWidget::VersioningPanelWidget(QWidget* parent) : QWidget(parent) {}

void EditorDialog::BuildVersioningPanel() {
  const auto& version_theme = AppTheme::Instance();

  // Primary action ("Commit All"): solid filled button, no border.
  const QString version_primary_btn_style =
      QStringLiteral("QPushButton {"
                     "  color: %1;"
                     "  background: %2;"
                     "  border: none;"
                     "  border-radius: 10px;"
                     "  font-size: 13px;"
                     "  font-weight: 600;"
                     "  padding: 8px 14px;"
                     "}"
                     "QPushButton:hover { background: %3; }"
                     "QPushButton:pressed { background: %4; }"
                     "QPushButton:disabled { color: %5; background: %6; }")
          .arg(version_theme.bgCanvasColor().name(QColor::HexRgb),
               version_theme.accentColor().name(QColor::HexArgb),
               QColor(version_theme.accentColor().red(), version_theme.accentColor().green(),
                      version_theme.accentColor().blue(), 232)
                   .name(QColor::HexArgb),
               version_theme.accentSecondaryColor().name(QColor::HexArgb),
               version_theme.textMutedColor().name(QColor::HexRgb),
               QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 160)
                   .name(QColor::HexArgb));

  // Secondary action ("Undo Last"): subtle gray pill, no border (avoid white strokes).
  const QString version_secondary_btn_style =
      QStringLiteral("QPushButton {"
                     "  color: %1;"
                     "  background: %2;"
                     "  border: none;"
                     "  border-radius: 10px;"
                     "  font-size: 13px;"
                     "  font-weight: 600;"
                     "  padding: 8px 14px;"
                     "}"
                     "QPushButton:hover { background: %3; }"
                     "QPushButton:pressed { background: %4; }"
                     "QPushButton:disabled { color: %5; background: %2; }")
          .arg(version_theme.textColor().name(QColor::HexRgb),
               QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 200)
                   .name(QColor::HexArgb),
               QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 238)
                   .name(QColor::HexArgb),
               QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 160)
                   .name(QColor::HexArgb),
               version_theme.textMutedColor().name(QColor::HexRgb));

  const QString compact_combo_font_style =
      QStringLiteral("QComboBox {"
                     "  font-size: 12px;"
                     "  background: %1;"
                     "  color: %2;"
                     "  border: none;"
                     "  border-radius: 8px;"
                     "  padding: 4px 10px;"
                     "}"
                     "QComboBox::drop-down { border: 0px; width: 20px; }"
                     "QComboBox QAbstractItemView {"
                     "  background: %1;"
                     "  color: %2;"
                     "  border: none;"
                     "  selection-background-color: %3;"
                     "  selection-color: %4;"
                     "}")
          .arg(QColor(version_theme.bgPanelColor().red(), version_theme.bgPanelColor().green(),
                      version_theme.bgPanelColor().blue(), 210)
                   .name(QColor::HexArgb),
               version_theme.textColor().name(QColor::HexRgb),
               QColor(version_theme.accentSecondaryColor().red(),
                      version_theme.accentSecondaryColor().green(),
                      version_theme.accentSecondaryColor().blue(), 224)
                   .name(QColor::HexArgb),
               version_theme.bgCanvasColor().name(QColor::HexRgb));

  // Subtle divider used for header separation (no borders to avoid aliasing).
  const QString divider_style =
      QStringLiteral("QFrame#VersioningDivider {"
                     "  background: %1;"
                     "  border: none;"
                     "  min-height: 1px;"
                     "  max-height: 1px;"
                     "}")
          .arg(QColor(version_theme.dividerColor().red(), version_theme.dividerColor().green(),
                      version_theme.dividerColor().blue(), 72)
                   .name(QColor::HexArgb));

  // Embedded list: transparent so the rows inherit the parent panel look.
  const QString embedded_list_style =
      QStringLiteral("QListWidget {"
                     "  background: transparent;"
                     "  border: none;"
                     "  border-radius: 0px;"
                     "  padding: 0px;"
                     "}"
                     "QListWidget::item {"
                     "  background: transparent;"
                     "  border: none;"
                     "  padding: 0px;"
                     "}"
                     "QListWidget::item:selected { background: transparent; }");

  shared_versioning_layout_->setContentsMargins(0, 0, 0, 0);
  shared_versioning_layout_->setSpacing(0);

  versioning_pages_stack_ = new QStackedWidget(shared_versioning_root_);
  versioning_pages_stack_->setObjectName("EditorVersioningPages");
  versioning_pages_stack_->setStyleSheet(QStringLiteral(
      "QStackedWidget#EditorVersioningPages { background: transparent; }"));
  shared_versioning_layout_->addWidget(versioning_pages_stack_, 1);

  const auto build_header_row = [&](const char* title_source, const char* pill_source) {
    auto* row = new QWidget(shared_versioning_root_);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    auto* title_label = NewLocalizedLabel(title_source, row);
    title_label->setStyleSheet(QStringLiteral("QLabel { color: %1; font-size: 15px; font-weight: 700; background: transparent; }")
                                   .arg(version_theme.textColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(title_label, AppTheme::FontRole::UiBodyStrong);
    layout->addWidget(title_label, 0);
    layout->addStretch(1);

    if (pill_source != nullptr && pill_source[0] != '\0') {
      auto* pill = NewLocalizedLabel(pill_source, row);
      pill->setStyleSheet(QStringLiteral("QLabel {"
                                         "  color: %1;"
                                         "  background: %2;"
                                         "  border: none;"
                                         "  border-radius: 10px;"
                                         "  font-size: 11px;"
                                         "  font-weight: 600;"
                                         "  padding: 4px 12px;"
                                         "}")
                              .arg(version_theme.textColor().name(QColor::HexRgb),
                                   QColor(version_theme.bgPanelColor().red(),
                                          version_theme.bgPanelColor().green(),
                                          version_theme.bgPanelColor().blue(), 220)
                                       .name(QColor::HexArgb)));
      AppTheme::MarkFontRole(pill, AppTheme::FontRole::UiCaption);
      layout->addWidget(pill, 0);
    }

    return row;
  };

  const auto build_divider = [&]() {
    auto* divider = new QFrame(shared_versioning_root_);
    divider->setObjectName("VersioningDivider");
    divider->setStyleSheet(divider_style);
    return divider;
  };

  // "COMMITTED STATE" label flanked by two thin lines — separates the
  // uncommitted transaction list from the baseline commit row on the
  // history page.
  const auto build_section_divider = [&](const char* label_source) {
    auto* row = new QWidget(shared_versioning_root_);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 4, 0, 4);
    layout->setSpacing(10);

    const QColor line_color =
        QColor(version_theme.dividerColor().red(), version_theme.dividerColor().green(),
               version_theme.dividerColor().blue(), 92);
    const QString line_style =
        QStringLiteral("QFrame { background: %1; border: none; min-height: 1px; max-height: 1px; }")
            .arg(line_color.name(QColor::HexArgb));

    auto* line_left = new QFrame(row);
    line_left->setStyleSheet(line_style);
    line_left->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    line_left->setFixedHeight(1);

    auto* label = NewLocalizedLabel(label_source, row);
    label->setStyleSheet(QStringLiteral("QLabel {"
                                        "  color: %1;"
                                        "  background: transparent;"
                                        "  font-size: 10px;"
                                        "  font-weight: 700;"
                                        "  letter-spacing: 2px;"
                                        "}")
                             .arg(version_theme.textMutedColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(label, AppTheme::FontRole::UiCaption);

    auto* line_right = new QFrame(row);
    line_right->setStyleSheet(line_style);
    line_right->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    line_right->setFixedHeight(1);

    layout->addWidget(line_left, 1);
    layout->addWidget(label, 0);
    layout->addWidget(line_right, 1);
    return row;
  };

  {
    auto* history_page = new QWidget(versioning_pages_stack_);
    history_page->setAttribute(Qt::WA_StyledBackground, false);
    history_page->setStyleSheet("background: transparent;");
    auto* page_layout  = new QVBoxLayout(history_page);
    page_layout->setContentsMargins(18, 18, 18, 18);
    page_layout->setSpacing(10);

    page_layout->addWidget(build_header_row("Edit History", "Uncommitted"), 0);
    page_layout->addWidget(build_divider(), 0);

    // Working transaction list — transparent so tx cards paint on the panel directly.
    tx_stack_ = new QListWidget(history_page);
    tx_stack_->setSelectionMode(QAbstractItemView::NoSelection);
    tx_stack_->setSpacing(4);
    tx_stack_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    tx_stack_->setMinimumHeight(170);
    tx_stack_->setStyleSheet(embedded_list_style);
    tx_stack_->setFrameShape(QFrame::NoFrame);
    page_layout->addWidget(tx_stack_, 1);

    // "COMMITTED STATE" divider — visual separator between the uncommitted
    // transaction list and the committed-baseline summary beneath it.
    page_layout->addWidget(build_section_divider("COMMITTED STATE"), 0);

    // Baseline summary row: icon + "Baseline" label + pending-edit status on
    // the right (reuses version_status_ which UpdateVersionUi populates).
    auto* baseline_row = new QWidget(history_page);
    auto* baseline_layout = new QHBoxLayout(baseline_row);
    baseline_layout->setContentsMargins(4, 4, 4, 4);
    baseline_layout->setSpacing(10);

    auto* baseline_label = NewLocalizedLabel("Baseline", baseline_row);
    baseline_label->setStyleSheet(
        QStringLiteral("QLabel { color: %1; background: transparent; font-size: 13px; font-weight: 600; }")
            .arg(version_theme.textColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(baseline_label, AppTheme::FontRole::UiBody);

    version_status_ = new QLabel(baseline_row);
    version_status_->setStyleSheet(
        QStringLiteral("QLabel { color: %1; background: transparent; font-size: 11px; font-weight: 500; }")
            .arg(version_theme.textMutedColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(version_status_, AppTheme::FontRole::DataCaption);

    baseline_layout->addWidget(baseline_label, 0);
    baseline_layout->addStretch(1);
    baseline_layout->addWidget(version_status_, 0);
    page_layout->addWidget(baseline_row, 0);

    auto* action_row = new QWidget(history_page);
    auto* action_layout = new QHBoxLayout(action_row);
    action_layout->setContentsMargins(0, 8, 0, 0);
    action_layout->setSpacing(10);

    undo_tx_btn_ = NewLocalizedButton("Undo Last", action_row);
    undo_tx_btn_->setMinimumHeight(38);
    undo_tx_btn_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    undo_tx_btn_->setStyleSheet(version_secondary_btn_style);

    commit_version_btn_ = NewLocalizedButton("Commit All", action_row);
    commit_version_btn_->setMinimumHeight(38);
    commit_version_btn_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    commit_version_btn_->setStyleSheet(version_primary_btn_style);

    action_layout->addWidget(undo_tx_btn_, 1);
    action_layout->addWidget(commit_version_btn_, 1);
    page_layout->addWidget(action_row, 0);

    QObject::connect(undo_tx_btn_, &QPushButton::clicked, this,
                     [this]() { UndoLastTransaction(); });
    QObject::connect(commit_version_btn_, &QPushButton::clicked, this,
                     [this]() { CommitWorkingVersion(); });

    versioning_pages_stack_->addWidget(history_page);
  }

  {
    auto* versions_page = new QWidget(versioning_pages_stack_);
    versions_page->setAttribute(Qt::WA_StyledBackground, false);
    versions_page->setStyleSheet("background: transparent;");
    auto* page_layout   = new QVBoxLayout(versions_page);
    page_layout->setContentsMargins(18, 18, 18, 18);
    page_layout->setSpacing(10);

    page_layout->addWidget(build_header_row("Version Tree", ""), 0);
    page_layout->addWidget(build_divider(), 0);

    // Working-mode control row — borderless, inline on the panel.
    auto* mode_row = new QWidget(versions_page);
    auto* mode_layout = new QHBoxLayout(mode_row);
    mode_layout->setContentsMargins(0, 2, 0, 2);
    mode_layout->setSpacing(10);

    auto* mode_label = NewLocalizedLabel("Working mode", mode_row);
    mode_label->setStyleSheet(
        QStringLiteral("QLabel { color: %1; background: transparent; font-size: 11px; font-weight: 600; letter-spacing: 1px; }")
            .arg(version_theme.textMutedColor().name(QColor::HexRgb)));
    AppTheme::MarkFontRole(mode_label, AppTheme::FontRole::UiCaption);

    working_mode_combo_ = new QComboBox(mode_row);
    working_mode_combo_->addItem(Tr("Plain"), static_cast<int>(WorkingMode::Plain));
    working_mode_combo_->addItem(Tr("Incremental"), static_cast<int>(WorkingMode::Incremental));
    working_mode_combo_->setMinimumHeight(32);
    working_mode_combo_->setMinimumWidth(118);
    working_mode_combo_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    working_mode_combo_->setStyleSheet(compact_combo_font_style);

    new_working_btn_ = NewLocalizedButton("New Working", mode_row);
    new_working_btn_->setMinimumHeight(32);
    new_working_btn_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    new_working_btn_->setStyleSheet(version_secondary_btn_style);

    mode_layout->addWidget(mode_label, 0);
    mode_layout->addWidget(working_mode_combo_, 1);
    mode_layout->addWidget(new_working_btn_, 0);
    page_layout->addWidget(mode_row, 0);

    page_layout->addWidget(build_section_divider("COMMITTED STATE"), 0);

    version_log_ = new QListWidget(versions_page);
    version_log_->setSelectionMode(QAbstractItemView::SingleSelection);
    version_log_->setSpacing(2);
    version_log_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    version_log_->setMinimumHeight(170);
    version_log_->setStyleSheet(embedded_list_style);
    version_log_->setFrameShape(QFrame::NoFrame);
    page_layout->addWidget(version_log_, 1);

    QObject::connect(new_working_btn_, &QPushButton::clicked, this,
                     [this]() { StartNewWorkingVersionFromUi(); });
    QObject::connect(working_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                     this, [this](int) { UpdateVersionUi(); });
    QObject::connect(version_log_, &QListWidget::itemSelectionChanged, this,
                     [this]() { RefreshVersionLogSelectionStyles(); });
    QObject::connect(version_log_, &QListWidget::itemClicked, this,
                     [this](QListWidgetItem* item) {
                       if (!item) {
                         return;
                       }
                       const QString version_id = item->data(Qt::UserRole).toString();
                       QTimer::singleShot(0, this,
                                          [this, version_id]() { CheckoutVersionById(version_id); });
                     });

    versioning_pages_stack_->addWidget(versions_page);
  }

  versioning_pages_stack_->setCurrentIndex(
      static_cast<int>(VersioningFlyoutPage::History));
  SetVersioningCollapsed(true, /*animate=*/false);
}

}  // namespace alcedo::ui
