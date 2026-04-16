// ctor_versioning.inc -- Constructor section for EditorDialog.
// This file is #included from the constructor body in dialog.cpp.
// It shares the constructor scope and has access to all local variables.

  const auto&  version_theme               = AppTheme::Instance();
  const QString version_primary_btn_style   = AppTheme::EditorPrimaryButtonStyle(true);
  const QString version_secondary_btn_style = AppTheme::EditorSecondaryButtonStyle();
  const QString compact_button_font_style   =
      QStringLiteral("QPushButton { font-size: 12px; font-weight: 500; }");
  const QString compact_combo_font_style    =
      QStringLiteral("QComboBox { font-size: 12px; }");
  const QString version_section_panel_style =
      QStringLiteral("QFrame#VersioningSectionPanel {"
                     "  background: %1;"
                     "  border: none;"
                     "  border-radius: 8px;"
                     "}")
          .arg(QColor(version_theme.bgBaseColor().red(), version_theme.bgBaseColor().green(),
                      version_theme.bgBaseColor().blue(), 238)
                   .name(QColor::HexArgb));
  const QString embedded_list_style =
      AppTheme::EditorListWidgetStyle() +
      QStringLiteral("QListWidget {"
                     "  background: transparent;"
                     "  border: none;"
                     "  border-radius: 0px;"
                     "  padding: 0px;"
                     "}");

    // Action row 1 (uncommitted changes).
    {
      auto* row       = new QWidget(shared_versioning_root);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 4, 0, 0);
      rowLayout->setSpacing(8);

      version_status_ = new QLabel(row);
      version_status_->setStyleSheet(
          QStringLiteral("QLabel { color: %1; font-size: 12px; }")
              .arg(version_theme.textMutedColor().name(QColor::HexRgb)));
      AppTheme::MarkFontRole(version_status_, AppTheme::FontRole::UiBodyStrong);
      version_status_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

      undo_tx_btn_ = new QPushButton(Tr("Undo"), row);
      undo_tx_btn_->setMinimumHeight(28);
      undo_tx_btn_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
      undo_tx_btn_->setStyleSheet(version_secondary_btn_style + compact_button_font_style);

      commit_version_btn_ = new QPushButton(Tr("Commit Version"), row);
      commit_version_btn_->setMinimumHeight(28);
      commit_version_btn_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
      commit_version_btn_->setStyleSheet(version_primary_btn_style + compact_button_font_style);

      rowLayout->addWidget(version_status_, /*stretch*/ 0);
      rowLayout->addStretch(1);
      rowLayout->addWidget(undo_tx_btn_, /*stretch*/ 0);
      rowLayout->addWidget(commit_version_btn_, /*stretch*/ 0);
      shared_versioning_layout->addWidget(row, 0);

      QObject::connect(undo_tx_btn_, &QPushButton::clicked, this,
                       [this]() { UndoLastTransaction(); });
      QObject::connect(commit_version_btn_, &QPushButton::clicked, this,
                       [this]() { CommitWorkingVersion(); });
    }

    // Action row 2 (working version).
    {
      auto* mode_row    = new QWidget(shared_versioning_root);
      auto* mode_layout = new QHBoxLayout(mode_row);
      mode_layout->setContentsMargins(0, 0, 0, 0);
      mode_layout->setSpacing(8);

      auto* mode_label = new QLabel(Tr("Working version:"), mode_row);
      mode_label->setStyleSheet(AppTheme::EditorLabelStyle(version_theme.textMutedColor()));
      AppTheme::MarkFontRole(mode_label, AppTheme::FontRole::UiCaption);

      working_mode_combo_ = new QComboBox(mode_row);
      working_mode_combo_->addItem(Tr("Plain"), static_cast<int>(WorkingMode::Plain));
      working_mode_combo_->addItem(Tr("Incr"), static_cast<int>(WorkingMode::Incremental));
      working_mode_combo_->setMinimumHeight(28);
      working_mode_combo_->setMinimumWidth(68);
      working_mode_combo_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
      working_mode_combo_->setStyleSheet(AppTheme::EditorComboBoxStyle() + compact_combo_font_style);

      new_working_btn_ = new QPushButton(Tr("New"), mode_row);
      new_working_btn_->setMinimumHeight(28);
      new_working_btn_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
      new_working_btn_->setStyleSheet(version_secondary_btn_style + compact_button_font_style);

      mode_layout->addWidget(mode_label, /*stretch*/ 0);
      mode_layout->addWidget(working_mode_combo_, /*stretch*/ 1);
      mode_layout->addWidget(new_working_btn_, /*stretch*/ 0);
      shared_versioning_layout->addWidget(mode_row, 0);

      QObject::connect(new_working_btn_, &QPushButton::clicked, this,
                       [this]() { StartNewWorkingVersionFromUi(); });
      QObject::connect(working_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                       this, [this](int) { UpdateVersionUi(); });
    }

    // Versions section.
    {
      auto* versions_label = new QLabel(Tr("Versions"), shared_versioning_root);
      versions_label->setStyleSheet(
          QStringLiteral("QLabel { color: %1; font-size: 12px; font-weight: 600; }")
              .arg(version_theme.textColor().name(QColor::HexRgb)));
      AppTheme::MarkFontRole(versions_label, AppTheme::FontRole::UiBodyStrong);
      shared_versioning_layout->addWidget(versions_label, 0);

      auto* versions_panel = new QFrame(shared_versioning_root);
      versions_panel->setObjectName("VersioningSectionPanel");
      versions_panel->setStyleSheet(version_section_panel_style);
      auto* versions_panel_layout = new QVBoxLayout(versions_panel);
      versions_panel_layout->setContentsMargins(12, 12, 12, 12);
      versions_panel_layout->setSpacing(0);

      version_log_ = new QListWidget(versions_panel);
      version_log_->setSelectionMode(QAbstractItemView::SingleSelection);
      version_log_->setSpacing(6);
      version_log_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      version_log_->setMinimumHeight(140);
      version_log_->setStyleSheet(embedded_list_style);
      versions_panel_layout->addWidget(version_log_, 1);
      shared_versioning_layout->addWidget(versions_panel, 0);

      QObject::connect(version_log_, &QListWidget::itemSelectionChanged, this,
                       [this]() { RefreshVersionLogSelectionStyles(); });
      QObject::connect(version_log_, &QListWidget::itemClicked, this,
                       [this](QListWidgetItem* item) { CheckoutSelectedVersion(item); });
    }

    // Uncommitted transactions stack section.
    {
      auto* tx_label = new QLabel(Tr("Uncommitted transactions (stack)"), shared_versioning_root);
      tx_label->setStyleSheet(
          QStringLiteral("QLabel { color: %1; font-size: 12px; font-weight: 600; }")
              .arg(version_theme.textColor().name(QColor::HexRgb)));
      AppTheme::MarkFontRole(tx_label, AppTheme::FontRole::UiBodyStrong);
      shared_versioning_layout->addWidget(tx_label, 0);

      auto* tx_panel = new QFrame(shared_versioning_root);
      tx_panel->setObjectName("VersioningSectionPanel");
      tx_panel->setStyleSheet(version_section_panel_style);
      auto* tx_panel_layout = new QVBoxLayout(tx_panel);
      tx_panel_layout->setContentsMargins(12, 12, 12, 12);
      tx_panel_layout->setSpacing(0);

      tx_stack_ = new QListWidget(tx_panel);
      tx_stack_->setSelectionMode(QAbstractItemView::NoSelection);
      tx_stack_->setSpacing(6);
      tx_stack_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      tx_stack_->setMinimumHeight(150);
      tx_stack_->setStyleSheet(embedded_list_style);
      tx_panel_layout->addWidget(tx_stack_, 1);
      shared_versioning_layout->addWidget(tx_panel, 1);
      shared_versioning_layout->addStretch();
    }

    SetVersioningCollapsed(true, /*animate=*/false);
