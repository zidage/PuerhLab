// ctor_versioning.inc -- Constructor section for EditorDialog.
// This file is #included from the constructor body in dialog.cpp.
// It shares the constructor scope and has access to all local variables.


  const QString version_primary_btn_style = QStringLiteral(
    "QPushButton {"
    "  color: #141414;"
    "  background: #FCC704;"
    "  border: none;"
    "  border-radius: 6px;"
    "  padding: 0 12px;"
    "}"
    "QPushButton:hover {"
    "  background: #F5C200;"
    "}"
    "QPushButton:pressed {"
    "  background: #E8B800;"
    "}"
    "QPushButton:disabled {"
    "  color: #6A6A6A;"
    "  background: #222222;"
    "}");

  const QString version_secondary_btn_style = QStringLiteral(
    "QPushButton {"
    "  color: #FCC704;"
    "  background: transparent;"
    "  border: 1px solid rgba(252, 199, 4, 0.85);"
    "  border-radius: 6px;"
    "  padding: 0 12px;"
    "}"
    "QPushButton:hover {"
    "  color: #141414;"
    "  background: #FCC704;"
    "}"
    "QPushButton:pressed {"
    "  background: #E8B800;"
    "}"
    "QPushButton:disabled {"
    "  color: #666666;"
    "  border-color: #3A3A3A;"
    "  background: transparent;"
    "}");

    // Edit-history commit controls.
    {
      auto* row       = new QWidget(shared_versioning_root);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(10);

      version_status_ = new QLabel(row);
      version_status_->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
      AppTheme::MarkFontRole(version_status_, AppTheme::FontRole::DataBody);
      version_status_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
      version_status_->setMinimumWidth(0);

      undo_tx_btn_ = new QPushButton(Tr("Undo"), row);
      undo_tx_btn_->setFixedHeight(32);
      undo_tx_btn_->setStyleSheet(version_secondary_btn_style);

      commit_version_btn_ = new QPushButton(Tr("Commit Version"), row);
      commit_version_btn_->setFixedHeight(32);
      commit_version_btn_->setStyleSheet(version_primary_btn_style);

      rowLayout->addWidget(version_status_, /*stretch*/ 1);
      rowLayout->addWidget(undo_tx_btn_, /*stretch*/ 0);
      rowLayout->addWidget(commit_version_btn_, /*stretch*/ 0);
      shared_versioning_layout->addWidget(row, 0);

      QObject::connect(undo_tx_btn_, &QPushButton::clicked, this,
                       [this]() { UndoLastTransaction(); });
      QObject::connect(commit_version_btn_, &QPushButton::clicked, this,
                       [this]() { CommitWorkingVersion(); });
    }

    // Edit-history visualization ("git log"-like) + working version mode.
    {
      auto* frame = new QFrame(shared_versioning_root);
      frame->setStyleSheet(AppTheme::EditorTransparentFrameStyle());
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(10, 10, 10, 10);
      layout->setSpacing(8);

      auto* mode_row    = new QWidget(frame);
      auto* mode_layout = new QHBoxLayout(mode_row);
      mode_layout->setContentsMargins(0, 0, 0, 0);
      mode_layout->setSpacing(8);

      auto* mode_label = new QLabel(Tr("Working version:"), mode_row);
      mode_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
      AppTheme::MarkFontRole(mode_label, AppTheme::FontRole::UiCaption);

      working_mode_combo_ = new QComboBox(mode_row);
      working_mode_combo_->addItem(Tr("Plain"), static_cast<int>(WorkingMode::Plain));
      working_mode_combo_->addItem(Tr("Incr"), static_cast<int>(WorkingMode::Incremental));
      working_mode_combo_->setFixedHeight(28);
      working_mode_combo_->setStyleSheet(AppTheme::EditorComboBoxStyle());

      new_working_btn_ = new QPushButton(Tr("New"), mode_row);
      new_working_btn_->setFixedHeight(28);
      new_working_btn_->setStyleSheet(version_secondary_btn_style);

      mode_layout->addWidget(mode_label, /*stretch*/ 0);
      mode_layout->addWidget(working_mode_combo_, /*stretch*/ 1);
      mode_layout->addWidget(new_working_btn_, /*stretch*/ 0);

      layout->addWidget(mode_row);

      auto* versions_label = new QLabel(Tr("Versions"), frame);
      versions_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
      AppTheme::MarkFontRole(versions_label, AppTheme::FontRole::UiBodyStrong);
      layout->addWidget(versions_label);

      version_log_ = new QListWidget(frame);
      version_log_->setSelectionMode(QAbstractItemView::SingleSelection);
      version_log_->setSpacing(6);
      version_log_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      version_log_->setMinimumHeight(110);
      version_log_->setStyleSheet(AppTheme::EditorListWidgetStyle());
      layout->addWidget(version_log_);

      QObject::connect(version_log_, &QListWidget::itemSelectionChanged, this,
                       [this]() { RefreshVersionLogSelectionStyles(); });
      QObject::connect(version_log_, &QListWidget::itemClicked, this,
                       [this](QListWidgetItem* item) { CheckoutSelectedVersion(item); });

      auto* tx_label = new QLabel(Tr("Uncommitted transactions (stack)"), frame);
      tx_label->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
      AppTheme::MarkFontRole(tx_label, AppTheme::FontRole::UiBodyStrong);
      layout->addWidget(tx_label);

      tx_stack_ = new QListWidget(frame);
      tx_stack_->setSelectionMode(QAbstractItemView::NoSelection);
      tx_stack_->setSpacing(6);
      tx_stack_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      tx_stack_->setMinimumHeight(130);
      tx_stack_->setStyleSheet(AppTheme::EditorListWidgetStyle());
      layout->addWidget(tx_stack_, /*stretch*/ 1);

      shared_versioning_layout->addWidget(frame, 0);
      shared_versioning_layout->addStretch();

      QObject::connect(new_working_btn_, &QPushButton::clicked, this,
                       [this]() { StartNewWorkingVersionFromUi(); });
      QObject::connect(working_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                       this, [this](int) { UpdateVersionUi(); });
    }

