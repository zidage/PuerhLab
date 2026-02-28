// ctor_versioning.inc -- Constructor section for EditorDialog.
// This file is #included from the constructor body in dialog.cpp.
// It shares the constructor scope and has access to all local variables.


    // Edit-history commit controls.
    {
      auto* row       = new QWidget(shared_versioning_root);
      auto* rowLayout = new QHBoxLayout(row);
      rowLayout->setContentsMargins(0, 0, 0, 0);
      rowLayout->setSpacing(10);

      version_status_ = new QLabel(row);
      version_status_->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 12px;"
          "}");
      version_status_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
      version_status_->setMinimumWidth(0);

      undo_tx_btn_ = new QPushButton("Undo", row);
      undo_tx_btn_->setFixedHeight(32);
      undo_tx_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 6px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #F5C200;"
          "}"
          "QPushButton:disabled {"
          "  color: #6A6A6A;"
          "  background: #1A1A1A;"
          "}");

      commit_version_btn_ = new QPushButton("Commit Version", row);
      commit_version_btn_->setFixedHeight(32);
      commit_version_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 6px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #F5C200;"
          "}"
          "QPushButton:disabled {"
          "  color: #6A6A6A;"
          "  background: #1A1A1A;"
          "}");

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
      frame->setStyleSheet(
          "QFrame {"
          "  background: transparent;"
          "  border: none;"
          "  border-radius: 12px;"
          "}");
      auto* layout = new QVBoxLayout(frame);
      layout->setContentsMargins(10, 10, 10, 10);
      layout->setSpacing(8);

      auto* mode_row    = new QWidget(frame);
      auto* mode_layout = new QHBoxLayout(mode_row);
      mode_layout->setContentsMargins(0, 0, 0, 0);
      mode_layout->setSpacing(8);

      auto* mode_label = new QLabel("Working version:", mode_row);
      mode_label->setStyleSheet(
          "QLabel {"
          "  color: #A3A3A3;"
          "  font-size: 12px;"
          "}");

      working_mode_combo_ = new QComboBox(mode_row);
      working_mode_combo_->addItem("Plain", static_cast<int>(WorkingMode::Plain));
      working_mode_combo_->addItem("Incr", static_cast<int>(WorkingMode::Incremental));
      working_mode_combo_->setFixedHeight(28);
      working_mode_combo_->setStyleSheet(
          "QComboBox {"
          "  background: #1A1A1A;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 4px 8px;"
          "}"
          "QComboBox::drop-down {"
          "  border: 0px;"
          "  width: 24px;"
          "}"
          "QComboBox QAbstractItemView {"
          "  background: #1A1A1A;"
          "  border: none;"
          "  selection-background-color: #FCC704;"
          "  selection-color: #121212;"
          "}");

      new_working_btn_ = new QPushButton("New", mode_row);
      new_working_btn_->setFixedHeight(28);
      new_working_btn_->setStyleSheet(
          "QPushButton {"
          "  color: #121212;"
          "  background: #FCC704;"
          "  border: none;"
          "  border-radius: 8px;"
          "  padding: 4px 10px;"
          "}"
          "QPushButton:hover {"
          "  background: #F5C200;"
          "}");

      mode_layout->addWidget(mode_label, /*stretch*/ 0);
      mode_layout->addWidget(working_mode_combo_, /*stretch*/ 1);
      mode_layout->addWidget(new_working_btn_, /*stretch*/ 0);

      layout->addWidget(mode_row);

      auto* versions_label = new QLabel("Versions", frame);
      versions_label->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 12px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(versions_label);

      version_log_ = new QListWidget(frame);
      version_log_->setSelectionMode(QAbstractItemView::SingleSelection);
      version_log_->setSpacing(6);
      version_log_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      version_log_->setMinimumHeight(110);
      version_log_->setStyleSheet(
          "QListWidget {"
          "  background: #121212;"
          "  border: none;"
          "  border-radius: 10px;"
          "  padding: 6px;"
          "}"
          "QListWidget::item {"
          "  padding: 2px;"
          "}"
          "QListWidget::item:selected {"
          "  background: transparent;"
          "}");
      layout->addWidget(version_log_);

      QObject::connect(version_log_, &QListWidget::itemSelectionChanged, this,
                       [this]() { RefreshVersionLogSelectionStyles(); });
      QObject::connect(version_log_, &QListWidget::itemClicked, this,
                       [this](QListWidgetItem* item) { CheckoutSelectedVersion(item); });

      auto* tx_label = new QLabel("Uncommitted transactions (stack)", frame);
      tx_label->setStyleSheet(
          "QLabel {"
          "  color: #E6E6E6;"
          "  font-size: 12px;"
          "  font-weight: 500;"
          "}");
      layout->addWidget(tx_label);

      tx_stack_ = new QListWidget(frame);
      tx_stack_->setSelectionMode(QAbstractItemView::NoSelection);
      tx_stack_->setSpacing(6);
      tx_stack_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
      tx_stack_->setMinimumHeight(130);
      tx_stack_->setStyleSheet(
          "QListWidget {"
          "  background: #121212;"
          "  border: none;"
          "  border-radius: 10px;"
          "  padding: 6px;"
          "}"
          "QListWidget::item {"
          "  padding: 2px;"
          "}");
      layout->addWidget(tx_stack_, /*stretch*/ 1);

      shared_versioning_layout->addWidget(frame, 0);
      shared_versioning_layout->addStretch();

      QObject::connect(new_working_btn_, &QPushButton::clicked, this,
                       [this]() { StartNewWorkingVersionFromUi(); });
      QObject::connect(working_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                       this, [this](int) { UpdateVersionUi(); });
    }

