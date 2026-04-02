//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/widgets/lut_browser_widget.hpp"

#include <QAbstractItemView>
#include <QCoreApplication>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QVBoxLayout>

#include "ui/puerhlab_main/app_theme.hpp"
#include "ui/puerhlab_main/i18n.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/history_cards.hpp"

namespace puerhlab::ui {
namespace {

auto Tr(const char* text) -> QString {
  return QCoreApplication::translate(PUERHLAB_I18N_CONTEXT, text);
}

class LutEntryItemWidget final : public QFrame {
 public:
  explicit LutEntryItemWidget(QWidget* parent = nullptr) : QFrame(parent) {
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(12, 10, 12, 10);
    layout->setSpacing(10);

    auto* text_layout = new QVBoxLayout();
    text_layout->setContentsMargins(0, 0, 0, 0);
    text_layout->setSpacing(4);

    title_label_ = new ElidedLabel({}, this);
    title_label_->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
    AppTheme::MarkFontRole(title_label_, AppTheme::FontRole::UiBodyStrong);

    subtitle_label_ = new ElidedLabel({}, this);
    subtitle_label_->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
    AppTheme::MarkFontRole(subtitle_label_, AppTheme::FontRole::UiCaption);

    text_layout->addWidget(title_label_);
    text_layout->addWidget(subtitle_label_);

    status_label_ = MakePillLabel({}, this);
    status_label_->hide();

    layout->addLayout(text_layout, 1);
    layout->addWidget(status_label_, 0, Qt::AlignTop);
  }

  void Bind(const lut_catalog::LutCatalogEntry& entry) {
    entry_ = entry;
    const auto& theme = AppTheme::Instance();
    title_label_->SetRawText(entry.display_name_);
    subtitle_label_->SetRawText(entry.secondary_text_);
    status_label_->setVisible(!entry.status_text_.isEmpty());
    status_label_->setText(entry.status_text_);

    if (entry.kind_ == lut_catalog::LutCatalogEntryKind::MissingCurrent) {
      status_label_->setStyleSheet(QStringLiteral("QLabel {"
                                                  "  color: #FFB454;"
                                                  "  background: rgba(255, 180, 84, 0.16);"
                                                  "  border: none;"
                                                  "  border-radius: 6px;"
                                                  "  padding: 2px 6px;"
                                                  "}"));
    } else if (!entry.valid_) {
      status_label_->setStyleSheet(QStringLiteral("QLabel {"
                                                  "  color: #FF7A7A;"
                                                  "  background: rgba(255, 122, 122, 0.16);"
                                                  "  border: none;"
                                                  "  border-radius: 6px;"
                                                  "  padding: 2px 6px;"
                                                  "}"));
    } else {
      status_label_->setStyleSheet(QStringLiteral("QLabel {"
                                                  "  color: %1;"
                                                  "  background: %2;"
                                                  "  border: 1px solid %3;"
                                                  "  border-radius: 6px;"
                                                  "  padding: 2px 6px;"
                                                  "}")
                                       .arg(theme.accentColor().name(QColor::HexRgb),
                                            QColor(theme.accentColor().red(),
                                                   theme.accentColor().green(),
                                                   theme.accentColor().blue(), 28)
                                                .name(QColor::HexArgb),
                                            QColor(theme.accentColor().red(),
                                                   theme.accentColor().green(),
                                                   theme.accentColor().blue(), 62)
                                                .name(QColor::HexArgb)));
    }
    SetSelected(false);
  }

  void SetSelected(bool selected) {
    QString border_color = AppTheme::Instance().dividerColor().name(QColor::HexArgb);
    QString background   = QColor(AppTheme::Instance().bgDeepColor().red(),
                                  AppTheme::Instance().bgDeepColor().green(),
                                  AppTheme::Instance().bgDeepColor().blue(), 230)
                             .name(QColor::HexArgb);
    QString title_color  = AppTheme::Instance().textColor().name(QColor::HexRgb);
    QString sub_color    = AppTheme::Instance().textMutedColor().name(QColor::HexRgb);

    if (!entry_.valid_) {
      border_color = entry_.kind_ == lut_catalog::LutCatalogEntryKind::MissingCurrent
                         ? QStringLiteral("#FFB454")
                         : QStringLiteral("#FF7A7A");
      background = QStringLiteral("#171313");
      sub_color  = entry_.kind_ == lut_catalog::LutCatalogEntryKind::MissingCurrent
                       ? QStringLiteral("#FFD39C")
                       : QStringLiteral("#FFB8B8");
    }
    if (selected) {
      border_color = QColor(AppTheme::Instance().accentColor().red(),
                            AppTheme::Instance().accentColor().green(),
                            AppTheme::Instance().accentColor().blue(), 148)
                         .name(QColor::HexArgb);
      background   = AppTheme::Instance().selectedTintColor().name(QColor::HexArgb);
    }

    setStyleSheet(QStringLiteral("QFrame {"
                                 "  background: %1;"
                                 "  border: 1px solid %2;"
                                 "  border-radius: 10px;"
                                 "}"
                                 "QLabel { border: none; background: transparent; }")
                      .arg(background, border_color));
    title_label_->setStyleSheet(AppTheme::EditorLabelStyle(QColor(title_color)));
    subtitle_label_->setStyleSheet(AppTheme::EditorLabelStyle(QColor(sub_color)));
  }

 private:
  lut_catalog::LutCatalogEntry entry_{};
  ElidedLabel*                 title_label_    = nullptr;
  ElidedLabel*                 subtitle_label_ = nullptr;
  QLabel*                      status_label_   = nullptr;
};

}  // namespace

LutBrowserWidget::LutBrowserWidget(QWidget* parent) : QWidget(parent) {
  auto* root = new QVBoxLayout(this);
  root->setContentsMargins(12, 12, 12, 12);
  root->setSpacing(10);

  title_label_ = new QLabel(this);
  title_label_->setObjectName("SectionTitle");
  title_label_->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
  AppTheme::MarkFontRole(title_label_, AppTheme::FontRole::UiHeadline);
  root->addWidget(title_label_, 0);

  subtitle_label_ = new QLabel(this);
  subtitle_label_->setWordWrap(true);
  subtitle_label_->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
  AppTheme::MarkFontRole(subtitle_label_, AppTheme::FontRole::UiHint);
  root->addWidget(subtitle_label_, 0);

  auto* actions_row = new QWidget(this);
  auto* actions_layout = new QHBoxLayout(actions_row);
  actions_layout->setContentsMargins(0, 0, 0, 0);
  actions_layout->setSpacing(8);

  directory_label_ = new QLabel(actions_row);
  directory_label_->setWordWrap(true);
  directory_label_->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textColor()));
  AppTheme::MarkFontRole(directory_label_, AppTheme::FontRole::UiCaptionStrong);

  open_folder_btn_ = new QPushButton(actions_row);
  open_folder_btn_->setFixedHeight(30);
  open_folder_btn_->setStyleSheet(AppTheme::EditorSecondaryButtonStyle());

  refresh_btn_ = new QPushButton(actions_row);
  refresh_btn_->setFixedHeight(30);
  refresh_btn_->setStyleSheet(AppTheme::EditorPrimaryButtonStyle());

  actions_layout->addWidget(directory_label_, 1);
  actions_layout->addWidget(open_folder_btn_, 0);
  actions_layout->addWidget(refresh_btn_, 0);
  root->addWidget(actions_row, 0);

  status_label_ = new QLabel(this);
  status_label_->setWordWrap(true);
  status_label_->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
  AppTheme::MarkFontRole(status_label_, AppTheme::FontRole::UiCaption);
  root->addWidget(status_label_, 0);

  entries_list_ = new QListWidget(this);
  entries_list_->setSpacing(6);
  entries_list_->setSelectionMode(QAbstractItemView::SingleSelection);
  entries_list_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
  entries_list_->setFocusPolicy(Qt::NoFocus);
  entries_list_->setStyleSheet(AppTheme::EditorListWidgetStyle());
  entries_list_->setFrameShape(QFrame::NoFrame);
  root->addWidget(entries_list_, 1);

  QObject::connect(open_folder_btn_, &QPushButton::clicked, this,
                   [this]() { emit OpenFolderRequested(); });
  QObject::connect(refresh_btn_, &QPushButton::clicked, this,
                   [this]() { emit RefreshRequested(); });
  QObject::connect(entries_list_, &QListWidget::currentItemChanged, this,
                   [this](QListWidgetItem* current, QListWidgetItem*) {
                     RefreshSelectionStyles();
                     if (updating_entries_ || !current) {
                       return;
                     }
                     const int row = entries_list_->row(current);
                     if (row < 0 || row >= static_cast<int>(entries_.size())) {
                       return;
                     }
                     if (!entries_[static_cast<size_t>(row)].selectable_) {
                       return;
                     }
                     emit LutPathActivated(QString::fromStdString(entries_[static_cast<size_t>(row)].path_));
                   });

  RetranslateUi();
}

void LutBrowserWidget::RetranslateUi() {
  title_label_->setText(Tr("Look"));
  subtitle_label_->setText(Tr("Browse LUT files and apply one to the current image."));
  open_folder_btn_->setText(Tr("Open Folder"));
  refresh_btn_->setText(Tr("Refresh"));
}

void LutBrowserWidget::SetDirectoryInfo(const QString& directory_text, const QString& status_text,
                                        bool can_open_directory) {
  directory_label_->setText(directory_text);
  status_label_->setText(status_text);
  open_folder_btn_->setEnabled(can_open_directory);
}

void LutBrowserWidget::SetEntries(const std::vector<lut_catalog::LutCatalogEntry>& entries,
                                  const QString& selected_path) {
  updating_entries_ = true;
  entries_          = entries;
  entries_list_->clear();

  int selected_row = -1;
  for (int i = 0; i < static_cast<int>(entries_.size()); ++i) {
    const auto& entry = entries_[static_cast<size_t>(i)];
    auto* item = new QListWidgetItem(entries_list_);
    item->setSizeHint(QSize(0, 66));
    item->setData(Qt::UserRole, QString::fromStdString(entry.path_));

    auto* row_widget = new LutEntryItemWidget(entries_list_);
    row_widget->Bind(entry);
    entries_list_->setItemWidget(item, row_widget);

    if (QString::fromStdString(entry.path_) == selected_path) {
      selected_row = i;
    }
  }

  if (selected_row < 0 && !entries_.empty()) {
    selected_row = 0;
  }
  if (selected_row >= 0) {
    entries_list_->setCurrentRow(selected_row);
  }

  RefreshSelectionStyles();
  updating_entries_ = false;
}

void LutBrowserWidget::RefreshSelectionStyles() {
  const int current_row = entries_list_ ? entries_list_->currentRow() : -1;
  for (int i = 0; i < entries_list_->count(); ++i) {
    auto* item = entries_list_->item(i);
    if (!item) {
      continue;
    }
    auto* widget = dynamic_cast<LutEntryItemWidget*>(entries_list_->itemWidget(item));
    if (!widget) {
      continue;
    }
    widget->SetSelected(i == current_row);
  }
}

}  // namespace puerhlab::ui
