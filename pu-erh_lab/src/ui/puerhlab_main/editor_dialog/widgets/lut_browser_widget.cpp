//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/widgets/lut_browser_widget.hpp"
#include <qlabel.h>

#include <QAbstractItemView>
#include <QComboBox>
#include <QCoreApplication>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QVBoxLayout>
#include <QVariant>
#include <algorithm>

#include "ui/puerhlab_main/app_theme.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/history_cards.hpp"
#include "ui/puerhlab_main/i18n.hpp"

namespace puerhlab::ui {
namespace {

auto Tr(const char* text) -> QString {
  return QCoreApplication::translate(PUERHLAB_I18N_CONTEXT, text);
}

auto BuildSearchContainerStyle() -> QString {
  return QStringLiteral(
             "QFrame#LutBrowserSearchContainer {"
             "  background: transparent;"
             "  border: none;"
             "}");
}

auto BuildLutEntriesListStyle() -> QString {
  return AppTheme::EditorListWidgetStyle() + QStringLiteral(
                                                 "QListWidget {"
                                                 "  border-bottom-left-radius: 14px;"
                                                 "  border-bottom-right-radius: 14px;"
                                                 "}");
}

auto BuildSearchEditStyle() -> QString {
  const auto& theme = AppTheme::Instance();
  return QStringLiteral(
             "QLineEdit {"
             "  background: rgba(255, 255, 255, 0.04);"
             "  color: %1;"
             "  border: 1px solid rgba(255, 255, 255, 0.08);"
             "  border-radius: 8px;"
             "  padding: 0px 14px;"
             "  selection-background-color: %2;"
             "  selection-color: %3;"
             "}"
             "QLineEdit:hover {"
             "  background: rgba(255, 255, 255, 0.08);"
             "  border: 1px solid rgba(255, 255, 255, 0.15);"
             "}"
             "QLineEdit:focus {"
             "  background: rgba(0, 0, 0, 0.2);"
             "  border: 1px solid %4;"
             "}"
             "QLineEdit::placeholder {"
             "  color: rgba(255, 255, 255, 0.3);"
             "}")
      .arg(theme.textColor().name(QColor::HexRgb),
           theme.accentColor().name(QColor::HexRgb), 
           theme.bgCanvasColor().name(QColor::HexRgb),
           QColor(theme.accentColor().red(), theme.accentColor().green(), theme.accentColor().blue(), 200).name(QColor::HexArgb));
}

auto EntryMatchesSearchToken(const lut_catalog::LutCatalogEntry& entry, const QString& token)
    -> bool {
  if (token.isEmpty()) {
    return true;
  }

  const QString haystack =
      QStringLiteral("%1 %2 %3")
          .arg(entry.display_name_, entry.secondary_text_, QString::fromStdString(entry.path_))
          .toCaseFolded();
  return haystack.contains(token);
}

auto CompareByNameAsc(const lut_catalog::LutCatalogEntry& a, const lut_catalog::LutCatalogEntry& b)
    -> bool {
  const int name_cmp = QString::compare(a.display_name_, b.display_name_, Qt::CaseInsensitive);
  if (name_cmp != 0) {
    return name_cmp < 0;
  }

  const QString a_path      = QString::fromStdString(a.path_);
  const QString b_path      = QString::fromStdString(b.path_);
  const int     path_cmp_ci = QString::compare(a_path, b_path, Qt::CaseInsensitive);
  if (path_cmp_ci != 0) {
    return path_cmp_ci < 0;
  }
  return QString::compare(a_path, b_path, Qt::CaseSensitive) < 0;
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
    subtitle_label_->setStyleSheet(
        AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
    AppTheme::MarkFontRole(subtitle_label_, AppTheme::FontRole::UiCaption);

    text_layout->addWidget(title_label_);
    text_layout->addWidget(subtitle_label_);

    status_label_ = MakePillLabel({}, this);
    status_label_->hide();

    layout->addLayout(text_layout, 1);
    layout->addWidget(status_label_, 0, Qt::AlignTop);
  }

  void Bind(const lut_catalog::LutCatalogEntry& entry) {
    entry_            = entry;
    const auto& theme = AppTheme::Instance();
    title_label_->SetRawText(entry.display_name_);
    subtitle_label_->SetRawText(entry.secondary_text_);
    status_label_->setVisible(!entry.status_text_.isEmpty());
    status_label_->setText(entry.status_text_);

    if (entry.kind_ == lut_catalog::LutCatalogEntryKind::MissingCurrent) {
      status_label_->setStyleSheet(
          QStringLiteral("QLabel {"
                         "  color: #E5A040;" 
                         "  background: rgba(229, 160, 64, 0.12);"
                         "  border: 1px solid rgba(229, 160, 64, 0.3);"
                         "  border-radius: 4px;"
                         "  padding: 2px 8px;"
                         "}"));
    } else if (!entry.valid_) {
      status_label_->setStyleSheet(
          QStringLiteral("QLabel {"
                         "  color: #E05C5C;"
                         "  background: rgba(224, 92, 92, 0.12);"
                         "  border: 1px solid rgba(224, 92, 92, 0.3);"
                         "  border-radius: 4px;"
                         "  padding: 2px 8px;"
                         "}"));
    } else {
      status_label_->setStyleSheet(
          QStringLiteral("QLabel {"
                         "  color: %1;"
                         "  background: %2;"
                         "  border: 1px solid %3;"
                         "  border-radius: 6px;"
                         "  padding: 2px 6px;"
                         "}")
              .arg(theme.accentColor().name(QColor::HexRgb),
                   QColor(theme.accentColor().red(), theme.accentColor().green(),
                          theme.accentColor().blue(), 28)
                       .name(QColor::HexArgb),
                   QColor(theme.accentColor().red(), theme.accentColor().green(),
                          theme.accentColor().blue(), 62)
                       .name(QColor::HexArgb)));
    }
    SetSelected(false);
  }

  void SetSelected(bool selected) {
    const auto& theme = AppTheme::Instance();
    
    QString border_color = "transparent"; 
    QString background = "transparent";
    QString title_color = theme.textColor().name(QColor::HexRgb);
    QString sub_color   = theme.textMutedColor().name(QColor::HexRgb);

    if (!entry_.valid_) {
      title_color = theme.textMutedColor().name(QColor::HexRgb);
      sub_color   = theme.dividerColor().name(QColor::HexRgb); // 甚至更暗
    }

    if (selected) {
      border_color = QColor(theme.accentColor().red(), theme.accentColor().green(), theme.accentColor().blue(), 120).name(QColor::HexArgb);
      background = theme.selectedTintColor().name(QColor::HexArgb);
      title_color = theme.textColor().name(QColor::HexRgb); // 选中时恢复高亮
    }

    setStyleSheet(QStringLiteral("QFrame {"
                                 "  background: %1;"
                                 "  border: 1px solid %2;"
                                 "  border-radius: 8px;"
                                 "}"
                                 "QFrame:hover {"
                                 "  background: rgba(255, 255, 255, 0.05);"
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

  auto* search_container = new QFrame(this);
  search_container->setObjectName(QStringLiteral("LutBrowserSearchContainer"));
  search_container->setStyleSheet(BuildSearchContainerStyle());

  auto* search_layout = new QVBoxLayout(search_container);
  search_layout->setContentsMargins(0, 0, 0, 0);
  search_layout->setSpacing(6);

  search_edit_ = new QLineEdit(search_container);
  search_edit_->setClearButtonEnabled(true); 
  search_edit_->setFixedHeight(36);          
  search_edit_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  search_edit_->setStyleSheet(BuildSearchEditStyle());
  AppTheme::MarkFontRole(search_edit_, AppTheme::FontRole::UiBody);
  search_layout->addWidget(search_edit_, 0);

  auto* filter_row    = new QWidget(search_container);
  auto* filter_layout = new QHBoxLayout(filter_row);
  filter_layout->setContentsMargins(4, 6, 4, 0); 
  filter_layout->setSpacing(8);

  search_summary_label_ = new QLabel(filter_row);
  search_summary_label_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
  search_summary_label_->setStyleSheet(
      AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
  AppTheme::MarkFontRole(search_summary_label_, AppTheme::FontRole::UiCaption);
  
  filter_layout->addWidget(search_summary_label_, 0); 

  filter_layout->addStretch(1); 

  sort_label_ = new QLabel(filter_row);
  sort_label_->setStyleSheet(AppTheme::EditorLabelStyle(AppTheme::Instance().textMutedColor()));
  AppTheme::MarkFontRole(sort_label_, AppTheme::FontRole::UiCaptionStrong);
  filter_layout->addWidget(sort_label_, 0);

  sort_field_combo_ = new QComboBox(filter_row);
  sort_field_combo_->setFixedHeight(28); 
  sort_field_combo_->setStyleSheet(AppTheme::EditorComboBoxStyle());
  sort_field_combo_->addItem({}, static_cast<int>(SortField::Name));
  sort_field_combo_->addItem({}, static_cast<int>(SortField::ModifiedTime));
  AppTheme::MarkFontRole(sort_field_combo_, AppTheme::FontRole::UiBody);
  filter_layout->addWidget(sort_field_combo_, 0);

  sort_order_combo_ = new QComboBox(filter_row);
  sort_order_combo_->setFixedHeight(28);
  sort_order_combo_->setStyleSheet(AppTheme::EditorComboBoxStyle());
  sort_order_combo_->addItem({}, static_cast<int>(SortOrder::Ascending));
  sort_order_combo_->addItem({}, static_cast<int>(SortOrder::Descending));
  AppTheme::MarkFontRole(sort_order_combo_, AppTheme::FontRole::UiBody);
  filter_layout->addWidget(sort_order_combo_, 0);

  search_layout->addWidget(filter_row, 0); 
  root->addWidget(search_container, 0);

  auto* actions_row    = new QWidget(this);
  auto* actions_layout = new QHBoxLayout(actions_row);
  actions_layout->setContentsMargins(0, 0, 0, 0);
  actions_layout->setSpacing(8);

  directory_label_ = new ElidedLabel({}, actions_row);
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
  entries_list_->setStyleSheet(BuildLutEntriesListStyle());
  entries_list_->setFrameShape(QFrame::NoFrame);
  root->addWidget(entries_list_, 1);

  QObject::connect(search_edit_, &QLineEdit::textChanged, this, [this](const QString&) {
    const QString selected_path = entries_list_ && entries_list_->currentItem()
                                      ? entries_list_->currentItem()->data(Qt::UserRole).toString()
                                      : QString{};
    RebuildVisibleEntries(selected_path);
  });
  QObject::connect(sort_field_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
                   [this](int) {
                     const QString selected_path =
                         entries_list_ && entries_list_->currentItem()
                             ? entries_list_->currentItem()->data(Qt::UserRole).toString()
                             : QString{};
                     RebuildVisibleEntries(selected_path);
                   });
  QObject::connect(sort_order_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
                   [this](int) {
                     const QString selected_path =
                         entries_list_ && entries_list_->currentItem()
                             ? entries_list_->currentItem()->data(Qt::UserRole).toString()
                             : QString{};
                     RebuildVisibleEntries(selected_path);
                   });
  QObject::connect(open_folder_btn_, &QPushButton::clicked, this,
                   [this]() { emit OpenFolderRequested(); });
  QObject::connect(refresh_btn_, &QPushButton::clicked, this,
                   [this]() { emit RefreshRequested(); });
  const auto emit_activation_for_item = [this](QListWidgetItem* item) {
    if (updating_entries_ || !item) {
      return;
    }
    const int row = entries_list_->row(item);
    if (row < 0 || row >= static_cast<int>(visible_entries_.size())) {
      return;
    }
    if (!visible_entries_[static_cast<size_t>(row)].selectable_) {
      return;
    }
    emit LutPathActivated(QString::fromStdString(visible_entries_[static_cast<size_t>(row)].path_));
  };

  QObject::connect(entries_list_, &QListWidget::currentItemChanged, this,
                   [this, emit_activation_for_item](QListWidgetItem* current, QListWidgetItem*) {
                     RefreshSelectionStyles();
                     emit_activation_for_item(current);
                   });
  QObject::connect(
      entries_list_, &QListWidget::itemClicked, this,
      [emit_activation_for_item](QListWidgetItem* item) { emit_activation_for_item(item); });
  QObject::connect(
      entries_list_, &QListWidget::itemActivated, this,
      [emit_activation_for_item](QListWidgetItem* item) { emit_activation_for_item(item); });

  RetranslateUi();
}

void LutBrowserWidget::RetranslateUi() {
  title_label_->setText(Tr("Look"));
  subtitle_label_->setText(Tr("Browse LUT files and apply one to the current image."));
  if (search_edit_) {
    search_edit_->setPlaceholderText(Tr("Search LUTs by file name or metadata"));
  }
  if (sort_label_) {
    sort_label_->setText(Tr("Sort"));
  }
  if (sort_field_combo_) {
    sort_field_combo_->setItemText(0, Tr("Name"));
    sort_field_combo_->setItemText(1, Tr("Modified Time"));
  }
  if (sort_order_combo_) {
    sort_order_combo_->setItemText(0, QStringLiteral("▲"));
    sort_order_combo_->setItemText(1, QStringLiteral("▼"));
  }
  open_folder_btn_->setText(Tr("Open Folder"));
  refresh_btn_->setText(Tr("Refresh"));
  UpdateSearchResultSummary();
}

void LutBrowserWidget::SetDirectoryInfo(const QString& directory_text, const QString& status_text,
                                        bool can_open_directory) {
  directory_label_->SetRawText(directory_text);
  directory_label_->setToolTip(directory_text);
  status_label_->setText(status_text);
  open_folder_btn_->setEnabled(can_open_directory);
}

void LutBrowserWidget::SetEntries(const std::vector<lut_catalog::LutCatalogEntry>& entries,
                                  const QString&                                   selected_path) {
  source_entries_ = entries;
  RebuildVisibleEntries(selected_path);
}

auto LutBrowserWidget::SelectRelativeEntry(int step) -> bool {
  if (!entries_list_ || visible_entries_.empty() || step == 0) {
    return false;
  }

  const int direction = step > 0 ? 1 : -1;
  int       index     = entries_list_->currentRow();
  if (index < 0) {
    index = direction > 0 ? -1 : static_cast<int>(visible_entries_.size());
  }

  while (true) {
    index += direction;
    if (index < 0 || index >= static_cast<int>(visible_entries_.size())) {
      break;
    }
    if (!visible_entries_[static_cast<size_t>(index)].selectable_) {
      continue;
    }

    if (entries_list_->currentRow() == index) {
      emit LutPathActivated(
          QString::fromStdString(visible_entries_[static_cast<size_t>(index)].path_));
      return true;
    }

    entries_list_->setCurrentRow(index);
    if (auto* item = entries_list_->item(index)) {
      entries_list_->scrollToItem(item, QAbstractItemView::PositionAtCenter);
    }
    return true;
  }

  const int current = entries_list_->currentRow();
  if (current >= 0 && current < static_cast<int>(visible_entries_.size()) &&
      visible_entries_[static_cast<size_t>(current)].selectable_) {
    emit LutPathActivated(
        QString::fromStdString(visible_entries_[static_cast<size_t>(current)].path_));
    return true;
  }
  return false;
}

void LutBrowserWidget::RebuildVisibleEntries(const QString& preferred_selected_path) {
  const QString token = search_edit_ ? search_edit_->text().trimmed().toCaseFolded() : QString{};

  std::vector<lut_catalog::LutCatalogEntry> file_entries;
  std::vector<lut_catalog::LutCatalogEntry> special_entries;
  file_entries.reserve(source_entries_.size());
  special_entries.reserve(source_entries_.size());

  for (const auto& entry : source_entries_) {
    if (!EntryMatchesSearchToken(entry, token)) {
      continue;
    }
    if (entry.kind_ == lut_catalog::LutCatalogEntryKind::File) {
      file_entries.push_back(entry);
      continue;
    }
    special_entries.push_back(entry);
  }

  const SortField sort_field = CurrentSortField();
  const SortOrder sort_order = CurrentSortOrder();
  std::sort(file_entries.begin(), file_entries.end(),
            [sort_field, sort_order](const lut_catalog::LutCatalogEntry& a,
                                     const lut_catalog::LutCatalogEntry& b) {
              if (sort_field == SortField::Name) {
                const bool by_name_asc = CompareByNameAsc(a, b);
                if (CompareByNameAsc(a, b) == CompareByNameAsc(b, a)) {
                  return false;
                }
                return sort_order == SortOrder::Ascending ? by_name_asc : !by_name_asc;
              }

              if (a.has_modified_time_ != b.has_modified_time_) {
                return a.has_modified_time_;
              }
              if (a.has_modified_time_ && b.has_modified_time_ &&
                  a.modified_time_sort_key_ != b.modified_time_sort_key_) {
                if (sort_order == SortOrder::Ascending) {
                  return a.modified_time_sort_key_ < b.modified_time_sort_key_;
                }
                return a.modified_time_sort_key_ > b.modified_time_sort_key_;
              }

              const bool by_name_asc = CompareByNameAsc(a, b);
              if (CompareByNameAsc(a, b) == CompareByNameAsc(b, a)) {
                return false;
              }
              return sort_order == SortOrder::Ascending ? by_name_asc : !by_name_asc;
            });

  visible_entries_.clear();
  visible_entries_.reserve(special_entries.size() + file_entries.size());
  visible_entries_.insert(visible_entries_.end(), special_entries.begin(), special_entries.end());
  visible_entries_.insert(visible_entries_.end(), file_entries.begin(), file_entries.end());

  updating_entries_ = true;
  entries_list_->clear();

  int selected_row = -1;
  for (int i = 0; i < static_cast<int>(visible_entries_.size()); ++i) {
    const auto& entry = visible_entries_[static_cast<size_t>(i)];
    auto*       item  = new QListWidgetItem(entries_list_);
    item->setSizeHint(QSize(0, 66));
    item->setData(Qt::UserRole, QString::fromStdString(entry.path_));

    auto* row_widget = new LutEntryItemWidget(entries_list_);
    row_widget->Bind(entry);
    entries_list_->setItemWidget(item, row_widget);

    if (!preferred_selected_path.isEmpty() &&
        QString::fromStdString(entry.path_) == preferred_selected_path) {
      selected_row = i;
    }
  }

  if (selected_row < 0) {
    for (int i = 0; i < static_cast<int>(visible_entries_.size()); ++i) {
      if (visible_entries_[static_cast<size_t>(i)].selectable_) {
        selected_row = i;
        break;
      }
    }
  }
  if (selected_row >= 0) {
    entries_list_->setCurrentRow(selected_row);
  }

  RefreshSelectionStyles();
  updating_entries_ = false;
  UpdateSearchResultSummary();
}

void LutBrowserWidget::UpdateSearchResultSummary() {
  if (!search_summary_label_) {
    return;
  }

  const int total_file_count =
      static_cast<int>(std::count_if(source_entries_.begin(), source_entries_.end(),
                                     [](const lut_catalog::LutCatalogEntry& entry) {
                                       return entry.kind_ == lut_catalog::LutCatalogEntryKind::File;
                                     }));
  const int visible_file_count =
      static_cast<int>(std::count_if(visible_entries_.begin(), visible_entries_.end(),
                                     [](const lut_catalog::LutCatalogEntry& entry) {
                                       return entry.kind_ == lut_catalog::LutCatalogEntryKind::File;
                                     }));
  const bool has_search = search_edit_ && !search_edit_->text().trimmed().isEmpty();

  if (!has_search) {
    search_summary_label_->setText(Tr("LUTs %1").arg(total_file_count));
    return;
  }
  search_summary_label_->setText(Tr("Shown %1/%2").arg(visible_file_count).arg(total_file_count));
}

auto LutBrowserWidget::CurrentSortField() const -> SortField {
  if (!sort_field_combo_) {
    return SortField::Name;
  }
  if (sort_field_combo_->currentData().toInt() == static_cast<int>(SortField::ModifiedTime)) {
    return SortField::ModifiedTime;
  }
  return SortField::Name;
}

auto LutBrowserWidget::CurrentSortOrder() const -> SortOrder {
  if (!sort_order_combo_) {
    return SortOrder::Ascending;
  }
  if (sort_order_combo_->currentData().toInt() == static_cast<int>(SortOrder::Descending)) {
    return SortOrder::Descending;
  }
  return SortOrder::Ascending;
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
