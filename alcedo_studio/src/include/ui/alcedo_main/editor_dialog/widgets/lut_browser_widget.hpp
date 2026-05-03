#pragma once

#include <QWidget>
#include <vector>

#include "ui/alcedo_main/editor_dialog/modules/lut_catalog.hpp"

class QAction;
class QLabel;
class QListWidget;
class QLineEdit;
class QToolButton;

namespace alcedo::ui {

class ElidedLabel;

class LutBrowserWidget final : public QWidget {
  Q_OBJECT

 public:
  explicit LutBrowserWidget(QWidget* parent = nullptr);

  void RetranslateUi();
  void SetDirectoryInfo(const QString& directory_text, const QString& status_text,
                        bool can_open_directory);
  void SetEntries(const std::vector<lut_catalog::LutCatalogEntry>& entries,
                  const QString& selected_path, bool preserve_scroll_position = false);
  auto SelectRelativeEntry(int step) -> bool;

 signals:
  void OpenFolderRequested();
  void RefreshRequested();
  void LutPathActivated(const QString& path);

 private:
  enum class SortField {
    Name,
    ModifiedTime,
  };

  enum class SortOrder {
    Ascending,
    Descending,
  };

  void         RebuildVisibleEntries(const QString& preferred_selected_path,
                                     bool           preserve_scroll_position = false);
  void         UpdateSearchResultSummary();
  auto         CurrentSortField() const -> SortField;
  auto         CurrentSortOrder() const -> SortOrder;
  void         RefreshSelectionStyles();

  QLabel*      title_label_          = nullptr;
  ElidedLabel* directory_label_      = nullptr;
  QLabel*      status_label_         = nullptr;
  QLabel*      search_summary_label_ = nullptr;
  QLineEdit*   search_edit_          = nullptr;
  QToolButton* sort_btn_             = nullptr;
  QToolButton* folder_btn_           = nullptr;
  QAction*     sort_field_name_action_    = nullptr;
  QAction*     sort_field_time_action_    = nullptr;
  QAction*     sort_order_asc_action_     = nullptr;
  QAction*     sort_order_desc_action_    = nullptr;
  QAction*     refresh_action_            = nullptr;
  QListWidget* entries_list_         = nullptr;
  std::vector<lut_catalog::LutCatalogEntry> source_entries_{};
  std::vector<lut_catalog::LutCatalogEntry> visible_entries_{};
  bool                                      updating_entries_ = false;
};

}  // namespace alcedo::ui
