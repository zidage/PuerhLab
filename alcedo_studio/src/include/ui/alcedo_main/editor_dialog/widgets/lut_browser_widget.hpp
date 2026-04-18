#pragma once

#include <QWidget>
#include <vector>

#include "ui/alcedo_main/editor_dialog/modules/lut_catalog.hpp"

class QLabel;
class QListWidget;
class QPushButton;
class QComboBox;
class QLineEdit;

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
                  const QString&                                   selected_path);
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

  void         RebuildVisibleEntries(const QString& preferred_selected_path);
  void         UpdateSearchResultSummary();
  auto         CurrentSortField() const -> SortField;
  auto         CurrentSortOrder() const -> SortOrder;
  void         RefreshSelectionStyles();

  QLabel*      title_label_          = nullptr;
  QLabel*      subtitle_label_       = nullptr;
  ElidedLabel* directory_label_      = nullptr;
  QLabel*      status_label_         = nullptr;
  QLabel*      sort_label_           = nullptr;
  QLabel*      search_summary_label_ = nullptr;
  QLineEdit*   search_edit_          = nullptr;
  QComboBox*   sort_field_combo_     = nullptr;
  QComboBox*   sort_order_combo_     = nullptr;
  QPushButton* open_folder_btn_      = nullptr;
  QPushButton* refresh_btn_          = nullptr;
  QListWidget* entries_list_         = nullptr;
  std::vector<lut_catalog::LutCatalogEntry> source_entries_{};
  std::vector<lut_catalog::LutCatalogEntry> visible_entries_{};
  bool                                      updating_entries_ = false;
};

}  // namespace alcedo::ui
