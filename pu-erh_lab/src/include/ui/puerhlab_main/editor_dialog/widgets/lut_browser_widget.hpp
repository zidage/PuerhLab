#pragma once

#include <vector>

#include <QWidget>

#include "ui/puerhlab_main/editor_dialog/modules/lut_catalog.hpp"

class QLabel;
class QListWidget;
class QPushButton;

namespace puerhlab::ui {

class LutBrowserWidget final : public QWidget {
  Q_OBJECT

 public:
  explicit LutBrowserWidget(QWidget* parent = nullptr);

  void RetranslateUi();
  void SetDirectoryInfo(const QString& directory_text, const QString& status_text,
                        bool can_open_directory);
  void SetEntries(const std::vector<lut_catalog::LutCatalogEntry>& entries,
                  const QString& selected_path);

 signals:
  void OpenFolderRequested();
  void RefreshRequested();
  void LutPathActivated(const QString& path);

 private:
  void RefreshSelectionStyles();

  QLabel*                            title_label_     = nullptr;
  QLabel*                            subtitle_label_  = nullptr;
  QLabel*                            directory_label_ = nullptr;
  QLabel*                            status_label_    = nullptr;
  QPushButton*                       open_folder_btn_ = nullptr;
  QPushButton*                       refresh_btn_     = nullptr;
  QListWidget*                       entries_list_    = nullptr;
  std::vector<lut_catalog::LutCatalogEntry> entries_{};
  bool                               updating_entries_ = false;
};

}  // namespace puerhlab::ui
