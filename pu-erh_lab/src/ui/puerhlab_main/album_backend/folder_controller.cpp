//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/album_backend/folder_controller.hpp"

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace puerhlab::ui {

#define PL_TEXT(text, ...)                                                    \
  i18n::MakeLocalizedText(PUERHLAB_I18N_CONTEXT,                              \
                          QT_TRANSLATE_NOOP(PUERHLAB_I18N_CONTEXT, text)      \
                              __VA_OPT__(, ) __VA_ARGS__)

FolderController::FolderController(AlbumBackend& backend) : backend_(backend) {}

void FolderController::RebuildFolderView() {
  std::sort(folder_entries_.begin(), folder_entries_.end(),
            [](const ExistingFolderEntry& lhs, const ExistingFolderEntry& rhs) {
              if (lhs.folder_id_ == 0 || rhs.folder_id_ == 0) {
                return lhs.folder_id_ == 0;
              }
              if (lhs.folder_path_ != rhs.folder_path_) {
                return lhs.folder_path_.generic_wstring() < rhs.folder_path_.generic_wstring();
              }
              return lhs.folder_id_ < rhs.folder_id_;
            });

  QVariantList next;
  next.reserve(static_cast<qsizetype>(folder_entries_.size()));

  for (const auto& folder : folder_entries_) {
    const QString name = folder.folder_id_ == 0
                             ? PL_TEXT("Root").Render()
                             : album_util::WStringToQString(folder.folder_name_);
    next.push_back(QVariantMap{{"folderId", static_cast<uint>(folder.folder_id_)},
                               {"name", name},
                               {"depth", folder.depth_},
                               {"path", album_util::FolderPathToDisplay(folder.folder_path_)},
                               {"deletable", folder.folder_id_ != 0}});
  }

  folders_ = std::move(next);
  emit backend_.FoldersChanged();
}

void FolderController::ApplyFolderSelection(sl_element_id_t folderId, bool emitSignal) {
  sl_element_id_t next_folder_id = folderId;
  if (!folder_path_by_id_.contains(next_folder_id)) {
    next_folder_id = 0;
  }
  if (!folder_path_by_id_.contains(next_folder_id) && !folder_entries_.empty()) {
    next_folder_id = folder_entries_.front().folder_id_;
  }

  const bool id_changed = current_folder_id_ != next_folder_id;
  current_folder_id_    = next_folder_id;
  const auto path_it    = folder_path_by_id_.find(current_folder_id_);
  const QString next_path_ui =
      path_it != folder_path_by_id_.end() ? album_util::FolderPathToDisplay(path_it->second)
                                          : album_util::RootPathText();
  const bool path_changed = current_folder_path_text_ != next_path_ui;
  current_folder_path_text_ = next_path_ui;

  if (emitSignal || id_changed || path_changed) {
    emit backend_.FolderSelectionChanged();
    emit backend_.folderSelectionChanged();
  }
}

auto FolderController::CurrentFolderFsPath() const -> std::filesystem::path {
  const auto it = folder_path_by_id_.find(current_folder_id_);
  if (it == folder_path_by_id_.end()) {
    return album_util::RootFsPath();
  }
  return it->second;
}

void FolderController::SelectFolder(uint folderId) {
  auto& ph = backend_.project_handler_;
  if (ph.project_loading() || !ph.project()) {
    return;
  }

  ApplyFolderSelection(static_cast<sl_element_id_t>(folderId), true);
  backend_.stats_.ClearFilters();
  backend_.ReloadCurrentFolder();
  emit backend_.StatsFilterChanged();
}

void FolderController::CreateFolder(const QString& folderName) {
  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    backend_.SetTaskState(PL_TEXT("Project is loading. Please wait."), 0, false);
    return;
  }
  if (!ph.project()) {
    backend_.SetTaskState(PL_TEXT("No project is loaded."), 0, false);
    return;
  }
  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    backend_.SetTaskState(PL_TEXT("Cannot create folder while import is running."), 0, false);
    return;
  }
  if (ie.export_inflight()) {
    backend_.SetTaskState(PL_TEXT("Cannot create folder while export is running."), 0, false);
    return;
  }

  const QString trimmed = folderName.trimmed();
  if (trimmed.isEmpty()) {
    backend_.SetTaskState(PL_TEXT("Folder name cannot be empty."), 0, false);
    return;
  }
  if (trimmed.contains('/') || trimmed.contains('\\')) {
    backend_.SetTaskState(PL_TEXT("Folder name cannot contain '/' or '\\'."), 0, false);
    return;
  }

  auto browse = ph.project()->GetAlbumBrowseService();
  if (!browse) {
    backend_.SetTaskState(PL_TEXT("Folder service is unavailable."), 0, false);
    return;
  }

  const auto created = browse->CreateFolder(current_folder_id_, trimmed.toStdWString());
  if (!created.has_value()) {
    backend_.SetTaskState(PL_TEXT("Failed to create folder."), 0, false);
    return;
  }

  bool save_ok = true;
  try {
    if (!ph.meta_path().empty()) {
      ph.project()->SaveProject(ph.meta_path());
    }
    QString ignored_error;
    if (!ph.PackageCurrentProjectFiles(&ignored_error)) {
      save_ok = false;
    }
  } catch (...) {
    save_ok = false;
  }

  backend_.ReloadFolderTree();

  auto msg = PL_TEXT("Created folder %1", album_util::WStringToQString(created->folder_name_));
  if (!save_ok) {
    msg = PL_TEXT("%1 Project state save failed.", msg.Render());
  }
  backend_.SetServiceMessageForCurrentProject(msg);
  backend_.SetTaskState(msg, 100, false);
  backend_.ScheduleIdleTaskStateReset(1200);
}

void FolderController::DeleteFolder(uint folderId) {
  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    backend_.SetTaskState(PL_TEXT("Project is loading. Please wait."), 0, false);
    return;
  }
  if (!ph.project()) {
    backend_.SetTaskState(PL_TEXT("No project is loaded."), 0, false);
    return;
  }
  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    backend_.SetTaskState(PL_TEXT("Cannot delete folder while import is running."), 0, false);
    return;
  }
  if (ie.export_inflight()) {
    backend_.SetTaskState(PL_TEXT("Cannot delete folder while export is running."), 0, false);
    return;
  }

  const auto folder_id = static_cast<sl_element_id_t>(folderId);
  if (folder_id == 0) {
    backend_.SetTaskState(PL_TEXT("Root folder cannot be deleted."), 0, false);
    return;
  }

  const auto parent_it_before = folder_parent_by_id_.find(folder_id);
  const sl_element_id_t fallback_folder =
      parent_it_before != folder_parent_by_id_.end() ? parent_it_before->second
                                                     : static_cast<sl_element_id_t>(0);

  auto browse = ph.project()->GetAlbumBrowseService();
  if (!browse || !browse->DeleteFolder(folder_id)) {
    backend_.SetTaskState(PL_TEXT("Failed to delete folder."), 0, false);
    return;
  }

  bool save_ok = true;
  try {
    if (!ph.meta_path().empty()) {
      ph.project()->SaveProject(ph.meta_path());
    }
    QString ignored_error;
    if (!ph.PackageCurrentProjectFiles(&ignored_error)) {
      save_ok = false;
    }
  } catch (...) {
    save_ok = false;
  }

  backend_.ReloadFolderTree();
  backend_.folder_ctrl_.ApplyFolderSelection(fallback_folder, true);
  backend_.stats_.ClearFilters();
  backend_.ReloadCurrentFolder();
  emit backend_.StatsFilterChanged();

  auto msg = PL_TEXT("Folder deleted.");
  if (!save_ok) {
    msg = PL_TEXT("%1 Project state save failed.", msg.Render());
  }
  backend_.SetServiceMessageForCurrentProject(msg);
  backend_.SetTaskState(msg, 100, false);
  backend_.ScheduleIdleTaskStateReset(1200);
}

void FolderController::SetFolderState(
    std::vector<ExistingFolderEntry> entries,
    std::unordered_map<sl_element_id_t, sl_element_id_t> parents,
    std::unordered_map<sl_element_id_t, std::filesystem::path> paths) {
  folder_entries_      = std::move(entries);
  folder_parent_by_id_ = std::move(parents);
  folder_path_by_id_   = std::move(paths);
}

void FolderController::ClearState() {
  folder_entries_.clear();
  folder_parent_by_id_.clear();
  folder_path_by_id_.clear();
  folders_.clear();
  current_folder_id_        = 0;
  current_folder_path_text_ = album_util::RootPathText();
}

}  // namespace puerhlab::ui

#undef PL_TEXT
