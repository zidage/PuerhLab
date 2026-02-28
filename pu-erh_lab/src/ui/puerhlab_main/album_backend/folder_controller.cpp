#include "ui/puerhlab_main/album_backend/folder_controller.hpp"

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filesystem.hpp"

namespace puerhlab::ui {

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
                             ? "Root"
                             : album_util::WStringToQString(folder.folder_name_);
    next.push_back(QVariantMap{
        {"folderId", static_cast<uint>(folder.folder_id_)},
        {"name", name},
        {"depth", folder.depth_},
        {"path", album_util::FolderPathToDisplay(folder.folder_path_)},
        {"deletable", folder.folder_id_ != 0},
    });
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
      path_it != folder_path_by_id_.end()
          ? album_util::FolderPathToDisplay(path_it->second)
          : album_util::RootPathText();
  const bool path_changed     = current_folder_path_text_ != next_path_ui;
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
  backend_.stats_.RebuildThumbnailView();
  backend_.stats_.RefreshStats();
}

void FolderController::CreateFolder(const QString& folderName) {
  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    backend_.SetTaskState("Project is loading. Please wait.", 0, false);
    return;
  }
  if (!ph.project()) {
    backend_.SetTaskState("No project is loaded.", 0, false);
    return;
  }
  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    backend_.SetTaskState("Cannot create folder while import is running.", 0, false);
    return;
  }
  if (ie.export_inflight()) {
    backend_.SetTaskState("Cannot create folder while export is running.", 0, false);
    return;
  }

  const QString trimmed = folderName.trimmed();
  if (trimmed.isEmpty()) {
    backend_.SetTaskState("Folder name cannot be empty.", 0, false);
    return;
  }
  if (trimmed.contains('/') || trimmed.contains('\\')) {
    backend_.SetTaskState("Folder name cannot contain '/' or '\\'.", 0, false);
    return;
  }

  const auto parent_path = album_util::RootFsPath();
  try {
    const auto create_result =
        ph.project()->GetSleeveService()->Write<std::shared_ptr<SleeveElement>>(
            [parent_path, trimmed](FileSystem& fs) {
              return fs.Create(parent_path, trimmed.toStdWString(), ElementType::FOLDER);
            });
    const auto created = create_result.first;
    if (!create_result.second.success_ || !created || created->type_ != ElementType::FOLDER) {
      throw std::runtime_error("Failed to create folder.");
    }

    ExistingFolderEntry folder_entry;
    folder_entry.folder_id_   = created->element_id_;
    folder_entry.parent_id_   = 0;
    folder_entry.folder_name_ = created->element_name_;
    folder_entry.folder_path_ = parent_path / created->element_name_;
    folder_entry.depth_       = 1;

    folder_entries_.push_back(folder_entry);
    folder_parent_by_id_[folder_entry.folder_id_] = folder_entry.parent_id_;
    folder_path_by_id_[folder_entry.folder_id_]   = folder_entry.folder_path_;
    RebuildFolderView();

    if (!ph.meta_path().empty()) {
      ph.project()->SaveProject(ph.meta_path());
    }

    backend_.SetServiceMessageForCurrentProject(
        QString("Created folder %1")
            .arg(album_util::WStringToQString(folder_entry.folder_name_)));
    backend_.SetTaskState(backend_.service_message_, 100, false);
    backend_.ScheduleIdleTaskStateReset(1200);
  } catch (const std::exception& e) {
    const QString err =
        QString("Failed to create folder: %1").arg(QString::fromUtf8(e.what()));
    backend_.SetServiceMessageForCurrentProject(err);
    backend_.SetTaskState(err, 0, false);
  }
}

void FolderController::DeleteFolder(uint folderId) {
  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    backend_.SetTaskState("Project is loading. Please wait.", 0, false);
    return;
  }
  if (!ph.project()) {
    backend_.SetTaskState("No project is loaded.", 0, false);
    return;
  }
  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    backend_.SetTaskState("Cannot delete folder while import is running.", 0, false);
    return;
  }
  if (ie.export_inflight()) {
    backend_.SetTaskState("Cannot delete folder while export is running.", 0, false);
    return;
  }

  const auto folder_id = static_cast<sl_element_id_t>(folderId);
  if (folder_id == 0) {
    backend_.SetTaskState("Root folder cannot be deleted.", 0, false);
    return;
  }

  const auto path_it = folder_path_by_id_.find(folder_id);
  if (path_it == folder_path_by_id_.end()) {
    backend_.SetTaskState("Folder no longer exists.", 0, false);
    return;
  }
  const auto parent_it_before = folder_parent_by_id_.find(folder_id);
  const sl_element_id_t fallback_folder =
      parent_it_before != folder_parent_by_id_.end() ? parent_it_before->second
                                                     : static_cast<sl_element_id_t>(0);

  std::unordered_set<sl_element_id_t> deleted_folder_ids;
  deleted_folder_ids.insert(folder_id);
  bool expanded = true;
  while (expanded) {
    expanded = false;
    for (const auto& [candidate_id, parent_id] : folder_parent_by_id_) {
      if (!deleted_folder_ids.contains(parent_id) || deleted_folder_ids.contains(candidate_id)) {
        continue;
      }
      deleted_folder_ids.insert(candidate_id);
      expanded = true;
    }
  }

  try {
    const auto remove_result = ph.project()->GetSleeveService()->Write<void>(
        [target_path = path_it->second](FileSystem& fs) { fs.Delete(target_path); });
    if (!remove_result.success_) {
      throw std::runtime_error(remove_result.message_);
    }

    if (!ph.meta_path().empty()) {
      ph.project()->SaveProject(ph.meta_path());
    }
  } catch (const std::exception& e) {
    const QString err =
        QString("Failed to delete folder: %1").arg(QString::fromUtf8(e.what()));
    backend_.SetServiceMessageForCurrentProject(err);
    backend_.SetTaskState(err, 0, false);
    return;
  }

  auto thumb_svc = ph.thumbnail_service();
  if (thumb_svc) {
    for (const auto& image : backend_.all_images_) {
      if (!deleted_folder_ids.contains(image.parent_folder_id)) {
        continue;
      }
      try {
        thumb_svc->InvalidateThumbnail(image.element_id);
        thumb_svc->ReleaseThumbnail(image.element_id);
      } catch (...) {
      }
    }
  }

  const auto snapshot = ph.CollectProjectSnapshot();

  backend_.thumb_.ReleaseVisibleThumbnailPins();
  backend_.all_images_.clear();
  backend_.index_by_element_id_.clear();
  backend_.visible_thumbnails_.clear();
  emit backend_.ThumbnailsChanged();
  emit backend_.thumbnailsChanged();

  SetFolderState(snapshot.folder_entries_, snapshot.folder_parent_by_id_,
                 snapshot.folder_path_by_id_);
  RebuildFolderView();

  for (const auto& entry : snapshot.album_entries_) {
    backend_.AddOrUpdateAlbumItem(entry.element_id_, entry.image_id_, entry.file_name_,
                                  entry.parent_folder_id_);
  }

  if (folder_path_by_id_.contains(current_folder_id_)) {
    ApplyFolderSelection(current_folder_id_, true);
  } else {
    ApplyFolderSelection(fallback_folder, true);
  }

  backend_.stats_.RebuildThumbnailView();
  backend_.stats_.RefreshStats();

  backend_.SetServiceMessageForCurrentProject("Folder deleted.");
  backend_.SetTaskState("Folder deleted.", 100, false);
  backend_.ScheduleIdleTaskStateReset(1200);
}

void FolderController::SetFolderState(
    std::vector<ExistingFolderEntry>                          entries,
    std::unordered_map<sl_element_id_t, sl_element_id_t>     parents,
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
