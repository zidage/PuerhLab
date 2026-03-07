//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/album_backend/image_controller.hpp"

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"

#include <filesystem>
#include <stdexcept>
#include <unordered_set>

#include "sleeve/sleeve_filesystem.hpp"

namespace puerhlab::ui {

namespace {
auto ToVariantIdList(const std::vector<sl_element_id_t>& ids) -> QVariantList {
  QVariantList out;
  out.reserve(static_cast<qsizetype>(ids.size()));
  for (const auto id : ids) {
    out.push_back(static_cast<uint>(id));
  }
  return out;
}
}  // namespace

ImageController::ImageController(AlbumBackend& backend) : backend_(backend) {}

auto ImageController::CollectDeleteTargets(const QVariantList& targetEntries) const
    -> std::vector<DeleteTarget> {
  std::vector<DeleteTarget> targets;
  targets.reserve(static_cast<size_t>(targetEntries.size()));

  std::unordered_set<sl_element_id_t> seen_element_ids;
  seen_element_ids.reserve(static_cast<size_t>(targetEntries.size()) * 2 + 1);

  for (const QVariant& row_var : targetEntries) {
    const QVariantMap row      = row_var.toMap();
    const auto        element_id = static_cast<sl_element_id_t>(row.value("elementId").toUInt());
    if (element_id == 0 || !seen_element_ids.insert(element_id).second) {
      continue;
    }

    DeleteTarget target;
    target.element_id_ = element_id;
    target.image_id_   = static_cast<image_id_t>(row.value("imageId").toUInt());

    const auto item_it = backend_.index_by_element_id_.find(element_id);
    if (item_it != backend_.index_by_element_id_.end()) {
      const auto& item = backend_.all_images_[item_it->second];
      if (target.image_id_ == 0) {
        target.image_id_ = item.image_id;
      }
      target.parent_folder_id_ = item.parent_folder_id;
    }

    targets.push_back(target);
  }

  return targets;
}

void ImageController::RebuildProjectViews(sl_element_id_t preferredFolderId) {
  auto& ph       = backend_.project_handler_;
  const auto snapshot = ph.CollectProjectSnapshot();

  backend_.thumb_.ReleaseVisibleThumbnailPins();

  backend_.all_images_.clear();
  backend_.index_by_element_id_.clear();
  backend_.visible_thumbnails_.clear();
  emit backend_.ThumbnailsChanged();
  emit backend_.thumbnailsChanged();

  backend_.folder_ctrl_.SetFolderState(snapshot.folder_entries_, snapshot.folder_parent_by_id_,
                                       snapshot.folder_path_by_id_);
  backend_.folder_ctrl_.RebuildFolderView();

  for (const auto& entry : snapshot.album_entries_) {
    backend_.AddOrUpdateAlbumItem(entry.element_id_, entry.image_id_, entry.file_name_,
                                  entry.parent_folder_id_);
  }

  if (backend_.folder_ctrl_.folder_path_by_id().contains(preferredFolderId)) {
    backend_.folder_ctrl_.ApplyFolderSelection(preferredFolderId, true);
  } else {
    backend_.folder_ctrl_.ApplyFolderSelection(0, true);
  }

  backend_.stats_.RebuildThumbnailView();
  backend_.stats_.RefreshStats();
}

auto ImageController::DeleteImages(const QVariantList& targetEntries) -> QVariantMap {
  QVariantMap result{
      {"success", false},
      {"deletedCount", 0},
      {"failedCount", 0},
      {"deletedElementIds", QVariantList{}},
      {"failedElementIds", QVariantList{}},
      {"message", QString{}},
  };

  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    const QString msg = "Project is loading. Please wait.";
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg;
    return result;
  }
  if (!ph.project()) {
    const QString msg = "No project is loaded.";
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg;
    return result;
  }

  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    const QString msg = "Cannot delete images while import is running.";
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg;
    return result;
  }
  if (ie.export_inflight()) {
    const QString msg = "Cannot delete images while export is running.";
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg;
    return result;
  }

  const auto targets = CollectDeleteTargets(targetEntries);
  if (targets.empty()) {
    const QString msg = "No valid images selected for deletion.";
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg;
    return result;
  }

  std::unordered_set<sl_element_id_t> target_ids;
  target_ids.reserve(targets.size() * 2 + 1);
  for (const auto& target : targets) {
    target_ids.insert(target.element_id_);
  }

  if (backend_.editor_.editor_active() &&
      target_ids.contains(backend_.editor_.editor_element_id())) {
    backend_.editor_.FinalizeEditorSession(true);
  }

  auto proj          = ph.project();
  auto image_pool    = proj->GetImagePoolService();
  auto export_svc    = ph.export_service();
  auto pipeline_svc  = ph.pipeline_service();
  auto history_svc   = ph.history_service();
  const auto& folder_paths = backend_.folder_ctrl_.folder_path_by_id();

  std::vector<sl_element_id_t> deleted_ids;
  std::vector<sl_element_id_t> failed_ids;
  deleted_ids.reserve(targets.size());
  failed_ids.reserve(targets.size());

  bool image_pool_dirty = false;
  for (const auto& target : targets) {
    bool deleted = false;
    try {
      const auto remove_result = proj->GetSleeveService()->Write<void>(
          [target, &folder_paths](FileSystem& fs) {
            auto element = fs.Get(target.element_id_);
            if (!element || element->type_ != ElementType::FILE) {
              throw std::runtime_error("Target element is not a file.");
            }

            std::filesystem::path parent_path = album_util::RootFsPath();
            const auto parent_it = folder_paths.find(target.parent_folder_id_);
            if (parent_it != folder_paths.end()) {
              parent_path = parent_it->second;
            }

            fs.Delete(parent_path / element->element_name_);
          });
      if (!remove_result.success_) {
        throw std::runtime_error(remove_result.message_);
      }
      deleted = true;
    } catch (...) {
      deleted = false;
    }

    if (!deleted) {
      failed_ids.push_back(target.element_id_);
      continue;
    }

    deleted_ids.push_back(target.element_id_);

    try {
      backend_.thumb_.RemoveThumbnailState(target.element_id_, target.image_id_);
    } catch (...) {
    }

    if (export_svc) {
      try {
        export_svc->RemoveExportTask(target.element_id_);
      } catch (...) {
      }
    }
    if (pipeline_svc) {
      try {
        pipeline_svc->DeletePipeline(target.element_id_);
      } catch (...) {
      }
    }
    if (history_svc) {
      try {
        history_svc->DeleteHistory(target.element_id_);
      } catch (...) {
      }
    }
    if (image_pool && target.image_id_ != 0) {
      try {
        image_pool->Remove(target.image_id_);
        image_pool_dirty = true;
      } catch (...) {
      }
    }
  }

  bool save_ok = true;
  if (image_pool_dirty && image_pool) {
    try {
      image_pool->SyncWithStorage();
    } catch (...) {
      save_ok = false;
    }
  }

  if (!deleted_ids.empty()) {
    try {
      if (!ph.meta_path().empty()) {
        proj->SaveProject(ph.meta_path());
      }
      QString ignored_error;
      if (!ph.PackageCurrentProjectFiles(&ignored_error)) {
        save_ok = false;
      }
    } catch (...) {
      save_ok = false;
    }
  }

  if (!deleted_ids.empty()) {
    RebuildProjectViews(backend_.folder_ctrl_.current_folder_id());
  }

  const int deleted_count = static_cast<int>(deleted_ids.size());
  const int failed_count  = static_cast<int>(failed_ids.size());

  QString msg;
  if (deleted_count == 0) {
    msg = "No images were deleted.";
  } else if (failed_count == 0) {
    msg = QString("Deleted %1 image(s).").arg(deleted_count);
  } else {
    msg = QString("Deleted %1 image(s); %2 failed.").arg(deleted_count).arg(failed_count);
  }
  if (!save_ok) {
    msg += " Project state save failed.";
  }

  backend_.SetServiceMessageForCurrentProject(msg);
  backend_.SetTaskState(msg, deleted_count > 0 ? 100 : 0, false);
  if (deleted_count > 0) {
    backend_.ScheduleIdleTaskStateReset(1500);
  }

  result["success"]           = deleted_count > 0;
  result["deletedCount"]      = deleted_count;
  result["failedCount"]       = failed_count;
  result["deletedElementIds"] = ToVariantIdList(deleted_ids);
  result["failedElementIds"]  = ToVariantIdList(failed_ids);
  result["message"]           = msg;
  return result;
}

}  // namespace puerhlab::ui
