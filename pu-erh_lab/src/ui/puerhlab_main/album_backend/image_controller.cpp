//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/album_backend/image_controller.hpp"

#include "ui/puerhlab_main/album_backend/album_backend.hpp"

#include <algorithm>
#include <unordered_set>

namespace puerhlab::ui {

#define PL_TEXT(text, ...)                                                    \
  i18n::MakeLocalizedText(PUERHLAB_I18N_CONTEXT,                              \
                          QT_TRANSLATE_NOOP(PUERHLAB_I18N_CONTEXT, text)      \
                              __VA_OPT__(, ) __VA_ARGS__)

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
    const QVariantMap row = row_var.toMap();
    const auto element_id = static_cast<sl_element_id_t>(row.value("elementId").toUInt());
    if (element_id == 0 || !seen_element_ids.insert(element_id).second) {
      continue;
    }

    DeleteTarget target;
    target.element_id_ = element_id;
    target.image_id_   = static_cast<image_id_t>(row.value("imageId").toUInt());

    if (const auto* item = backend_.FindAlbumItem(element_id); item) {
      if (target.image_id_ == 0) {
        target.image_id_ = item->image_id;
      }
      target.file_path_ = item->file_path_;
    }

    targets.push_back(target);
  }

  return targets;
}

auto ImageController::DeleteImages(const QVariantList& targetEntries) -> QVariantMap {
  QVariantMap result{{"success", false},
                     {"deletedCount", 0},
                     {"failedCount", 0},
                     {"deletedElementIds", QVariantList{}},
                     {"failedElementIds", QVariantList{}},
                     {"message", QString{}}};

  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    const auto msg = PL_TEXT("Project is loading. Please wait.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }
  if (!ph.project()) {
    const auto msg = PL_TEXT("No project is loaded.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }

  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    const auto msg = PL_TEXT("Cannot delete images while import is running.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }
  if (ie.export_inflight()) {
    const auto msg = PL_TEXT("Cannot delete images while export is running.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }

  const auto targets = CollectDeleteTargets(targetEntries);
  if (targets.empty()) {
    const auto msg = PL_TEXT("No valid images selected for deletion.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }

  std::unordered_set<sl_element_id_t> target_ids;
  target_ids.reserve(targets.size() * 2 + 1);
  std::vector<std::filesystem::path> delete_paths;
  delete_paths.reserve(targets.size());
  for (const auto& target : targets) {
    target_ids.insert(target.element_id_);
    if (!target.file_path_.empty()) {
      delete_paths.push_back(target.file_path_);
    }
  }

  if (backend_.editor_.editor_active() &&
      target_ids.contains(backend_.editor_.editor_element_id())) {
    backend_.editor_.FinalizeEditorSession(true);
  }

  auto proj       = ph.project();
  auto browse     = proj->GetAlbumBrowseService();
  auto image_pool = proj->GetImagePoolService();
  auto export_svc = ph.export_service();
  auto pipeline_svc = ph.pipeline_service();
  auto history_svc = ph.history_service();

  if (!browse) {
    const auto msg = PL_TEXT("Image service is unavailable.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }

  const auto delete_result = browse->DeleteFiles(delete_paths);
  std::vector<sl_element_id_t> deleted_ids;
  deleted_ids.reserve(delete_result.deleted_files_.size());
  for (const auto& file : delete_result.deleted_files_) {
    deleted_ids.push_back(file.element_id_);
  }

  std::vector<sl_element_id_t> failed_ids;
  failed_ids.reserve(targets.size());
  for (const auto& target : targets) {
    if (target.file_path_.empty()) {
      failed_ids.push_back(target.element_id_);
    }
  }
  for (const auto& path : delete_result.failed_paths_) {
    const auto it = std::find_if(targets.begin(), targets.end(), [&path](const DeleteTarget& target) {
      return target.file_path_.lexically_normal() == path.lexically_normal();
    });
    if (it != targets.end()) {
      failed_ids.push_back(it->element_id_);
    }
  }

  bool image_pool_dirty = false;
  for (const auto& target : targets) {
    if (std::find(deleted_ids.begin(), deleted_ids.end(), target.element_id_) ==
        deleted_ids.end()) {
      continue;
    }

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
    backend_.ReloadCurrentFolder();
  }

  const int deleted_count = static_cast<int>(deleted_ids.size());
  const int failed_count  = static_cast<int>(failed_ids.size());

  auto msg = i18n::LocalizedText{};
  if (deleted_count == 0) {
    msg = PL_TEXT("No images were deleted.");
  } else if (failed_count == 0) {
    msg = PL_TEXT("Deleted %1 image(s).", deleted_count);
  } else {
    msg = PL_TEXT("Deleted %1 image(s); %2 failed.", deleted_count, failed_count);
  }
  if (!save_ok) {
    msg = PL_TEXT("%1 Project state save failed.", msg.Render());
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
  result["message"]           = msg.Render();
  return result;
}

}  // namespace puerhlab::ui

#undef PL_TEXT
