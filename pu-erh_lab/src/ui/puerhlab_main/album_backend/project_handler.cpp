//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/album_backend/project_handler.hpp"

#include <QMetaObject>
#include <QPointer>

#include <stdexcept>
#include <thread>

#include "app/project_package_service.hpp"
#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"

namespace puerhlab::ui {

#define PL_TEXT(text, ...)                                                                      \
  i18n::MakeLocalizedText(PUERHLAB_I18N_CONTEXT, QT_TRANSLATE_NOOP(PUERHLAB_I18N_CONTEXT, text) \
                                                     __VA_OPT__(, ) __VA_ARGS__)

ProjectHandler::ProjectHandler(AlbumBackend& backend) : backend_(backend) {}

bool ProjectHandler::InitializeServices(const std::filesystem::path& dbPath,
                                        const std::filesystem::path& metaPath,
                                        ProjectOpenMode              openMode,
                                        const std::filesystem::path& packagePath,
                                        const std::filesystem::path& workspaceDir) {
  if (project_loading_) {
    backend_.SetServiceMessageForCurrentProject(PL_TEXT("A project load is already in progress."));
    return false;
  }

  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    backend_.SetServiceMessageForCurrentProject(
        PL_TEXT("Cannot switch project while an import is running."));
    return false;
  }
  if (ie.export_inflight()) {
    backend_.SetServiceMessageForCurrentProject(
        PL_TEXT("Cannot switch project while export is running."));
    return false;
  }

  if (backend_.editor_.editor_active()) {
    backend_.editor_.FinalizeEditorSession(true);
  }

  backend_.SetServiceMessageForCurrentProject((openMode == ProjectOpenMode::kCreateNew)
                                                  ? PL_TEXT("Creating project...")
                                                  : PL_TEXT("Loading project..."));
  SetProjectLoadingState(true, (openMode == ProjectOpenMode::kCreateNew)
                                   ? PL_TEXT("Creating project...")
                                   : PL_TEXT("Loading project..."));
  backend_.SetTaskState(PL_TEXT("Opening project..."), 0, false);

  const auto request_id = ++project_load_request_id_;

  auto old_project   = project_;
  auto old_pipeline  = pipeline_service_;
  auto old_meta      = meta_path_;
  auto old_package   = project_package_path_;
  auto old_workspace = project_workspace_dir_;

  QPointer<AlbumBackend> self(&backend_);
  std::thread([self, request_id, old_project = std::move(old_project),
               old_pipeline = std::move(old_pipeline), old_meta = std::move(old_meta),
               old_package = std::move(old_package), old_workspace = std::move(old_workspace),
               dbPath, metaPath, packagePath, workspaceDir, openMode]() mutable {
    struct LoadResult {
      bool                                    success_ = false;
      QString                                 error_{};
      std::shared_ptr<ProjectService>         project_{};
      std::shared_ptr<PipelineMgmtService>    pipeline_{};
      std::shared_ptr<EditHistoryMgmtService> history_{};
      std::shared_ptr<ThumbnailService>       thumbnail_{};
      std::unique_ptr<ImportServiceImpl>      import_{};
      std::shared_ptr<ExportService>          export_{};
      std::filesystem::path                   db_path_{};
      std::filesystem::path                   meta_path_{};
      std::filesystem::path                   package_path_{};
      std::filesystem::path                   workspace_dir_{};
      std::filesystem::path                   workspace_to_cleanup_{};
    };

    auto result = std::make_shared<LoadResult>();

    try {
      if (old_pipeline) {
        old_pipeline->Sync();
      }

      if (old_project && !old_meta.empty()) {
        old_project->GetSleeveService()->Sync();
        old_project->GetImagePoolService()->SyncWithStorage();
        old_project->SaveProject(old_meta);

        if (!old_package.empty()) {
          auto package_service = old_project->GetProjectPackageService();
          if (!package_service) {
            throw std::runtime_error("Project package service is unavailable.");
          }

          QString               package_error;
          std::filesystem::path snapshot_path;
          if (!package_service->BuildTempDbSnapshotPath(&snapshot_path, &package_error) ||
              !package_service->CreateLiveDbSnapshot(old_project, snapshot_path, &package_error) ||
              !package_service->WritePackedProject(old_package, old_meta, snapshot_path,
                                                   &package_error)) {
            std::error_code ec;
            if (!snapshot_path.empty()) {
              std::filesystem::remove(snapshot_path, ec);
            }
            const QByteArray err = package_error.toUtf8();
            throw std::runtime_error(err.isEmpty() ? "Failed to pack previous project."
                                                   : err.constData());
          }

          std::error_code ec;
          std::filesystem::remove(snapshot_path, ec);
        }
      }

      result->workspace_to_cleanup_ = old_workspace;

      result->project_   = std::make_shared<ProjectService>(dbPath, metaPath, openMode);
      result->pipeline_  = std::make_shared<PipelineMgmtService>(result->project_->GetStorageService());
      result->history_   = std::make_shared<EditHistoryMgmtService>(result->project_->GetStorageService());
      result->thumbnail_ = std::make_shared<ThumbnailService>(
          result->project_->GetSleeveService(), result->project_->GetImagePoolService(),
          result->pipeline_);
      result->import_ = std::make_unique<ImportServiceImpl>(
          result->project_->GetSleeveService(), result->project_->GetImagePoolService());
      result->export_ = std::make_shared<ExportService>(result->project_->GetSleeveService(),
                                                        result->project_->GetImagePoolService(),
                                                        result->pipeline_);

      if (openMode == ProjectOpenMode::kCreateNew) {
        result->project_->GetSleeveService()->Sync();
        result->project_->GetImagePoolService()->SyncWithStorage();
        result->project_->SaveProject(metaPath);
      }

      result->db_path_   = result->project_->GetDBPath();
      result->meta_path_ = result->project_->GetMetaPath();
      if (result->meta_path_.empty()) {
        result->meta_path_ = metaPath;
      }
      result->package_path_  = packagePath;
      result->workspace_dir_ = workspaceDir;
      result->success_       = true;
    } catch (const std::exception& e) {
      result->success_ = false;
      result->error_   = QString::fromUtf8(e.what());
    } catch (...) {
      result->success_ = false;
      result->error_   = PL_TEXT("Unknown project load error.").Render();
    }

    if (!result->success_ && !workspaceDir.empty()) {
      album_util::CleanupWorkspaceDirectory(workspaceDir);
    }

    if (!self) {
      return;
    }

    QMetaObject::invokeMethod(
        self,
        [self, request_id, result]() mutable {
          if (!self || request_id != self->project_handler_.project_load_request_id_) {
            return;
          }

          auto& ph = self->project_handler_;

          if (!result->success_) {
            ph.SetProjectLoadingState(false, {});
            self->SetServiceMessageForCurrentProject(
                ph.project_ ? PL_TEXT("Requested project failed to open: %1", result->error_)
                            : PL_TEXT("Project open failed: %1", result->error_));
            self->SetTaskState(PL_TEXT("Project open failed."), 0, false);
            return;
          }

          ph.project_               = std::move(result->project_);
          ph.pipeline_service_      = std::move(result->pipeline_);
          ph.history_service_       = std::move(result->history_);
          ph.thumbnail_service_     = std::move(result->thumbnail_);
          ph.import_service_        = std::move(result->import_);
          ph.export_service_        = std::move(result->export_);
          ph.db_path_               = std::move(result->db_path_);
          ph.meta_path_             = std::move(result->meta_path_);
          ph.project_package_path_  = std::move(result->package_path_);
          ph.project_workspace_dir_ = std::move(result->workspace_dir_);

          ph.ClearProjectData();
          self->import_export_.ResetExportState();
          self->ReloadFolderTree();
          self->stats_.ClearFilters();
          self->ReloadCurrentFolder();
          emit self->StatsFilterChanged();
          self->SetTaskState(PL_TEXT("No background tasks"), 0, false);

          self->SetServiceState(
              true, ph.project_package_path_.empty()
                        ? PL_TEXT("Loaded project. DB: %1  Meta: %2",
                                  album_util::PathToQString(ph.db_path_),
                                  album_util::PathToQString(ph.meta_path_))
                        : PL_TEXT("Loaded packed project: %1 (DB temp: %2)",
                                  album_util::PathToQString(ph.project_package_path_),
                                  album_util::PathToQString(ph.db_path_)));
          emit self->ProjectChanged();
          emit self->projectChanged();
          ph.SetProjectLoadingState(false, {});

          if (!result->workspace_to_cleanup_.empty() &&
              result->workspace_to_cleanup_ != ph.project_workspace_dir_) {
            album_util::CleanupWorkspaceDirectory(result->workspace_to_cleanup_);
          }
        },
        Qt::QueuedConnection);
  }).detach();

  return true;
}

bool ProjectHandler::PersistCurrentProjectState() {
  try {
    if (pipeline_service_) {
      pipeline_service_->Sync();
    }
    if (project_) {
      project_->GetSleeveService()->Sync();
      project_->GetImagePoolService()->SyncWithStorage();
      if (!meta_path_.empty()) {
        project_->SaveProject(meta_path_);
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool ProjectHandler::PackageCurrentProjectFiles(QString* errorOut) const {
  if (!project_ || db_path_.empty() || meta_path_.empty() || project_package_path_.empty()) {
    return true;
  }

  auto package_service = project_->GetProjectPackageService();
  if (!package_service) {
    if (errorOut) {
      *errorOut = QStringLiteral("Project package service is unavailable.");
    }
    return false;
  }

  std::filesystem::path snapshot_path;
  if (!package_service->BuildTempDbSnapshotPath(&snapshot_path, errorOut)) {
    return false;
  }

  const bool snapshot_ok = package_service->CreateLiveDbSnapshot(project_, snapshot_path, errorOut);
  if (!snapshot_ok) {
    std::error_code ec;
    std::filesystem::remove(snapshot_path, ec);
    return false;
  }

  const bool packed_ok =
      package_service->WritePackedProject(project_package_path_, meta_path_, snapshot_path, errorOut);
  std::error_code ec;
  std::filesystem::remove(snapshot_path, ec);
  return packed_ok;
}

void ProjectHandler::SetProjectLoadingState(bool loading, const i18n::LocalizedText& message) {
  const i18n::LocalizedText next_message = loading ? message : i18n::LocalizedText{};
  if (project_loading_ == loading &&
      project_loading_message_text_.source_ == next_message.source_ &&
      project_loading_message_text_.args_ == next_message.args_) {
    return;
  }
  project_loading_              = loading;
  project_loading_message_text_ = next_message;
  emit backend_.ProjectLoadStateChanged();
}

void ProjectHandler::ClearProjectData() {
  backend_.thumb_.ReleaseVisibleThumbnailPins();

  backend_.view_state_.all_images_.clear();
  backend_.view_state_.visible_thumbnails_.clear();
  backend_.folder_ctrl_.ClearState();
  backend_.import_export_.ClearImportTarget();

  emit backend_.ThumbnailsChanged();
  emit backend_.thumbnailsChanged();
  emit backend_.FoldersChanged();
  emit backend_.FolderSelectionChanged();
  emit backend_.folderSelectionChanged();
  emit backend_.CountsChanged();
}

}  // namespace puerhlab::ui

#undef PL_TEXT
