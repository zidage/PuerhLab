bool AlbumBackend::InitializeServices(const std::filesystem::path& dbPath,
                                      const std::filesystem::path& metaPath,
                                      ProjectOpenMode              openMode,
                                      const std::filesystem::path& packagePath,
                                      const std::filesystem::path& workspaceDir) {
  if (project_loading_) {
    SetServiceMessageForCurrentProject("A project load is already in progress.");
    return false;
  }

  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    SetServiceMessageForCurrentProject("Cannot switch project while an import is running.");
    return false;
  }
  if (export_inflight_) {
    SetServiceMessageForCurrentProject("Cannot switch project while export is running.");
    return false;
  }

  if (editor_active_) {
    FinalizeEditorSession(true);
  }

  SetServiceMessageForCurrentProject((openMode == ProjectOpenMode::kCreateNew)
                                         ? "Creating project..."
                                         : "Loading project...");
  SetProjectLoadingState(true, service_message_);
  SetTaskState("Opening project...", 0, false);

  const auto request_id = ++project_load_request_id_;

  auto old_project  = project_;
  auto old_pipeline = pipeline_service_;
  auto old_meta     = meta_path_;
  auto old_package  = project_package_path_;
  auto old_workspace = project_workspace_dir_;

  QPointer<AlbumBackend> self(this);
  std::thread([self, request_id, old_project = std::move(old_project),
               old_pipeline = std::move(old_pipeline), old_meta = std::move(old_meta),
               old_package = std::move(old_package), old_workspace = std::move(old_workspace),
               dbPath, metaPath, packagePath, workspaceDir, openMode]() mutable {
    struct LoadResult {
      bool                                 success_ = false;
      QString                              error_{};
      std::shared_ptr<ProjectService>      project_{};
      std::shared_ptr<PipelineMgmtService> pipeline_{};
      std::shared_ptr<EditHistoryMgmtService> history_{};
      std::shared_ptr<ThumbnailService>    thumbnail_{};
      std::unique_ptr<SleeveFilterService> filter_{};
      std::unique_ptr<ImportServiceImpl>   import_{};
      std::shared_ptr<ExportService>       export_{};
      std::filesystem::path                db_path_{};
      std::filesystem::path                meta_path_{};
      std::filesystem::path                package_path_{};
      std::filesystem::path                workspace_dir_{};
      std::filesystem::path                workspace_to_cleanup_{};
      std::vector<ExistingAlbumEntry>      album_entries_{};
      std::vector<ExistingFolderEntry>     folder_entries_{};
      std::unordered_map<sl_element_id_t, sl_element_id_t> folder_parent_by_id_{};
      std::unordered_map<sl_element_id_t, std::filesystem::path> folder_path_by_id_{};
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
          QString package_error;
          std::filesystem::path snapshot_path;
          if (!BuildTempDbSnapshotPath(&snapshot_path, &package_error) ||
              !CreateLiveDbSnapshot(old_project, snapshot_path, &package_error) ||
              !WritePackedProject(old_package, old_meta, snapshot_path, &package_error)) {
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

      result->project_ = std::make_shared<ProjectService>(dbPath, metaPath, openMode);
      result->pipeline_ =
          std::make_shared<PipelineMgmtService>(result->project_->GetStorageService());
      result->history_ =
          std::make_shared<EditHistoryMgmtService>(result->project_->GetStorageService());
      result->thumbnail_ = std::make_shared<ThumbnailService>(
          result->project_->GetSleeveService(), result->project_->GetImagePoolService(),
          result->pipeline_);
      result->filter_ =
          std::make_unique<SleeveFilterService>(result->project_->GetStorageService());
      result->import_ = std::make_unique<ImportServiceImpl>(result->project_->GetSleeveService(),
                                                            result->project_->GetImagePoolService());
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
      result->package_path_ = packagePath;
      result->workspace_dir_ = workspaceDir;

      if (self) {
        auto snapshot              = self->CollectProjectSnapshot(result->project_);
        result->album_entries_     = std::move(snapshot.album_entries_);
        result->folder_entries_    = std::move(snapshot.folder_entries_);
        result->folder_parent_by_id_ = std::move(snapshot.folder_parent_by_id_);
        result->folder_path_by_id_ = std::move(snapshot.folder_path_by_id_);
      }
      result->success_ = true;
    } catch (const std::exception& e) {
      result->success_ = false;
      result->error_   = QString::fromUtf8(e.what());
    } catch (...) {
      result->success_ = false;
      result->error_   = "Unknown project load error.";
    }

    if (!result->success_ && !workspaceDir.empty()) {
      CleanupWorkspaceDirectory(workspaceDir);
    }

    if (!self) {
      return;
    }

    QMetaObject::invokeMethod(
        self,
        [self, request_id, result]() mutable {
          if (!self || request_id != self->project_load_request_id_) {
            return;
          }

          if (!result->success_) {
            self->SetProjectLoadingState(false, QString());
            self->SetServiceMessageForCurrentProject(
                self->project_
                    ? QString("Requested project failed to open: %1").arg(result->error_)
                    : QString("Project open failed: %1").arg(result->error_));
            self->SetTaskState("Project open failed.", 0, false);
            return;
          }

          self->project_           = std::move(result->project_);
          self->pipeline_service_  = std::move(result->pipeline_);
          self->history_service_   = std::move(result->history_);
          self->thumbnail_service_ = std::move(result->thumbnail_);
          self->filter_service_    = std::move(result->filter_);
          self->import_service_    = std::move(result->import_);
          self->export_service_    = std::move(result->export_);
          self->db_path_           = std::move(result->db_path_);
          self->meta_path_         = std::move(result->meta_path_);
          self->project_package_path_ = std::move(result->package_path_);
          self->project_workspace_dir_ = std::move(result->workspace_dir_);

          self->ClearProjectData();
          self->ResetExportState();
          self->pending_project_entries_      = std::move(result->album_entries_);
          self->pending_folder_entries_       = std::move(result->folder_entries_);
          self->pending_folder_parent_by_id_  = std::move(result->folder_parent_by_id_);
          self->pending_folder_path_by_id_    = std::move(result->folder_path_by_id_);
          self->pending_project_entry_index_  = 0;
          self->ApplyLoadedProjectEntriesBatch();

          if (!result->workspace_to_cleanup_.empty() &&
              result->workspace_to_cleanup_ != self->project_workspace_dir_) {
            CleanupWorkspaceDirectory(result->workspace_to_cleanup_);
          }
        },
        Qt::QueuedConnection);
  }).detach();

  return true;
}

bool AlbumBackend::PersistCurrentProjectState() {
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

bool AlbumBackend::PackageCurrentProjectFiles(QString* errorOut) const {
  if (!project_ || db_path_.empty() || meta_path_.empty() || project_package_path_.empty()) {
    return true;
  }

  std::filesystem::path snapshot_path;
  if (!BuildTempDbSnapshotPath(&snapshot_path, errorOut)) {
    return false;
  }

  const bool snapshot_ok = CreateLiveDbSnapshot(project_, snapshot_path, errorOut);
  if (!snapshot_ok) {
    std::error_code ec;
    std::filesystem::remove(snapshot_path, ec);
    return false;
  }

  const bool packed_ok =
      WritePackedProject(project_package_path_, meta_path_, snapshot_path, errorOut);
  std::error_code ec;
  std::filesystem::remove(snapshot_path, ec);
  return packed_ok;
}

auto AlbumBackend::CollectProjectSnapshot(const std::shared_ptr<ProjectService>& project) const
    -> ProjectSnapshot {
  ProjectSnapshot snapshot;
  if (!project) {
    return snapshot;
  }

  const auto sleeve_service = project->GetSleeveService();
  if (!sleeve_service) {
    return snapshot;
  }

  try {
    snapshot = sleeve_service->Read<ProjectSnapshot>(
        [](FileSystem& fs) -> ProjectSnapshot {
          ProjectSnapshot local_snapshot;

          struct FolderVisit {
            sl_element_id_t      folder_id_ = 0;
            sl_element_id_t      parent_id_ = 0;
            std::filesystem::path folder_path_{};
            int                  depth_     = 0;
          };

          std::shared_ptr<SleeveElement> root_element;
          const auto root_path = RootFsPath();
          try {
            root_element = fs.Get(root_path, false);
          } catch (...) {
            root_element.reset();
          }

          if (!root_element || root_element->type_ != ElementType::FOLDER ||
              root_element->sync_flag_ == SyncFlag::DELETED) {
            return local_snapshot;
          }

          const auto root_id = root_element->element_id_;
          std::vector<FolderVisit>        stack{{root_id, root_id, root_path, 0}};
          std::unordered_set<sl_element_id_t> visited;
          visited.reserve(4096);

          while (!stack.empty()) {
            const auto visit = stack.back();
            stack.pop_back();

            if (!visited.insert(visit.folder_id_).second) {
              continue;
            }

            std::shared_ptr<SleeveElement> folder_element;
            try {
              folder_element = fs.Get(visit.folder_path_, false);
            } catch (...) {
              try {
                folder_element = fs.Get(visit.folder_id_);
              } catch (...) {
                continue;
              }
            }

            if (!folder_element || folder_element->sync_flag_ == SyncFlag::DELETED ||
                folder_element->type_ != ElementType::FOLDER) {
              continue;
            }

            ExistingFolderEntry folder_entry;
            folder_entry.folder_id_ = visit.folder_id_;
            if (visit.folder_id_ == root_id) {
              folder_entry.parent_id_   = root_id;
              folder_entry.folder_name_ = L"";
              folder_entry.folder_path_ = root_path;
              folder_entry.depth_       = 0;
            } else {
              folder_entry.parent_id_   = visit.parent_id_;
              folder_entry.folder_name_ = folder_element->element_name_;
              folder_entry.folder_path_ = visit.folder_path_;
              folder_entry.depth_       = visit.depth_;
            }
            local_snapshot.folder_entries_.push_back(folder_entry);
            local_snapshot.folder_parent_by_id_[folder_entry.folder_id_] = folder_entry.parent_id_;
            local_snapshot.folder_path_by_id_[folder_entry.folder_id_]   = folder_entry.folder_path_;

            std::vector<sl_element_id_t> children;
            try {
              children = fs.ListFolderContent(visit.folder_id_);
            } catch (...) {
              continue;
            }

            std::vector<std::shared_ptr<SleeveElement>> child_elements;
            child_elements.reserve(children.size());
            for (const auto child_id : children) {
              if (child_id == visit.folder_id_) {
                continue;
              }
              try {
                auto child = fs.Get(child_id);  // Subfolders expose child ids; resolve by id.
                if (!child || child->sync_flag_ == SyncFlag::DELETED) {
                  continue;
                }
                child_elements.push_back(std::move(child));
              } catch (...) {
              }
            }

            std::sort(child_elements.begin(), child_elements.end(),
                      [](const std::shared_ptr<SleeveElement>& lhs,
                         const std::shared_ptr<SleeveElement>& rhs) {
                        if (lhs->type_ != rhs->type_) {
                          return lhs->type_ == ElementType::FOLDER;
                        }
                        return lhs->element_name_ < rhs->element_name_;
                      });

            for (auto it = child_elements.rbegin(); it != child_elements.rend(); ++it) {
              const auto& child = *it;
              if (child->type_ == ElementType::FOLDER) {
                stack.push_back({child->element_id_, visit.folder_id_,
                                 visit.folder_path_ / child->element_name_, visit.depth_ + 1});
                continue;
              }

              std::shared_ptr<SleeveElement> file_element = child;
              if (visit.folder_id_ == root_id) {
                try {
                  file_element = fs.Get(visit.folder_path_ / child->element_name_,
                                        false);  // Root file access by path.
                } catch (...) {
                  continue;
                }
              }

              const auto file = std::dynamic_pointer_cast<SleeveFile>(file_element);
              if (!file || file->image_id_ == 0) {
                continue;
              }
              local_snapshot.album_entries_.push_back(
                  {file->element_id_, visit.folder_id_, file->image_id_, file->element_name_});
            }
          }

          return local_snapshot;
        });
  } catch (...) {
  }

  std::sort(snapshot.album_entries_.begin(), snapshot.album_entries_.end(),
            [](const ExistingAlbumEntry& lhs, const ExistingAlbumEntry& rhs) {
              if (lhs.file_name_ != rhs.file_name_) {
                return lhs.file_name_ < rhs.file_name_;
              }
              return lhs.element_id_ < rhs.element_id_;
            });

  std::sort(snapshot.folder_entries_.begin(), snapshot.folder_entries_.end(),
            [](const ExistingFolderEntry& lhs, const ExistingFolderEntry& rhs) {
              if (lhs.folder_id_ == 0 || rhs.folder_id_ == 0) {
                return lhs.folder_id_ == 0;
              }
              if (lhs.folder_path_ != rhs.folder_path_) {
                return lhs.folder_path_.generic_wstring() < rhs.folder_path_.generic_wstring();
              }
              return lhs.folder_id_ < rhs.folder_id_;
            });

  return snapshot;
}

void AlbumBackend::ApplyLoadedProjectEntriesBatch() {
  if (!project_loading_) {
    return;
  }

  const size_t total = pending_project_entries_.size();
  if (total == 0 || pending_project_entry_index_ >= total) {
    pending_project_entries_.clear();
    pending_project_entry_index_ = 0;
    folder_entries_     = std::move(pending_folder_entries_);
    folder_parent_by_id_ = std::move(pending_folder_parent_by_id_);
    folder_path_by_id_   = std::move(pending_folder_path_by_id_);
    RebuildFolderView();
    ApplyFolderSelection(0, true);
    RebuildThumbnailView(std::nullopt);
    SetTaskState("No background tasks", 0, false);

    SetServiceState(true, project_package_path_.empty()
                              ? QString("Loaded project. DB: %1  Meta: %2")
                                    .arg(PathToQString(db_path_))
                                    .arg(PathToQString(meta_path_))
                              : QString("Loaded packed project: %1 (DB temp: %2)")
                                    .arg(PathToQString(project_package_path_))
                                    .arg(PathToQString(db_path_)));
    emit ProjectChanged();
    emit projectChanged();
    SetProjectLoadingState(false, QString());
    return;
  }

  constexpr size_t kBatchSize = 24;
  const size_t     end_index  = std::min(total, pending_project_entry_index_ + kBatchSize);

  for (; pending_project_entry_index_ < end_index; ++pending_project_entry_index_) {
    const auto& entry = pending_project_entries_[pending_project_entry_index_];
    AddOrUpdateAlbumItem(entry.element_id_, entry.image_id_, entry.file_name_,
                         entry.parent_folder_id_);
  }

  const int pct =
      total == 0 ? 0 : static_cast<int>((pending_project_entry_index_ * 100ULL) / total);
  SetTaskState(
      QString("Loading album... %1/%2").arg(static_cast<int>(pending_project_entry_index_)).arg(
          static_cast<int>(total)),
      pct, false);
  SetProjectLoadingState(
      true, QString("Loading album... %1/%2")
                .arg(static_cast<int>(pending_project_entry_index_))
                .arg(static_cast<int>(total)));

  QTimer::singleShot(0, this, [this]() { ApplyLoadedProjectEntriesBatch(); });
}

void AlbumBackend::SetProjectLoadingState(bool loading, const QString& message) {
  const QString next_message = loading ? message : QString();
  if (project_loading_ == loading && project_loading_message_ == next_message) {
    return;
  }
  project_loading_         = loading;
  project_loading_message_ = next_message;
  emit ProjectLoadStateChanged();
}

void AlbumBackend::ClearProjectData() {
  ReleaseVisibleThumbnailPins();

  all_images_.clear();
  index_by_element_id_.clear();
  visible_thumbnails_.clear();
  folder_entries_.clear();
  folder_parent_by_id_.clear();
  folder_path_by_id_.clear();
  folders_.clear();
  current_folder_id_        = 0;
  current_folder_path_text_ = RootPathText();
  active_filter_ids_.reset();
  pending_project_entries_.clear();
  pending_folder_entries_.clear();
  pending_folder_parent_by_id_.clear();
  pending_folder_path_by_id_.clear();
  pending_project_entry_index_ = 0;
  import_target_folder_id_     = 0;
  import_target_folder_path_.clear();

  rule_model_.ClearAndReset();
  last_join_op_ = FilterOp::AND;

  if (!sql_preview_.isEmpty()) {
    sql_preview_.clear();
    emit SqlPreviewChanged();
  }
  if (!validation_error_.isEmpty()) {
    validation_error_.clear();
    emit ValidationErrorChanged();
  }

  emit ThumbnailsChanged();
  emit thumbnailsChanged();
  emit FoldersChanged();
  emit FolderSelectionChanged();
  emit folderSelectionChanged();
  emit CountsChanged();
}

