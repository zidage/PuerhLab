auto AlbumBackend::FieldOptions() const -> QVariantList {
  return rule_model_.FieldOptions();
}

auto AlbumBackend::FilterInfo() const -> QString {
  return FormatFilterInfo(ShownCount(), TotalCount());
}

int AlbumBackend::TotalCount() const {
  int count = 0;
  for (const auto& image : all_images_) {
    if (IsImageInCurrentFolder(image)) {
      ++count;
    }
  }
  return count;
}

void AlbumBackend::AddRule() {
  rule_model_.AddRule();
}

void AlbumBackend::RemoveRule(int index) {
  rule_model_.RemoveRule(index);
}

void AlbumBackend::SetRuleField(int index, int fieldValue) {
  rule_model_.SetField(index, fieldValue);
}

void AlbumBackend::SetRuleOp(int index, int opValue) {
  rule_model_.SetOp(index, opValue);
}

void AlbumBackend::SetRuleValue(int index, const QString& value) {
  rule_model_.SetValue(index, value);
}

void AlbumBackend::SetRuleValue2(int index, const QString& value) {
  rule_model_.SetValue2(index, value);
}

void AlbumBackend::ApplyFilters(int joinOpValue) {
  auto parsedJoin = static_cast<FilterOp>(joinOpValue);
  if (parsedJoin != FilterOp::AND && parsedJoin != FilterOp::OR) {
    parsedJoin = FilterOp::AND;
  }

  last_join_op_      = parsedJoin;

  const BuildResult result = BuildFilterNode(parsedJoin);
  if (!result.error.isEmpty()) {
    if (validation_error_ != result.error) {
      validation_error_ = result.error;
      emit ValidationErrorChanged();
    }
    return;
  }

  if (!validation_error_.isEmpty()) {
    validation_error_.clear();
    emit ValidationErrorChanged();
  }

  QString nextSql;
  if (result.node.has_value()) {
    nextSql = QString::fromStdWString(FilterSQLCompiler::Compile(result.node.value()));
  }
  if (sql_preview_ != nextSql) {
    sql_preview_ = nextSql;
    emit SqlPreviewChanged();
  }

  if (!result.node.has_value()) {
    active_filter_ids_.reset();
    RebuildThumbnailView(std::nullopt);
    return;
  }

  if (!filter_service_) {
    active_filter_ids_ = std::unordered_set<sl_element_id_t>{};
    RebuildThumbnailView(active_filter_ids_);
    return;
  }

  try {
    const auto filterId = filter_service_->CreateFilterCombo(result.node.value());
    const auto idsOpt   = filter_service_->ApplyFilterOn(filterId, current_folder_id_);
    filter_service_->RemoveFilterCombo(filterId);

    std::unordered_set<sl_element_id_t> nextIds;
    if (idsOpt.has_value()) {
      nextIds.reserve(idsOpt->size() * 2 + 1);
      for (const auto id : idsOpt.value()) {
        nextIds.insert(id);
      }
    }

    active_filter_ids_ = std::move(nextIds);
    RebuildThumbnailView(active_filter_ids_);
  } catch (const std::exception& e) {
    const QString error = QString("Filter execution failed: %1").arg(QString::fromUtf8(e.what()));
    if (validation_error_ != error) {
      validation_error_ = error;
      emit ValidationErrorChanged();
    }
    active_filter_ids_ = std::unordered_set<sl_element_id_t>{};
    RebuildThumbnailView(active_filter_ids_);
  }
}

void AlbumBackend::ClearFilters() {
  rule_model_.ClearAndReset();
  last_join_op_ = FilterOp::AND;

  if (!validation_error_.isEmpty()) {
    validation_error_.clear();
    emit ValidationErrorChanged();
  }
  if (!sql_preview_.isEmpty()) {
    sql_preview_.clear();
    emit SqlPreviewChanged();
  }

  active_filter_ids_.reset();
  RebuildThumbnailView(std::nullopt);
}

bool AlbumBackend::LoadProject(const QString& metaFileUrlOrPath) {
  if (project_loading_) {
    SetServiceMessageForCurrentProject("A project load is already in progress.");
    return false;
  }

  const auto project_path_opt = InputToPath(metaFileUrlOrPath);
  if (!project_path_opt.has_value()) {
    SetServiceMessageForCurrentProject("Select a valid project file.");
    return false;
  }

  const auto project_path = project_path_opt.value();
  std::error_code ec;
  if (!std::filesystem::is_regular_file(project_path, ec) || ec) {
    SetServiceMessageForCurrentProject("Project file was not found.");
    return false;
  }

  if (IsPackedProjectPath(project_path) || IsPackedProjectFile(project_path)) {
    const QString project_name = QFileInfo(PathToQString(project_path)).completeBaseName();
    std::filesystem::path workspace_dir;
    QString               workspace_error;
    if (!CreateProjectWorkspace(project_name, &workspace_dir, &workspace_error)) {
      SetServiceMessageForCurrentProject(
          workspace_error.isEmpty() ? "Failed to prepare project temp workspace."
                                    : workspace_error);
      return false;
    }

    std::filesystem::path unpacked_db_path;
    std::filesystem::path unpacked_meta_path;
    QString               unpack_error;
    if (!UnpackProjectToWorkspace(project_path, workspace_dir, project_name, &unpacked_db_path,
                                  &unpacked_meta_path, &unpack_error)) {
      CleanupWorkspaceDirectory(workspace_dir);
      SetServiceMessageForCurrentProject(
          unpack_error.isEmpty() ? "Failed to unpack project package." : unpack_error);
      return false;
    }

    return InitializeServices(unpacked_db_path, unpacked_meta_path, ProjectOpenMode::kLoadExisting,
                              project_path, workspace_dir);
  }

  if (!IsMetadataJsonPath(project_path)) {
    SetServiceMessageForCurrentProject(
        "Unsupported project format. Choose a .json or .puerhproj file.");
    return false;
  }

  const auto db_hint_path =
      project_path.parent_path() / (project_path.stem().wstring() + L".db");
  return InitializeServices(db_hint_path, project_path, ProjectOpenMode::kLoadExisting,
                            BuildBundlePathFromMetaPath(project_path), {});
}

bool AlbumBackend::CreateProjectInFolder(const QString& folderUrlOrPath) {
  return CreateProjectInFolderNamed(folderUrlOrPath, "album_editor_project");
}

bool AlbumBackend::CreateProjectInFolderNamed(const QString& folderUrlOrPath,
                                              const QString& projectName) {
  if (project_loading_) {
    SetServiceMessageForCurrentProject("A project load is already in progress.");
    return false;
  }

  const auto folder_path_opt = InputToPath(folderUrlOrPath);
  if (!folder_path_opt.has_value()) {
    SetServiceMessageForCurrentProject("Select a valid folder for the new project.");
    return false;
  }

  QString build_error;
  const auto packed_path_opt =
      BuildUniquePackedProjectPath(folder_path_opt.value(), projectName, &build_error);
  if (!packed_path_opt.has_value()) {
    SetServiceMessageForCurrentProject(
        build_error.isEmpty() ? "Failed to prepare project package path in selected folder."
                              : build_error);
    return false;
  }

  std::filesystem::path workspace_dir;
  QString               workspace_error;
  if (!CreateProjectWorkspace(projectName, &workspace_dir, &workspace_error)) {
    SetServiceMessageForCurrentProject(workspace_error.isEmpty()
                                           ? "Failed to prepare project temp workspace."
                                           : workspace_error);
    return false;
  }

  const auto runtime_pair = BuildRuntimeProjectPair(workspace_dir, projectName);
  const bool started =
      InitializeServices(runtime_pair.first, runtime_pair.second, ProjectOpenMode::kCreateNew,
                         packed_path_opt.value(), workspace_dir);
  if (!started) {
    CleanupWorkspaceDirectory(workspace_dir);
  }
  return started;
}

bool AlbumBackend::SaveProject() {
  if (project_loading_) {
    SetServiceMessageForCurrentProject("Please wait until project loading finishes.");
    return false;
  }

  if (!project_ || meta_path_.empty()) {
    SetServiceState(false, "No project is loaded yet.");
    SetTaskState("No project to save.", 0, false);
    return false;
  }

  if (editor_active_) {
    FinalizeEditorSession(true);
  }

  if (!PersistCurrentProjectState()) {
    SetServiceMessageForCurrentProject("Project save failed.");
    SetTaskState("Project save failed.", 0, false);
    return false;
  }

  QString package_error;
  if (!PackageCurrentProjectFiles(&package_error)) {
    SetServiceMessageForCurrentProject(package_error.isEmpty() ? "Project saved, but packing failed."
                                                               : package_error);
    SetTaskState("Project packing failed.", 0, false);
    return false;
  }

  SetServiceMessageForCurrentProject(project_package_path_.empty()
                                         ? QString("Project saved to %1")
                                               .arg(PathToQString(meta_path_))
                                         : QString("Project saved and packed to %1")
                                               .arg(PathToQString(project_package_path_)));
  SetTaskState(project_package_path_.empty() ? "Project saved." : "Project saved and packed.", 100,
               false);
  ScheduleIdleTaskStateReset(1200);
  return true;
}

auto AlbumBackend::CompareOptionsForField(int fieldValue) const -> QVariantList {
  return FilterRuleModel::CompareOptionsForField(static_cast<FilterField>(fieldValue));
}

auto AlbumBackend::PlaceholderForField(int fieldValue) const -> QString {
  return FilterRuleModel::PlaceholderForField(static_cast<FilterField>(fieldValue));
}

void AlbumBackend::SelectFolder(uint folderId) {
  if (project_loading_ || !project_) {
    return;
  }

  ApplyFolderSelection(static_cast<sl_element_id_t>(folderId), true);
  ReapplyCurrentFilters();
}

void AlbumBackend::CreateFolder(const QString& folderName) {
  if (project_loading_) {
    SetTaskState("Project is loading. Please wait.", 0, false);
    return;
  }
  if (!project_) {
    SetTaskState("No project is loaded.", 0, false);
    return;
  }
  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    SetTaskState("Cannot create folder while import is running.", 0, false);
    return;
  }
  if (export_inflight_) {
    SetTaskState("Cannot create folder while export is running.", 0, false);
    return;
  }

  const QString trimmed = folderName.trimmed();
  if (trimmed.isEmpty()) {
    SetTaskState("Folder name cannot be empty.", 0, false);
    return;
  }
  if (trimmed.contains('/') || trimmed.contains('\\')) {
    SetTaskState("Folder name cannot contain '/' or '\\'.", 0, false);
    return;
  }

  const auto parent_path = RootFsPath();
  try {
    const auto create_result = project_->GetSleeveService()->Write<std::shared_ptr<SleeveElement>>(
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

    if (!meta_path_.empty()) {
      project_->SaveProject(meta_path_);
    }

    SetServiceMessageForCurrentProject(
        QString("Created folder %1").arg(WStringToQString(folder_entry.folder_name_)));
    SetTaskState(service_message_, 100, false);
    ScheduleIdleTaskStateReset(1200);
  } catch (const std::exception& e) {
    const QString err = QString("Failed to create folder: %1").arg(QString::fromUtf8(e.what()));
    SetServiceMessageForCurrentProject(err);
    SetTaskState(err, 0, false);
  }
}

void AlbumBackend::DeleteFolder(uint folderId) {
  if (project_loading_) {
    SetTaskState("Project is loading. Please wait.", 0, false);
    return;
  }
  if (!project_) {
    SetTaskState("No project is loaded.", 0, false);
    return;
  }
  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    SetTaskState("Cannot delete folder while import is running.", 0, false);
    return;
  }
  if (export_inflight_) {
    SetTaskState("Cannot delete folder while export is running.", 0, false);
    return;
  }

  const auto folder_id = static_cast<sl_element_id_t>(folderId);
  if (folder_id == 0) {
    SetTaskState("Root folder cannot be deleted.", 0, false);
    return;
  }

  const auto path_it = folder_path_by_id_.find(folder_id);
  if (path_it == folder_path_by_id_.end()) {
    SetTaskState("Folder no longer exists.", 0, false);
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
    const auto remove_result = project_->GetSleeveService()->Write<void>(
        [target_path = path_it->second](FileSystem& fs) { fs.Delete(target_path); });
    if (!remove_result.success_) {
      throw std::runtime_error(remove_result.message_);
    }

    if (!meta_path_.empty()) {
      project_->SaveProject(meta_path_);
    }
  } catch (const std::exception& e) {
    const QString err = QString("Failed to delete folder: %1").arg(QString::fromUtf8(e.what()));
    SetServiceMessageForCurrentProject(err);
    SetTaskState(err, 0, false);
    return;
  }

  if (thumbnail_service_) {
    for (const auto& image : all_images_) {
      if (!deleted_folder_ids.contains(image.parent_folder_id)) {
        continue;
      }
      try {
        thumbnail_service_->InvalidateThumbnail(image.element_id);
        thumbnail_service_->ReleaseThumbnail(image.element_id);
      } catch (...) {
      }
    }
  }

  const auto snapshot = CollectProjectSnapshot(project_);

  ReleaseVisibleThumbnailPins();
  all_images_.clear();
  index_by_element_id_.clear();
  visible_thumbnails_.clear();
  emit ThumbnailsChanged();
  emit thumbnailsChanged();

  folder_entries_      = snapshot.folder_entries_;
  folder_parent_by_id_ = snapshot.folder_parent_by_id_;
  folder_path_by_id_   = snapshot.folder_path_by_id_;
  RebuildFolderView();

  for (const auto& entry : snapshot.album_entries_) {
    AddOrUpdateAlbumItem(entry.element_id_, entry.image_id_, entry.file_name_,
                         entry.parent_folder_id_);
  }

  if (folder_path_by_id_.contains(current_folder_id_)) {
    ApplyFolderSelection(current_folder_id_, true);
  } else {
    ApplyFolderSelection(fallback_folder, true);
  }

  active_filter_ids_.reset();
  ReapplyCurrentFilters();

  SetServiceMessageForCurrentProject("Folder deleted.");
  SetTaskState("Folder deleted.", 100, false);
  ScheduleIdleTaskStateReset(1200);
}

