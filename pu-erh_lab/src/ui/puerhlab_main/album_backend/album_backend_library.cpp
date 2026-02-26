// This file is intentionally include-injected by album_backend.cpp.
// It shares that translation unit's includes and anonymous namespace helpers.

void AlbumBackend::SetThumbnailVisible(uint elementId, uint imageId, bool visible) {
  const auto id       = static_cast<sl_element_id_t>(elementId);
  const auto image_id = static_cast<image_id_t>(imageId);
  if (id == 0 || image_id == 0) {
    return;
  }

  if (visible) {
    if (!thumbnail_service_) {
      return;
    }
    auto& ref = thumbnail_pin_ref_counts_[id];
    ref++;
    if (ref == 1) {
      RequestThumbnail(id, image_id);
    }
    return;
  }

  const auto it = thumbnail_pin_ref_counts_.find(id);
  if (it == thumbnail_pin_ref_counts_.end()) {
    return;
  }

  if (it->second > 1) {
    it->second--;
    return;
  }

  thumbnail_pin_ref_counts_.erase(it);
  UpdateThumbnailDataUrl(id, QString());
  if (thumbnail_service_) {
    try {
      thumbnail_service_->ReleaseThumbnail(id);
    } catch (...) {
    }
  }
}

void AlbumBackend::RebuildFolderView() {
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
    const QString name =
        folder.folder_id_ == 0 ? "Root" : WStringToQString(folder.folder_name_);
    next.push_back(QVariantMap{
        {"folderId", static_cast<uint>(folder.folder_id_)},
        {"name", name},
        {"depth", folder.depth_},
        {"path", FolderPathToDisplay(folder.folder_path_)},
        {"deletable", folder.folder_id_ != 0},
    });
  }

  folders_ = std::move(next);
  emit FoldersChanged();
}

void AlbumBackend::ApplyFolderSelection(sl_element_id_t folderId, bool emitSignal) {
  sl_element_id_t next_folder_id = folderId;
  if (!folder_path_by_id_.contains(next_folder_id)) {
    next_folder_id = 0;
  }
  if (!folder_path_by_id_.contains(next_folder_id) && !folder_entries_.empty()) {
    next_folder_id = folder_entries_.front().folder_id_;
  }

  const bool    id_changed   = current_folder_id_ != next_folder_id;
  current_folder_id_         = next_folder_id;
  const auto    path_it      = folder_path_by_id_.find(current_folder_id_);
  const QString next_path_ui =
      path_it != folder_path_by_id_.end() ? FolderPathToDisplay(path_it->second)
                                          : RootPathText();
  const bool path_changed = current_folder_path_text_ != next_path_ui;
  current_folder_path_text_ = next_path_ui;

  if (emitSignal || id_changed || path_changed) {
    emit FolderSelectionChanged();
    emit folderSelectionChanged();
  }
}

auto AlbumBackend::CurrentFolderFsPath() const -> std::filesystem::path {
  const auto it = folder_path_by_id_.find(current_folder_id_);
  if (it == folder_path_by_id_.end()) {
    return RootFsPath();
  }
  return it->second;
}

void AlbumBackend::ReleaseVisibleThumbnailPins() {
  if (thumbnail_pin_ref_counts_.empty()) {
    return;
  }

  for (const auto& [id, _] : thumbnail_pin_ref_counts_) {
    const auto index_it = index_by_element_id_.find(id);
    if (index_it != index_by_element_id_.end()) {
      all_images_[index_it->second].thumb_data_url.clear();
    }
    if (thumbnail_service_) {
      try {
        thumbnail_service_->ReleaseThumbnail(id);
      } catch (...) {
      }
    }
  }
  thumbnail_pin_ref_counts_.clear();
}

void AlbumBackend::RebuildThumbnailView(
    const std::optional<std::unordered_set<sl_element_id_t>>& allowedElementIds) {
  ReleaseVisibleThumbnailPins();

  QVariantList next;
  next.reserve(static_cast<qsizetype>(all_images_.size()));

  int index = 0;
  for (const AlbumItem& image : all_images_) {
    if (!IsImageInCurrentFolder(image)) {
      continue;
    }
    if (allowedElementIds.has_value() && !allowedElementIds->contains(image.element_id)) {
      continue;
    }
    next.push_back(MakeThumbMap(image, index++));
  }

  visible_thumbnails_ = std::move(next);
  emit ThumbnailsChanged();
  emit thumbnailsChanged();
  emit CountsChanged();
}

void AlbumBackend::AddImportedEntries(const ImportLogSnapshot& snapshot) {
  std::unordered_set<image_id_t> metadataOk;
  metadataOk.reserve(snapshot.metadata_ok_.size() * 2 + 1);
  for (const auto id : snapshot.metadata_ok_) {
    metadataOk.insert(id);
  }

  for (const auto& created : snapshot.created_) {
    if (!metadataOk.empty() && !metadataOk.contains(created.image_id_)) {
      continue;
    }
    AddOrUpdateAlbumItem(created.element_id_, created.image_id_, created.file_name_,
                         import_target_folder_id_);
  }
}

void AlbumBackend::AddOrUpdateAlbumItem(sl_element_id_t elementId, image_id_t imageId,
                                        const file_name_t& fallbackName,
                                        sl_element_id_t parentFolderId) {
  AlbumItem* item = nullptr;

  if (const auto it = index_by_element_id_.find(elementId); it != index_by_element_id_.end()) {
    item = &all_images_[it->second];
  } else {
    AlbumItem next;
    next.element_id   = elementId;
    next.image_id     = imageId;
    next.file_name    = WStringToQString(fallbackName);
    next.extension    = ExtensionFromFileName(next.file_name);
    next.accent = AccentForIndex(all_images_.size());

    all_images_.push_back(std::move(next));
    index_by_element_id_[elementId] = all_images_.size() - 1;
    item = &all_images_.back();
  }

  if (!item) {
    return;
  }

  item->element_id = elementId;
  item->image_id   = imageId;
  item->parent_folder_id = parentFolderId;

  if (project_) {
    try {
      const auto infoOpt = project_->GetSleeveService()->Read<std::optional<std::pair<QString, QDate>>>(
          [elementId, parentFolderId, fallbackName](
              FileSystem& fs) -> std::optional<std::pair<QString, QDate>> {
            std::shared_ptr<SleeveElement> element;
            if (parentFolderId == 0) {
              const auto root_file_path = RootFsPath() / fallbackName;
              try {
                element = fs.Get(root_file_path, false);
              } catch (...) {
                element.reset();
              }
            }
            if (!element) {
              element = fs.Get(elementId);  // Subfolder entries resolve by id.
            }

            if (!element || element->type_ != ElementType::FILE) {
              return std::nullopt;
            }
            return std::make_pair(WStringToQString(element->element_name_),
                                  DateFromTimeT(element->added_time_));
          });

      if (infoOpt.has_value()) {
        if (!infoOpt->first.isEmpty()) {
          item->file_name = infoOpt->first;
        }
        if (infoOpt->second.isValid()) {
          item->import_date = infoOpt->second;
        }
      }
    } catch (...) {
    }

    try {
      project_->GetImagePoolService()->Read<void>(
          imageId,
          [item](std::shared_ptr<Image> image) {
            if (!image) {
              return;
            }

            if (!image->image_name_.empty()) {
              item->file_name = WStringToQString(image->image_name_);
            }
            if (!image->image_path_.empty()) {
              item->extension = ExtensionUpper(image->image_path_);
            }

            const auto& exif = image->exif_display_;
            item->camera_model = QString::fromUtf8(exif.model_.c_str());
            item->iso          = static_cast<int>(exif.iso_);
            item->aperture     = static_cast<double>(exif.aperture_);
            item->focal_length = static_cast<double>(exif.focal_);
            item->rating       = exif.rating_;
            const QDate captureDate = DateFromExifString(exif.date_time_str_);
            if (captureDate.isValid()) {
              item->capture_date = captureDate;
            }
          });
    } catch (...) {
    }
  }

  if (!item->import_date.isValid()) {
    item->import_date = QDate::currentDate();
  }
  if (item->extension.isEmpty()) {
    item->extension = ExtensionFromFileName(item->file_name);
  }
}

void AlbumBackend::RequestThumbnail(sl_element_id_t elementId, image_id_t imageId) {
  if (!thumbnail_service_) {
    return;
  }

  auto                 service = thumbnail_service_;
  QPointer<AlbumBackend> self(this);

  CallbackDispatcher dispatcher = [](std::function<void()> fn) {
    auto* app = QCoreApplication::instance();
    if (!app) {
      fn();
      return;
    }
    QMetaObject::invokeMethod(app, std::move(fn), Qt::QueuedConnection);
  };

  service->GetThumbnail(
      elementId, imageId,
      [self, service, elementId](std::shared_ptr<ThumbnailGuard> guard) {
        if (!guard || !guard->thumbnail_buffer_) {
          if (self && !self->IsThumbnailPinned(elementId) && service) {
            try {
              service->ReleaseThumbnail(elementId);
            } catch (...) {
            }
          }
          return;
        }
        if (!self) {
          try {
            if (service) {
              service->ReleaseThumbnail(elementId);
            }
          } catch (...) {
          }
          return;
        }

        std::thread([self, service, elementId, guard = std::move(guard)]() mutable {
          QString dataUrl;
          try {
            auto* buffer = guard->thumbnail_buffer_.get();
            if (buffer) {
              if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
                buffer->SyncToCPU();
              }
              if (buffer->cpu_data_valid_) {
                QImage image = MatRgba32fToQImageCopy(buffer->GetCPUData());
                if (!image.isNull()) {
                  QImage scaled = image.scaled(220, 160, Qt::KeepAspectRatio,
                                               Qt::SmoothTransformation);
                  dataUrl = DataUrlFromImage(scaled);
                }
              }
            }
          } catch (...) {
          }

          if (self) {
            QMetaObject::invokeMethod(
                self,
                [self, service, elementId, dataUrl]() {
                  if (!self) {
                    return;
                  }
                  const bool pinned = self->IsThumbnailPinned(elementId);
                  if (pinned) {
                    self->UpdateThumbnailDataUrl(elementId, dataUrl);
                  } else {
                    self->UpdateThumbnailDataUrl(elementId, QString());
                  }
                  if (!pinned && service) {
                    try {
                      service->ReleaseThumbnail(elementId);
                    } catch (...) {
                    }
                  }
                },
                Qt::QueuedConnection);
          }
        }).detach();
      },
      true, dispatcher);
}

void AlbumBackend::UpdateThumbnailDataUrl(sl_element_id_t elementId, const QString& dataUrl) {
  const auto it = index_by_element_id_.find(elementId);
  if (it == index_by_element_id_.end()) {
    return;
  }

  auto& item = all_images_[it->second];
  if (item.thumb_data_url == dataUrl) {
    return;
  }

  item.thumb_data_url = dataUrl;

  for (qsizetype i = 0; i < visible_thumbnails_.size(); ++i) {
    QVariantMap row = visible_thumbnails_.at(i).toMap();
    if (static_cast<sl_element_id_t>(row.value("elementId").toUInt()) != elementId) {
      continue;
    }
    row.insert("thumbUrl", dataUrl);
    visible_thumbnails_[i] = row;
    break;
  }

  emit ThumbnailUpdated(static_cast<uint>(elementId), dataUrl);
  emit thumbnailUpdated(static_cast<uint>(elementId), dataUrl);
}

bool AlbumBackend::IsThumbnailPinned(sl_element_id_t elementId) const {
  const auto it = thumbnail_pin_ref_counts_.find(elementId);
  return it != thumbnail_pin_ref_counts_.end() && it->second > 0;
}

void AlbumBackend::FinishImport(const ImportResult& result) {
  const auto importJob = current_import_job_;
  current_import_job_.reset();

  if (!importJob || !importJob->import_log_) {
    SetTaskState("Import finished but no log snapshot is available.", 0, false);
    return;
  }

  const auto snapshot = importJob->import_log_->Snapshot();

  bool state_saved = true;
  try {
    if (import_service_) {
      import_service_->SyncImports(snapshot, import_target_folder_path_);
    }
    if (project_) {
      project_->GetSleeveService()->Sync();
      project_->GetImagePoolService()->SyncWithStorage();
      project_->SaveProject(meta_path_);
    }
  } catch (...) {
    state_saved = false;
  }

  QString package_error;
  bool    package_saved = true;
  if (state_saved) {
    package_saved = PackageCurrentProjectFiles(&package_error);
  }

  AddImportedEntries(snapshot);
  ReapplyCurrentFilters();

  import_target_folder_id_   = current_folder_id_;
  import_target_folder_path_ = CurrentFolderFsPath();

  QString task_text =
      QString("Import complete: %1 imported, %2 failed").arg(result.imported_).arg(result.failed_);
  if (!state_saved) {
    task_text += " (project sync/save failed)";
    SetServiceMessageForCurrentProject("Import finished, but saving project state failed.");
  } else if (!package_saved) {
    task_text += " (project packing failed)";
    SetServiceMessageForCurrentProject(
        package_error.isEmpty() ? "Import finished, but project packing failed."
                                : package_error);
  }
  SetTaskState(task_text, 100, false);
  ScheduleIdleTaskStateReset(1800);
}

void AlbumBackend::FinishExport(const std::shared_ptr<std::vector<ExportResult>>& results,
                                int skippedCount) {
  export_inflight_ = false;

  int         ok   = 0;
  int         fail = 0;
  QStringList errors;
  if (results) {
    for (const auto& r : *results) {
      if (r.success_) {
        ++ok;
      } else {
        ++fail;
        if (!r.message_.empty() && errors.size() < 8) {
          errors << QString::fromUtf8(r.message_.c_str());
        }
      }
    }
  }

  const int total  = ok + fail;
  export_total_    = std::max(export_total_, total);
  export_completed_ = total;
  export_succeeded_ = ok;
  export_failed_    = fail;
  export_skipped_   = skippedCount;
  export_error_summary_.clear();
  if (!errors.isEmpty()) {
    export_error_summary_ = errors.join('\n');
  }

  export_status_ = QString("Export complete. Written %1/%2 image(s), failed %3.")
                       .arg(ok)
                       .arg(total)
                       .arg(fail);
  if (skippedCount > 0) {
    export_status_ += QString(" Skipped %1 invalid item(s).").arg(skippedCount);
  }
  emit ExportStateChanged();
  emit exportStateChanged();

  SetTaskState(QString("Export complete: %1 ok, %2 failed").arg(ok).arg(fail), 100, false);
  ScheduleIdleTaskStateReset(1800);
}

void AlbumBackend::ReapplyCurrentFilters() {
  ApplyFilters(static_cast<int>(last_join_op_));
  if (!validation_error_.isEmpty()) {
    RebuildThumbnailView(active_filter_ids_);
  }
}

void AlbumBackend::SetServiceState(bool ready, const QString& message) {
  if (service_ready_ == ready && service_message_ == message) {
    return;
  }
  service_ready_   = ready;
  service_message_ = message;
  emit ServiceStateChanged();
}

void AlbumBackend::SetServiceMessageForCurrentProject(const QString& message) {
  SetServiceState(project_ != nullptr, message);
}

void AlbumBackend::ScheduleIdleTaskStateReset(int delayMs) {
  QTimer::singleShot(std::max(delayMs, 0), this, [this]() {
    if (!export_inflight_ && !task_cancel_visible_) {
      SetTaskState("No background tasks", 0, false);
    }
  });
}

void AlbumBackend::SetExportFailureState(const QString& message) {
  export_status_ = message;
  emit ExportStateChanged();
  emit exportStateChanged();
  SetTaskState(message, 0, false);
}

void AlbumBackend::ResetExportProgressState(const QString& status) {
  export_status_        = status;
  export_error_summary_.clear();
  export_total_         = 0;
  export_completed_     = 0;
  export_succeeded_     = 0;
  export_failed_        = 0;
  export_skipped_       = 0;
  emit ExportStateChanged();
  emit exportStateChanged();
}

auto AlbumBackend::CollectExportTargets(const QVariantList& targetEntries) const
    -> std::vector<ExportTarget> {
  const QVariantList& source = targetEntries.empty() ? visible_thumbnails_ : targetEntries;
  std::vector<ExportTarget> targets;
  targets.reserve(static_cast<size_t>(source.size()));

  std::unordered_set<uint64_t> dedupe;
  dedupe.reserve(static_cast<size_t>(source.size()) * 2 + 1);

  for (const QVariant& entry : source) {
    const auto map       = entry.toMap();
    const auto elementId = static_cast<sl_element_id_t>(map.value("elementId").toUInt());
    const auto imageId   = static_cast<image_id_t>(map.value("imageId").toUInt());
    if (elementId == 0 || imageId == 0) {
      continue;
    }

    if (!dedupe.insert(ExportTargetKey(elementId, imageId)).second) {
      continue;
    }
    targets.emplace_back(elementId, imageId);
  }
  return targets;
}

auto AlbumBackend::BuildExportQueue(const std::vector<ExportTarget>& targets,
                                    const std::filesystem::path&   outputDir,
                                    ImageFormatType                format,
                                    bool                           resizeEnabled,
                                    int                            maxLengthSide,
                                    int                            quality,
                                    ExportFormatOptions::BIT_DEPTH bitDepth,
                                    int                            pngCompressionLevel,
                                    ExportFormatOptions::TIFF_COMPRESS tiffCompression)
    -> ExportQueueBuildResult {
  ExportQueueBuildResult summary;
  if (!project_ || !export_service_) {
    summary.first_error_ = "Export service is unavailable.";
    return summary;
  }

  for (const auto& [elementId, imageId] : targets) {
    try {
      const auto srcPath = project_->GetImagePoolService()->Read<std::filesystem::path>(
          imageId,
          [](const std::shared_ptr<Image>& image) { return image ? image->image_path_ : image_path_t{}; });
      if (srcPath.empty()) {
        ++summary.skipped_count_;
        if (summary.first_error_.isEmpty()) {
          summary.first_error_ = "Image source path is empty.";
        }
        continue;
      }

      ExportTask task;
      task.sleeve_id_                  = elementId;
      task.image_id_                   = imageId;
      task.options_.format_            = format;
      task.options_.resize_enabled_    = resizeEnabled;
      task.options_.max_length_side_   = resizeEnabled ? maxLengthSide : 0;
      task.options_.quality_           = quality;
      task.options_.bit_depth_         = bitDepth;
      task.options_.compression_level_ = pngCompressionLevel;
      task.options_.tiff_compress_     = tiffCompression;
      task.options_.export_path_ =
          ExportPathForOptions(srcPath, outputDir, elementId, imageId, format);

      export_service_->EnqueueExportTask(task);
      ++summary.queued_count_;
    } catch (const std::exception& e) {
      ++summary.skipped_count_;
      if (summary.first_error_.isEmpty()) {
        summary.first_error_ = QString::fromUtf8(e.what());
      }
    } catch (...) {
      ++summary.skipped_count_;
      if (summary.first_error_.isEmpty()) {
        summary.first_error_ = "Unknown error while preparing export task.";
      }
    }
  }

  return summary;
}

