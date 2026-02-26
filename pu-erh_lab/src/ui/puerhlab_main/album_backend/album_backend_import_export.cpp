void AlbumBackend::StartImport(const QStringList& fileUrlsOrPaths) {
  if (project_loading_) {
    SetTaskState("Project is loading. Please wait.", 0, false);
    return;
  }
  if (!import_service_) {
    SetTaskState("Import service is unavailable.", 0, false);
    return;
  }
  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    SetTaskState("Import already running.", task_progress_, true);
    return;
  }

  std::vector<image_path_t>        paths;
  std::unordered_set<std::wstring> seen;

  for (const QString& raw : fileUrlsOrPaths) {
    const auto pathOpt = InputToPath(raw);
    if (!pathOpt.has_value()) {
      continue;
    }

    std::error_code ec;
    if (!std::filesystem::is_regular_file(pathOpt.value(), ec) || ec) {
      continue;
    }
    if (!is_supported_file(pathOpt.value())) {
      continue;
    }

    const std::wstring key = pathOpt->wstring();
    if (!seen.insert(key).second) {
      continue;
    }
    paths.push_back(pathOpt.value());
  }

  if (paths.empty()) {
    SetTaskState("No supported files selected.", 0, false);
    return;
  }

  import_target_folder_id_   = current_folder_id_;
  import_target_folder_path_ = CurrentFolderFsPath();

  auto job            = std::make_shared<ImportJob>();
  current_import_job_ = job;

  SetTaskState(QString("Importing %1 file(s)...").arg(static_cast<int>(paths.size())), 0, true);

  QPointer<AlbumBackend> self(this);
  job->on_progress_ = [self](const ImportProgress& progress) {
    if (!self) {
      return;
    }

    const uint32_t total        = std::max<uint32_t>(progress.total_, 1);
    const uint32_t placeholders = progress.placeholders_created_.load();
    const uint32_t metadataDone = progress.metadata_done_.load();
    const uint32_t failed       = progress.failed_.load();
    const uint32_t done         = std::max(placeholders, metadataDone);
    const int      pct          = static_cast<int>((done * 100U) / total);

    QMetaObject::invokeMethod(
        self,
        [self, done, total, metadataDone, failed, pct]() {
          if (!self) {
            return;
          }
          self->SetTaskState(
              QString("Importing... %1/%2 (meta %3, failed %4)")
                  .arg(done)
                  .arg(total)
                  .arg(metadataDone)
                  .arg(failed),
              pct, true);
        },
        Qt::QueuedConnection);
  };

  job->on_finished_ = [self](const ImportResult& result) {
    if (!self) {
      return;
    }

    QMetaObject::invokeMethod(
        self,
        [self, result]() {
          if (!self) {
            return;
          }
          self->FinishImport(result);
        },
        Qt::QueuedConnection);
  };

  try {
    ImportOptions options;
    current_import_job_ =
        import_service_->ImportToFolder(paths, import_target_folder_path_, options, job);
  } catch (const std::exception& e) {
    current_import_job_.reset();
    SetTaskState(QString("Import failed: %1").arg(QString::fromUtf8(e.what())), 0, false);
  }
}

void AlbumBackend::CancelImport() {
  if (!current_import_job_) {
    return;
  }
  current_import_job_->canceled_.store(true);
  SetTaskState("Cancelling import...", task_progress_, true);
}

void AlbumBackend::StartExport(const QString& outputDirUrlOrPath) {
  StartExportWithOptionsForTargets(outputDirUrlOrPath, "JPEG", false, 4096, 95, 16, 5, "NONE",
                                   {});
}

void AlbumBackend::StartExportWithOptions(const QString& outputDirUrlOrPath,
                                          const QString& formatName, bool resizeEnabled,
                                          int maxLengthSide, int quality, int bitDepth,
                                          int pngCompressionLevel,
                                          const QString& tiffCompression) {
  StartExportWithOptionsForTargets(outputDirUrlOrPath, formatName, resizeEnabled, maxLengthSide,
                                   quality, bitDepth, pngCompressionLevel, tiffCompression, {});
}

void AlbumBackend::StartExportWithOptionsForTargets(const QString& outputDirUrlOrPath,
                                                    const QString& formatName,
                                                    bool resizeEnabled, int maxLengthSide,
                                                    int quality, int bitDepth,
                                                    int pngCompressionLevel,
                                                    const QString& tiffCompression,
                                                    const QVariantList& targetEntries) {
  if (project_loading_) {
    SetExportFailureState("Project is loading. Please wait.");
    return;
  }

  if (!export_service_ || !project_) {
    SetExportFailureState("Export service is unavailable.");
    return;
  }
  if (export_inflight_) {
    SetExportFailureState("Export already running.");
    return;
  }

  ResetExportProgressState("Preparing export queue...");

  const auto outDirOpt = InputToPath(outputDirUrlOrPath);
  if (!outDirOpt.has_value()) {
    SetExportFailureState("No export folder selected.");
    return;
  }

  std::error_code ec;
  if (!std::filesystem::exists(outDirOpt.value(), ec)) {
    std::filesystem::create_directories(outDirOpt.value(), ec);
  }
  if (ec || !std::filesystem::is_directory(outDirOpt.value(), ec) || ec) {
    SetExportFailureState("Export folder is invalid.");
    return;
  }

  const auto targets = CollectExportTargets(targetEntries);

  if (targets.empty()) {
    SetExportFailureState("No images to export.");
    return;
  }

  const ImageFormatType format        = FormatFromName(formatName);
  const int             clamped_max   = std::clamp(maxLengthSide, 256, 16384);
  const int             clamped_q     = std::clamp(quality, 1, 100);
  const auto            bit_depth     = BitDepthFromInt(bitDepth);
  const int             clamped_png   = std::clamp(pngCompressionLevel, 0, 9);
  const auto            tiff_compress = TiffCompressFromName(tiffCompression);

  export_service_->ClearAllExportTasks();
  const auto queue_result =
      BuildExportQueue(targets, outDirOpt.value(), format, resizeEnabled, clamped_max, clamped_q,
                       bit_depth, clamped_png, tiff_compress);

  if (queue_result.queued_count_ == 0) {
    export_status_ = "No export tasks were queued.";
    if (!queue_result.first_error_.isEmpty()) {
      export_error_summary_ = queue_result.first_error_;
    }
    emit ExportStateChanged();
    emit exportStateChanged();
    SetTaskState("No valid export tasks could be created.", 0, false);
    return;
  }

  export_inflight_ = true;
  export_total_    = queue_result.queued_count_;
  export_skipped_  = queue_result.skipped_count_;
  if (queue_result.skipped_count_ > 0) {
    export_status_ = QString("Exporting %1 image(s). Skipped %2 invalid item(s).")
                         .arg(queue_result.queued_count_)
                         .arg(queue_result.skipped_count_);
  } else {
    export_status_ = QString("Exporting %1 image(s)...").arg(queue_result.queued_count_);
  }
  emit ExportStateChanged();
  emit exportStateChanged();
  SetTaskState(export_status_, 0, false);

  QPointer<AlbumBackend> self(this);
  export_service_->ExportAll(
      [self](const ExportProgress& progress) {
        if (!self) {
          return;
        }
        QMetaObject::invokeMethod(
            self,
            [self, progress]() {
              if (!self) {
                return;
              }
              const int completed =
                  static_cast<int>(std::min(progress.completed_, progress.total_));
              if (completed < self->export_completed_) {
                return;
              }

              self->export_total_     = static_cast<int>(std::max<size_t>(progress.total_, 1));
              self->export_completed_ = completed;
              self->export_succeeded_ = static_cast<int>(progress.succeeded_);
              self->export_failed_    = static_cast<int>(progress.failed_);
              self->export_status_    = QString("Exporting... processed %1/%2, written %3, failed %4.")
                                          .arg(self->export_completed_)
                                          .arg(self->export_total_)
                                          .arg(self->export_succeeded_)
                                          .arg(self->export_failed_);
              emit self->ExportStateChanged();

              const int percent =
                  self->export_total_ > 0 ? (self->export_completed_ * 100) / self->export_total_ : 0;
              self->SetTaskState(self->export_status_, percent, false);
            },
            Qt::QueuedConnection);
      },
      [self, skipped = queue_result.skipped_count_](std::shared_ptr<std::vector<ExportResult>> results) {
        if (!self) {
          return;
        }

        QMetaObject::invokeMethod(
            self,
            [self, results, skipped]() {
              if (!self) {
                return;
              }
              self->FinishExport(results, skipped);
            },
            Qt::QueuedConnection);
      });
}

void AlbumBackend::ResetExportState() {
  if (export_inflight_) {
    return;
  }
  ResetExportProgressState("Ready to export.");
}

void AlbumBackend::OpenEditor(uint elementId, uint imageId) {
  if (project_loading_) {
    editor_status_ = "Project is loading. Please wait.";
    emit EditorStateChanged();
    return;
  }
  if (!pipeline_service_ || !project_ || !history_service_) {
    editor_status_ = "Editor service is unavailable.";
    emit EditorStateChanged();
    return;
  }

  const auto nextElementId = static_cast<sl_element_id_t>(elementId);
  const auto nextImageId   = static_cast<image_id_t>(imageId);
  if (nextElementId == 0 || nextImageId == 0) {
    return;
  }

  FinalizeEditorSession(true);

  try {
    auto pipeline_guard = pipeline_service_->LoadPipeline(nextElementId);
    if (!pipeline_guard || !pipeline_guard->pipeline_) {
      throw std::runtime_error("Pipeline is unavailable.");
    }

    auto history_guard = history_service_->LoadHistory(nextElementId);
    if (!history_guard || !history_guard->history_) {
      throw std::runtime_error("History is unavailable.");
    }

    editor_element_id_ = nextElementId;
    editor_image_id_   = nextImageId;

    editor_title_ = QString("Editing %1")
                        .arg(index_by_element_id_.contains(nextElementId)
                                 ? all_images_[index_by_element_id_.at(nextElementId)].file_name
                                 : QString("image #%1").arg(nextImageId));
    editor_status_ = "OpenGL editor window is active.";
    editor_active_ = true;
    editor_busy_   = false;
    emit EditorStateChanged();

    OpenEditorDialog(project_->GetImagePoolService(), pipeline_guard, history_service_, history_guard,
                     nextElementId, nextImageId, QApplication::activeWindow());

    pipeline_service_->SavePipeline(pipeline_guard);
    pipeline_service_->Sync();
    history_service_->SaveHistory(history_guard);
    history_service_->Sync();
    project_->GetImagePoolService()->SyncWithStorage();
    project_->SaveProject(meta_path_);

    if (thumbnail_service_) {
      try {
        thumbnail_service_->InvalidateThumbnail(nextElementId);
      } catch (...) {
      }
      if (IsThumbnailPinned(nextElementId)) {
        RequestThumbnail(nextElementId, nextImageId);
      } else {
        UpdateThumbnailDataUrl(nextElementId, QString());
      }
    }

    editor_status_ = "Editor closed. Changes saved.";
  } catch (const std::exception& e) {
    editor_status_ = QString("Failed to open editor: %1").arg(QString::fromUtf8(e.what()));
  }

  if (!editor_preview_url_.isEmpty()) {
    editor_preview_url_.clear();
    emit EditorPreviewChanged();
  }
  editor_active_     = false;
  editor_busy_       = false;
  editor_element_id_ = 0;
  editor_image_id_   = 0;
  editor_title_.clear();
  emit EditorStateChanged();
}

void AlbumBackend::CloseEditor() {
  FinalizeEditorSession(true);
}

void AlbumBackend::ResetEditorAdjustments() {
  if (!editor_active_) {
    return;
  }
  editor_state_     = editor_initial_state_;
  editor_lut_index_ = LutIndexForPath(editor_state_.lut_path_);
  emit EditorStateChanged();
  QueueEditorRender(RenderType::FULL_RES_PREVIEW);
}

void AlbumBackend::RequestEditorFullPreview() {
  if (!editor_active_) {
    return;
  }
  QueueEditorRender(RenderType::FULL_RES_PREVIEW);
}

void AlbumBackend::SetEditorLutIndex(int index) {
  if (!editor_active_ || index < 0 || index >= static_cast<int>(editor_lut_paths_.size())) {
    return;
  }
  if (editor_lut_index_ == index) {
    return;
  }
  editor_lut_index_       = index;
  editor_state_.lut_path_ = editor_lut_paths_[static_cast<size_t>(index)];
  emit EditorStateChanged();
  QueueEditorRender(RenderType::FAST_PREVIEW);
}

void AlbumBackend::SetEditorExposure(double value) {
  SetEditorAdjustment(editor_state_.exposure_, value, -10.0, 10.0);
}

void AlbumBackend::SetEditorContrast(double value) {
  SetEditorAdjustment(editor_state_.contrast_, value, -100.0, 100.0);
}

void AlbumBackend::SetEditorSaturation(double value) {
  SetEditorAdjustment(editor_state_.saturation_, value, -100.0, 100.0);
}

void AlbumBackend::SetEditorTint(double value) {
  SetEditorAdjustment(editor_state_.tint_, value, -100.0, 100.0);
}

void AlbumBackend::SetEditorBlacks(double value) {
  SetEditorAdjustment(editor_state_.blacks_, value, -100.0, 100.0);
}

void AlbumBackend::SetEditorWhites(double value) {
  SetEditorAdjustment(editor_state_.whites_, value, -100.0, 100.0);
}

void AlbumBackend::SetEditorShadows(double value) {
  SetEditorAdjustment(editor_state_.shadows_, value, -100.0, 100.0);
}

void AlbumBackend::SetEditorHighlights(double value) {
  SetEditorAdjustment(editor_state_.highlights_, value, -100.0, 100.0);
}

void AlbumBackend::SetEditorSharpen(double value) {
  SetEditorAdjustment(editor_state_.sharpen_, value, -100.0, 100.0);
}

void AlbumBackend::SetEditorClarity(double value) {
  SetEditorAdjustment(editor_state_.clarity_, value, -100.0, 100.0);
}

