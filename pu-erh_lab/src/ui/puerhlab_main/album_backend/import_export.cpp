#include "ui/puerhlab_main/album_backend/import_export.hpp"

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"

#include <QCoreApplication>
#include <QMetaObject>
#include <QPointer>
#include <QStandardPaths>

#include <algorithm>
#include <thread>
#include <unordered_set>

namespace puerhlab::ui {

using namespace album_util;

ImportExportHandler::ImportExportHandler(AlbumBackend& backend) : backend_(backend) {
  const QString pictures =
      QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
  if (!pictures.isEmpty()) {
    default_export_folder_ = pictures;
  }
}

void ImportExportHandler::StartImport(const QStringList& fileUrlsOrPaths) {
  if (backend_.project_handler_.project_loading()) {
    backend_.SetTaskState("Project is loading. Please wait.", 0, false);
    return;
  }
  auto* isvc = backend_.project_handler_.import_service();
  if (!isvc) {
    backend_.SetTaskState("Import service is unavailable.", 0, false);
    return;
  }
  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    backend_.SetTaskState("Import already running.", backend_.task_progress_, true);
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
    backend_.SetTaskState("No supported files selected.", 0, false);
    return;
  }

  import_target_folder_id_   = backend_.folder_ctrl_.current_folder_id();
  import_target_folder_path_ = backend_.folder_ctrl_.CurrentFolderFsPath();

  auto job            = std::make_shared<ImportJob>();
  current_import_job_ = job;

  backend_.SetTaskState(
      QString("Importing %1 file(s)...").arg(static_cast<int>(paths.size())), 0, true);

  QPointer<AlbumBackend> self(&backend_);
  job->on_progress_ = [self](const ImportProgress& progress) {
    if (!self) return;
    const uint32_t total        = std::max<uint32_t>(progress.total_, 1);
    const uint32_t placeholders = progress.placeholders_created_.load();
    const uint32_t metadataDone = progress.metadata_done_.load();
    const uint32_t failed       = progress.failed_.load();
    const uint32_t done         = std::max(placeholders, metadataDone);
    const int      pct          = static_cast<int>((done * 100U) / total);

    QMetaObject::invokeMethod(
        self,
        [self, done, total, metadataDone, failed, pct]() {
          if (!self) return;
          self->SetTaskState(
              QString("Importing... %1/%2 (meta %3, failed %4)")
                  .arg(done).arg(total).arg(metadataDone).arg(failed),
              pct, true);
        },
        Qt::QueuedConnection);
  };

  job->on_finished_ = [self](const ImportResult& result) {
    if (!self) return;
    QMetaObject::invokeMethod(
        self,
        [self, result]() {
          if (!self) return;
          self->import_export_.FinishImport(result);
        },
        Qt::QueuedConnection);
  };

  try {
    ImportOptions options;
    current_import_job_ =
        isvc->ImportToFolder(paths, import_target_folder_path_, options, job);
  } catch (const std::exception& e) {
    current_import_job_.reset();
    backend_.SetTaskState(
        QString("Import failed: %1").arg(QString::fromUtf8(e.what())), 0, false);
  }
}

void ImportExportHandler::CancelImport() {
  if (!current_import_job_) return;
  current_import_job_->canceled_.store(true);
  backend_.SetTaskState("Cancelling import...", backend_.task_progress_, true);
}

void ImportExportHandler::StartExport(const QString& outputDirUrlOrPath) {
  StartExportWithOptionsForTargets(outputDirUrlOrPath, "JPEG", false, 4096, 95, 16, 5, "NONE",
                                   {});
}

void ImportExportHandler::StartExportWithOptions(const QString& outputDirUrlOrPath,
                                                  const QString& formatName,
                                                  bool resizeEnabled, int maxLengthSide,
                                                  int quality, int bitDepth,
                                                  int pngCompressionLevel,
                                                  const QString& tiffCompression) {
  StartExportWithOptionsForTargets(outputDirUrlOrPath, formatName, resizeEnabled, maxLengthSide,
                                   quality, bitDepth, pngCompressionLevel, tiffCompression, {});
}

void ImportExportHandler::StartExportWithOptionsForTargets(
    const QString& outputDirUrlOrPath, const QString& formatName, bool resizeEnabled,
    int maxLengthSide, int quality, int bitDepth, int pngCompressionLevel,
    const QString& tiffCompression, const QVariantList& targetEntries) {
  if (backend_.project_handler_.project_loading()) {
    SetExportFailureState("Project is loading. Please wait.");
    return;
  }

  const auto& esvc = backend_.project_handler_.export_service();
  auto  proj = backend_.project_handler_.project();
  if (!esvc || !proj) {
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

  esvc->ClearAllExportTasks();
  const auto queue_result =
      BuildExportQueue(targets, outDirOpt.value(), format, resizeEnabled, clamped_max,
                       clamped_q, bit_depth, clamped_png, tiff_compress);

  if (queue_result.queued_count_ == 0) {
    export_status_ = "No export tasks were queued.";
    if (!queue_result.first_error_.isEmpty()) {
      export_error_summary_ = queue_result.first_error_;
    }
    emit backend_.ExportStateChanged();
    emit backend_.exportStateChanged();
    backend_.SetTaskState("No valid export tasks could be created.", 0, false);
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
  emit backend_.ExportStateChanged();
  emit backend_.exportStateChanged();
  backend_.SetTaskState(export_status_, 0, false);

  QPointer<AlbumBackend> self(&backend_);
  esvc->ExportAll(
      [self](const ExportProgress& progress) {
        if (!self) return;
        QMetaObject::invokeMethod(
            self,
            [self, progress]() {
              if (!self) return;
              auto& ie = self->import_export_;
              const int completed =
                  static_cast<int>(std::min(progress.completed_, progress.total_));
              if (completed < ie.export_completed_) return;

              ie.export_total_     = static_cast<int>(std::max<size_t>(progress.total_, 1));
              ie.export_completed_ = completed;
              ie.export_succeeded_ = static_cast<int>(progress.succeeded_);
              ie.export_failed_    = static_cast<int>(progress.failed_);
              ie.export_status_    =
                  QString("Exporting... processed %1/%2, written %3, failed %4.")
                      .arg(ie.export_completed_)
                      .arg(ie.export_total_)
                      .arg(ie.export_succeeded_)
                      .arg(ie.export_failed_);
              emit self->ExportStateChanged();

              const int percent =
                  ie.export_total_ > 0
                      ? (ie.export_completed_ * 100) / ie.export_total_
                      : 0;
              self->SetTaskState(ie.export_status_, percent, false);
            },
            Qt::QueuedConnection);
      },
      [self, skipped = queue_result.skipped_count_](
          std::shared_ptr<std::vector<ExportResult>> results) {
        if (!self) return;
        QMetaObject::invokeMethod(
            self,
            [self, results, skipped]() {
              if (!self) return;
              self->import_export_.FinishExport(results, skipped);
            },
            Qt::QueuedConnection);
      });
}

void ImportExportHandler::ResetExportState() {
  if (export_inflight_) return;
  ResetExportProgressState("Ready to export.");
}

void ImportExportHandler::FinishImport(const ImportResult& result) {
  const auto importJob = current_import_job_;
  current_import_job_.reset();

  if (!importJob || !importJob->import_log_) {
    backend_.SetTaskState("Import finished but no log snapshot is available.", 0, false);
    return;
  }

  const auto snapshot = importJob->import_log_->Snapshot();

  bool state_saved = true;
  try {
    auto* isvc = backend_.project_handler_.import_service();
    if (isvc) {
      isvc->SyncImports(snapshot, import_target_folder_path_);
    }
    auto proj = backend_.project_handler_.project();
    if (proj) {
      proj->GetSleeveService()->Sync();
      proj->GetImagePoolService()->SyncWithStorage();
      proj->SaveProject(backend_.project_handler_.meta_path());
    }
  } catch (...) {
    state_saved = false;
  }

  QString package_error;
  bool    package_saved = true;
  if (state_saved) {
    package_saved = backend_.project_handler_.PackageCurrentProjectFiles(&package_error);
  }

  AddImportedEntries(snapshot);
  backend_.stats_.RebuildThumbnailView();
  backend_.stats_.RefreshStats();

  import_target_folder_id_   = backend_.folder_ctrl_.current_folder_id();
  import_target_folder_path_ = backend_.folder_ctrl_.CurrentFolderFsPath();

  QString task_text =
      QString("Import complete: %1 imported, %2 failed").arg(result.imported_).arg(result.failed_);
  if (!state_saved) {
    task_text += " (project sync/save failed)";
    backend_.SetServiceMessageForCurrentProject(
        "Import finished, but saving project state failed.");
  } else if (!package_saved) {
    task_text += " (project packing failed)";
    backend_.SetServiceMessageForCurrentProject(
        package_error.isEmpty() ? "Import finished, but project packing failed."
                                : package_error);
  }
  backend_.SetTaskState(task_text, 100, false);
  backend_.ScheduleIdleTaskStateReset(1800);
}

void ImportExportHandler::FinishExport(
    const std::shared_ptr<std::vector<ExportResult>>& results, int skippedCount) {
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

  const int total   = ok + fail;
  export_total_     = std::max(export_total_, total);
  export_completed_ = total;
  export_succeeded_ = ok;
  export_failed_    = fail;
  export_skipped_   = skippedCount;
  export_error_summary_.clear();
  if (!errors.isEmpty()) {
    export_error_summary_ = errors.join('\n');
  }

  export_status_ = QString("Export complete. Written %1/%2 image(s), failed %3.")
                       .arg(ok).arg(total).arg(fail);
  if (skippedCount > 0) {
    export_status_ += QString(" Skipped %1 invalid item(s).").arg(skippedCount);
  }
  emit backend_.ExportStateChanged();
  emit backend_.exportStateChanged();

  backend_.SetTaskState(
      QString("Export complete: %1 ok, %2 failed").arg(ok).arg(fail), 100, false);
  backend_.ScheduleIdleTaskStateReset(1800);
}

void ImportExportHandler::AddImportedEntries(const ImportLogSnapshot& snapshot) {
  std::unordered_set<image_id_t> metadataOk;
  metadataOk.reserve(snapshot.metadata_ok_.size() * 2 + 1);
  for (const auto id : snapshot.metadata_ok_) {
    metadataOk.insert(id);
  }

  for (const auto& created : snapshot.created_) {
    if (!metadataOk.empty() && !metadataOk.contains(created.image_id_)) {
      continue;
    }
    backend_.AddOrUpdateAlbumItem(created.element_id_, created.image_id_,
                                  created.file_name_, import_target_folder_id_);
  }
}

auto ImportExportHandler::CollectExportTargets(const QVariantList& targetEntries) const
    -> std::vector<ExportTarget> {
  const QVariantList& source = targetEntries.empty() ? backend_.visible_thumbnails_ : targetEntries;
  std::vector<ExportTarget> targets;
  targets.reserve(static_cast<size_t>(source.size()));

  std::unordered_set<uint64_t> dedupe;
  dedupe.reserve(static_cast<size_t>(source.size()) * 2 + 1);

  for (const QVariant& entry : source) {
    const auto map       = entry.toMap();
    const auto elementId = static_cast<sl_element_id_t>(map.value("elementId").toUInt());
    const auto imageId   = static_cast<image_id_t>(map.value("imageId").toUInt());
    if (elementId == 0 || imageId == 0) continue;
    if (!dedupe.insert(ExportTargetKey(elementId, imageId)).second) continue;
    targets.emplace_back(elementId, imageId);
  }
  return targets;
}

auto ImportExportHandler::BuildExportQueue(
    const std::vector<ExportTarget>& targets, const std::filesystem::path& outputDir,
    ImageFormatType format, bool resizeEnabled, int maxLengthSide, int quality,
    ExportFormatOptions::BIT_DEPTH bitDepth, int pngCompressionLevel,
    ExportFormatOptions::TIFF_COMPRESS tiffCompression) -> ExportQueueBuildResult {
  ExportQueueBuildResult summary;
  auto  proj = backend_.project_handler_.project();
  const auto& esvc = backend_.project_handler_.export_service();
  if (!proj || !esvc) {
    summary.first_error_ = "Export service is unavailable.";
    return summary;
  }

  for (const auto& [elementId, imageId] : targets) {
    try {
      const auto srcPath = proj->GetImagePoolService()->Read<std::filesystem::path>(
          imageId,
          [](const std::shared_ptr<Image>& image) {
            return image ? image->image_path_ : image_path_t{};
          });
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

      esvc->EnqueueExportTask(task);
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

void ImportExportHandler::ResetExportProgressState(const QString& status) {
  export_status_ = status;
  export_error_summary_.clear();
  export_total_     = 0;
  export_completed_ = 0;
  export_succeeded_ = 0;
  export_failed_    = 0;
  export_skipped_   = 0;
  emit backend_.ExportStateChanged();
  emit backend_.exportStateChanged();
}

void ImportExportHandler::SetExportFailureState(const QString& message) {
  export_status_ = message;
  emit backend_.ExportStateChanged();
  emit backend_.exportStateChanged();
  backend_.SetTaskState(message, 0, false);
}

}  // namespace puerhlab::ui
