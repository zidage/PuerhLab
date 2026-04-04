//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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
#include <utility>

namespace puerhlab::ui {

using namespace album_util;

#define PL_TEXT(text, ...)                                                    \
  i18n::MakeLocalizedText(PUERHLAB_I18N_CONTEXT,                              \
                          QT_TRANSLATE_NOOP(PUERHLAB_I18N_CONTEXT, text)      \
                              __VA_OPT__(, ) __VA_ARGS__)

namespace {

auto EffectiveExportFormat(ImageFormatType format,
                           ExportFormatOptions::HDR_EXPORT_MODE hdrExportMode) -> ImageFormatType {
  if (hdrExportMode == ExportFormatOptions::HDR_EXPORT_MODE::ULTRA_HDR) {
    return ImageFormatType::JPEG;
  }
  return format;
}

auto SanitizeBitDepth(ImageFormatType format,
                      ExportFormatOptions::BIT_DEPTH requested) -> ExportFormatOptions::BIT_DEPTH {
  switch (format) {
    case ImageFormatType::JPEG:
    case ImageFormatType::WEBP:
      return ExportFormatOptions::BIT_DEPTH::BIT_8;
    case ImageFormatType::PNG:
      return requested == ExportFormatOptions::BIT_DEPTH::BIT_8
                 ? ExportFormatOptions::BIT_DEPTH::BIT_8
                 : ExportFormatOptions::BIT_DEPTH::BIT_16;
    case ImageFormatType::EXR:
      return requested == ExportFormatOptions::BIT_DEPTH::BIT_32
                 ? ExportFormatOptions::BIT_DEPTH::BIT_32
                 : ExportFormatOptions::BIT_DEPTH::BIT_16;
    case ImageFormatType::TIFF:
      return requested;
    default:
      return ExportFormatOptions::BIT_DEPTH::BIT_8;
  }
}

}  // namespace

ImportExportHandler::ImportExportHandler(AlbumBackend& backend) : backend_(backend) {
  const QString pictures =
      QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
  if (!pictures.isEmpty()) {
    default_export_folder_ = pictures;
  }
  export_status_text_ = PL_TEXT("Ready to export.");
}

void ImportExportHandler::StartImport(const QStringList& fileUrlsOrPaths) {
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
    backend_.SetTaskState(PL_TEXT("No supported files selected."), 0, false);
    return;
  }
  StartImportResolvedPaths(std::move(paths), false);
}

void ImportExportHandler::StartImportPaths(const std::vector<image_path_t>& paths,
                                           const bool                      preserveTarget) {
  std::vector<image_path_t> deduped_paths;
  std::unordered_set<std::wstring> seen;
  deduped_paths.reserve(paths.size());
  for (const auto& path : paths) {
    if (path.empty()) {
      continue;
    }
    const std::wstring key = path.wstring();
    if (!seen.insert(key).second) {
      continue;
    }
    deduped_paths.push_back(path);
  }
  if (deduped_paths.empty()) {
    backend_.SetTaskState(PL_TEXT("No supported files selected."), 0, false);
    return;
  }
  StartImportResolvedPaths(std::move(deduped_paths), preserveTarget);
}

void ImportExportHandler::StartImportResolvedPaths(std::vector<image_path_t> paths,
                                                   const bool preserveTarget) {
  if (backend_.project_handler_.project_loading()) {
    backend_.SetTaskState(PL_TEXT("Project is loading. Please wait."), 0, false);
    return;
  }
  auto* isvc = backend_.project_handler_.import_service();
  if (!isvc) {
    backend_.SetTaskState(PL_TEXT("Import service is unavailable."), 0, false);
    return;
  }
  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    backend_.SetTaskState(PL_TEXT("Import already running."), backend_.task_progress_, true);
    return;
  }

  if (!preserveTarget) {
    import_target_folder_id_   = backend_.folder_ctrl_.CurrentFolderElementId().value_or(0);
    import_target_folder_path_ = backend_.folder_ctrl_.CurrentFolderFsPath();
  }

  auto job            = std::make_shared<ImportJob>();
  current_import_job_ = job;

  import_running_   = true;
  import_total_     = static_cast<int>(paths.size());
  import_completed_ = 0;
  import_failed_    = 0;
  import_status_text_ = PL_TEXT("Importing %1 file(s)...", import_total_);
  emit backend_.ImportStateChanged();
  emit backend_.importStateChanged();

  backend_.SetTaskState(import_status_text_, 0, true);

  QPointer<AlbumBackend> self(&backend_);
  job->on_progress_ = [self](const ImportProgress& progress) {
    if (!self) return;
    const uint32_t total        = std::max<uint32_t>(progress.total_, 1);
    const uint32_t metadataDone = progress.metadata_done_.load();
    const uint32_t failed       = progress.failed_.load();
    const uint32_t done         = metadataDone + failed;
    const int      pct          = static_cast<int>((done * 100U) / total);

    QMetaObject::invokeMethod(
        self,
        [self, metadataDone, total, failed, pct]() {
          if (!self) return;
          auto& ie = self->import_export_;
          ie.import_completed_ = static_cast<int>(metadataDone);
          ie.import_failed_    = static_cast<int>(failed);
          ie.import_status_text_ =
              PL_TEXT("Importing... %1/%2 (failed %3)", metadataDone, total, failed);
          emit self->ImportStateChanged();
          emit self->importStateChanged();
          self->SetTaskState(ie.import_status_text_, pct, true);
          if (self->nikon_he_recovery_.is_reimporting()) {
            self->nikon_he_recovery_.UpdateReimportProgress(metadataDone, total, failed);
          }
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
    import_running_ = false;
    import_status_text_ = PL_TEXT("Import failed: %1", QString::fromUtf8(e.what()));
    emit backend_.ImportStateChanged();
    emit backend_.importStateChanged();
    backend_.SetTaskState(import_status_text_, 0, false);
  }
}

void ImportExportHandler::CancelImport() {
  if (!current_import_job_) return;
  current_import_job_->canceled_.store(true);
  import_status_text_ = PL_TEXT("Cancelling import...");
  emit backend_.ImportStateChanged();
  emit backend_.importStateChanged();
  backend_.SetTaskState(PL_TEXT("Cancelling import..."), backend_.task_progress_, true);
}

void ImportExportHandler::StartExport(const QString& outputDirUrlOrPath) {
  StartExportWithOptionsForTargets(outputDirUrlOrPath, "JPEG", "ULTRA_HDR", false, 4096, 95, 16,
                                   5, "NONE", {});
}

void ImportExportHandler::StartExportWithOptions(const QString& outputDirUrlOrPath,
                                                 const QString& formatName,
                                                 const QString& hdrExportMode,
                                                 bool resizeEnabled, int maxLengthSide,
                                                 int quality, int bitDepth,
                                                 int pngCompressionLevel,
                                                 const QString& tiffCompression) {
  StartExportWithOptionsForTargets(outputDirUrlOrPath, formatName, hdrExportMode, resizeEnabled,
                                   maxLengthSide, quality, bitDepth, pngCompressionLevel,
                                   tiffCompression, {});
}

void ImportExportHandler::StartExportWithOptionsForTargets(
    const QString& outputDirUrlOrPath, const QString& formatName, const QString& hdrExportMode,
    bool resizeEnabled, int maxLengthSide, int quality, int bitDepth,
    int pngCompressionLevel, const QString& tiffCompression,
    const QVariantList& targetEntries) {
  if (backend_.project_handler_.project_loading()) {
    SetExportFailureState(PL_TEXT("Project is loading. Please wait."));
    return;
  }

  const auto& esvc = backend_.project_handler_.export_service();
  auto  proj = backend_.project_handler_.project();
  if (!esvc || !proj) {
    SetExportFailureState(PL_TEXT("Export service is unavailable."));
    return;
  }
  if (export_inflight_) {
    SetExportFailureState(PL_TEXT("Export already running."));
    return;
  }

  ResetExportProgressState(PL_TEXT("Preparing export queue..."));

  const auto outDirOpt = InputToPath(outputDirUrlOrPath);
  if (!outDirOpt.has_value()) {
    SetExportFailureState(PL_TEXT("No export folder selected."));
    return;
  }

  std::error_code ec;
  if (!std::filesystem::exists(outDirOpt.value(), ec)) {
    std::filesystem::create_directories(outDirOpt.value(), ec);
  }
  if (ec || !std::filesystem::is_directory(outDirOpt.value(), ec) || ec) {
    SetExportFailureState(PL_TEXT("Export folder is invalid."));
    return;
  }

  const auto targets = CollectExportTargets(targetEntries);
  if (targets.empty()) {
    SetExportFailureState(PL_TEXT("No images to export."));
    return;
  }

  const ImageFormatType requested_format = FormatFromName(formatName);
  const auto            hdr_mode         = HdrExportModeFromName(hdrExportMode);
  const int             clamped_max   = std::clamp(maxLengthSide, 256, 16384);
  const int             clamped_q     = std::clamp(quality, 1, 100);
  const ImageFormatType effective_format = EffectiveExportFormat(requested_format, hdr_mode);
  const auto            bit_depth        = SanitizeBitDepth(effective_format, BitDepthFromInt(bitDepth));
  const int             clamped_png   = std::clamp(pngCompressionLevel, 0, 9);
  const auto            tiff_compress = TiffCompressFromName(tiffCompression);

  esvc->ClearAllExportTasks();
  const auto queue_result =
      BuildExportQueue(targets, outDirOpt.value(), effective_format, hdr_mode, resizeEnabled, clamped_max,
                       clamped_q, bit_depth, clamped_png, tiff_compress);

  if (queue_result.queued_count_ == 0) {
    export_status_text_ = PL_TEXT("No export tasks were queued.");
    if (!queue_result.first_error_.isEmpty()) {
      export_error_summary_text_ = PL_TEXT("%1", queue_result.first_error_);
    }
    emit backend_.ExportStateChanged();
    emit backend_.exportStateChanged();
    backend_.SetTaskState(PL_TEXT("No valid export tasks could be created."), 0, false);
    return;
  }

  export_inflight_ = true;
  export_total_    = queue_result.queued_count_;
  export_skipped_  = queue_result.skipped_count_;
  if (queue_result.skipped_count_ > 0) {
    export_status_text_ = PL_TEXT("Exporting %1 image(s). Skipped %2 invalid item(s).",
                                  queue_result.queued_count_, queue_result.skipped_count_);
  } else {
    export_status_text_ = PL_TEXT("Exporting %1 image(s)...", queue_result.queued_count_);
  }
  emit backend_.ExportStateChanged();
  emit backend_.exportStateChanged();
  backend_.SetTaskState(export_status_text_, 0, false);

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
              ie.export_status_text_ =
                  PL_TEXT("Exporting... processed %1/%2, written %3, failed %4.",
                          ie.export_completed_, ie.export_total_, ie.export_succeeded_,
                          ie.export_failed_);
              emit self->ExportStateChanged();

              const int percent =
                  ie.export_total_ > 0
                      ? (ie.export_completed_ * 100) / ie.export_total_
                      : 0;
              self->SetTaskState(ie.export_status_text_, percent, false);
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
  ResetExportProgressState(PL_TEXT("Ready to export."));
}

void ImportExportHandler::FinishImport(const ImportResult& result) {
  const auto importJob = current_import_job_;
  current_import_job_.reset();

  if (!importJob || !importJob->import_log_) {
    backend_.SetTaskState(PL_TEXT("Import finished but no log snapshot is available."), 0, false);
    return;
  }

  const auto snapshot = importJob->import_log_->Snapshot();
  const bool reimporting_nikon_he = backend_.nikon_he_recovery_.is_reimporting();
  const auto recovery_target_folder_id = import_target_folder_id_;
  const auto recovery_target_folder_path = import_target_folder_path_;

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

  backend_.ReloadCurrentFolder();
  backend_.stats_.ClearFilters();
  emit backend_.StatsFilterChanged();

  import_target_folder_id_   = backend_.folder_ctrl_.CurrentFolderElementId().value_or(0);
  import_target_folder_path_ = backend_.folder_ctrl_.CurrentFolderFsPath();

  auto task_text = PL_TEXT("Import complete: %1 imported, %2 failed", result.imported_,
                           result.failed_);
  if (!state_saved) {
    backend_.SetServiceMessageForCurrentProject(
        PL_TEXT("Import finished, but saving project state failed."));
  } else if (!package_saved) {
    backend_.SetServiceMessageForCurrentProject(
        package_error.isEmpty() ? PL_TEXT("Import finished, but project packing failed.")
                                : PL_TEXT("%1", package_error));
  }
  import_running_   = false;
  import_completed_ = static_cast<int>(result.imported_);
  import_failed_    = static_cast<int>(result.failed_);
  import_status_text_ = task_text;
  emit backend_.ImportStateChanged();
  emit backend_.importStateChanged();

  backend_.SetTaskState(task_text, 100, false);
  backend_.ScheduleIdleTaskStateReset(1800);

  if (reimporting_nikon_he) {
    backend_.nikon_he_recovery_.HandleReimportFinished(result);
    return;
  }

  if (!snapshot.unsupported_nikon_he_.empty()) {
    backend_.nikon_he_recovery_.BeginRecovery(snapshot.unsupported_nikon_he_,
                                              recovery_target_folder_id,
                                              recovery_target_folder_path);
  }
}

void ImportExportHandler::FinishExport(
    const std::shared_ptr<std::vector<ExportResult>>& results, int skippedCount) {
  export_inflight_ = false;

  int         ok   = 0;
  int         fail = 0;
  int         ultra_hdr_fallbacks = 0;
  QStringList errors;
  if (results) {
    for (const auto& r : *results) {
      if (r.success_) {
        ++ok;
        if (r.used_embedded_profile_fallback_) {
          ++ultra_hdr_fallbacks;
        }
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
  export_error_summary_text_ = {};
  if (!errors.isEmpty()) {
    export_error_summary_text_ = PL_TEXT("%1", errors.join('\n'));
  }

  export_status_text_ = PL_TEXT("Export complete. Written %1/%2 image(s), failed %3.", ok, total,
                                fail);
  if (skippedCount > 0) {
    export_status_text_ = PL_TEXT(
        "Export complete. Written %1/%2 image(s), failed %3. Skipped %4 invalid item(s).", ok,
        total, fail, skippedCount);
  }
  if (ultra_hdr_fallbacks > 0) {
    export_status_text_ = PL_TEXT(
        "Export complete. Written %1/%2 image(s), failed %3. %4 item(s) used embedded ICC fallback instead of Ultra HDR.",
        ok, total, fail, ultra_hdr_fallbacks);
    if (skippedCount > 0) {
      export_status_text_ = PL_TEXT(
          "Export complete. Written %1/%2 image(s), failed %3. Skipped %4 invalid item(s). %5 item(s) used embedded ICC fallback instead of Ultra HDR.",
          ok, total, fail, skippedCount, ultra_hdr_fallbacks);
    }
  }
  emit backend_.ExportStateChanged();
  emit backend_.exportStateChanged();

  backend_.SetTaskState(
      PL_TEXT("Export complete: %1 ok, %2 failed", ok, fail), 100, false);
  backend_.ScheduleIdleTaskStateReset(1800);
}

void ImportExportHandler::AddImportedEntries(const ImportLogSnapshot& snapshot) {
  (void)snapshot;
  // Deprecated path: folder content is now reloaded through AlbumBrowseService.
}

auto ImportExportHandler::CollectExportTargets(const QVariantList& targetEntries) const
    -> std::vector<ExportTarget> {
  const QVariantList& source =
      targetEntries.empty() ? backend_.view_state_.visible_thumbnails_ : targetEntries;
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
    ImageFormatType format, ExportFormatOptions::HDR_EXPORT_MODE hdrExportMode,
    bool resizeEnabled, int maxLengthSide, int quality, ExportFormatOptions::BIT_DEPTH bitDepth,
    int pngCompressionLevel,
    ExportFormatOptions::TIFF_COMPRESS tiffCompression) -> ExportQueueBuildResult {
  ExportQueueBuildResult summary;
  auto                   proj = backend_.project_handler_.project();
  const auto&            esvc = backend_.project_handler_.export_service();
  if (!proj || !esvc) {
    summary.first_error_ = PL_TEXT("Export service is unavailable.").Render();
    return summary;
  }

  std::unordered_set<std::wstring> planned_export_paths;
  planned_export_paths.reserve(targets.size() * 2 + 1);

  for (const auto& [elementId, imageId] : targets) {
    try {
      const auto source_info =
          proj->GetImagePoolService()->Read<std::pair<std::filesystem::path, std::wstring>>(
              imageId, [](const std::shared_ptr<Image>& image) {
                if (!image) {
                  return std::pair<std::filesystem::path, std::wstring>{};
                }
                std::wstring image_name = image->image_name_;
                if (image_name.empty() && !image->image_path_.empty()) {
                  image_name = image->image_path_.filename().wstring();
                }
                return std::make_pair(image->image_path_, std::move(image_name));
              });
      const auto& srcPath = source_info.first;
      if (srcPath.empty()) {
        ++summary.skipped_count_;
        if (summary.first_error_.isEmpty()) {
          summary.first_error_ = PL_TEXT("Image source path is empty.").Render();
        }
        continue;
      }

      std::filesystem::path name_source_path;
      if (!source_info.second.empty()) {
        name_source_path = std::filesystem::path(source_info.second).filename();
      }
      if (name_source_path.empty()) {
        name_source_path = srcPath.filename();
      }
      if (name_source_path.empty()) {
        name_source_path = std::filesystem::path(L"image");
      }

      auto       export_path = ExportPathForOptions(name_source_path, outputDir, elementId, imageId, format);
      const auto path_exists = [](const std::filesystem::path& p) {
        std::error_code ec;
        return std::filesystem::exists(p, ec);
      };
      if (planned_export_paths.contains(export_path.wstring()) || path_exists(export_path)) {
        const std::wstring stem = export_path.stem().wstring();
        const std::wstring ext  = export_path.extension().wstring();
        int                suffix_idx = 1;
        while (true) {
          const auto candidate =
              outputDir / (stem + L" (" + std::to_wstring(suffix_idx) + L")" + ext);
          if (!planned_export_paths.contains(candidate.wstring()) && !path_exists(candidate)) {
            export_path = candidate;
            break;
          }
          ++suffix_idx;
        }
      }
      planned_export_paths.insert(export_path.wstring());

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
      task.options_.hdr_export_mode_   = hdrExportMode;
      task.options_.export_path_       = std::move(export_path);

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
        summary.first_error_ = PL_TEXT("Unknown error while preparing export task.").Render();
      }
    }
  }
  return summary;
}

void ImportExportHandler::ResetExportProgressState(const i18n::LocalizedText& status) {
  export_status_text_        = status;
  export_error_summary_text_ = {};
  export_total_              = 0;
  export_completed_          = 0;
  export_succeeded_          = 0;
  export_failed_             = 0;
  export_skipped_            = 0;
  emit backend_.ExportStateChanged();
  emit backend_.exportStateChanged();
}

void ImportExportHandler::SetExportFailureState(const i18n::LocalizedText& message) {
  export_status_text_        = message;
  export_error_summary_text_ = {};
  emit backend_.ExportStateChanged();
  emit backend_.exportStateChanged();
  backend_.SetTaskState(message, 0, false);
}

}  // namespace puerhlab::ui

#undef PL_TEXT
