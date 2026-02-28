#pragma once

#include <QStringList>
#include <QVariantList>

#include <filesystem>
#include <memory>
#include <vector>

#include "ui/puerhlab_main/album_backend/album_types.hpp"
#include "app/export_service.hpp"
#include "app/import_service.hpp"
#include "type/supported_file_type.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Handles file import and image export workflows.
class ImportExportHandler {
 public:
  explicit ImportExportHandler(AlbumBackend& backend);

  void StartImport(const QStringList& fileUrlsOrPaths);
  void CancelImport();
  void StartExport(const QString& outputDirUrlOrPath);
  void StartExportWithOptions(const QString& outputDirUrlOrPath, const QString& formatName,
                              bool resizeEnabled, int maxLengthSide, int quality, int bitDepth,
                              int pngCompressionLevel, const QString& tiffCompression);
  void StartExportWithOptionsForTargets(const QString& outputDirUrlOrPath,
                                        const QString& formatName, bool resizeEnabled,
                                        int maxLengthSide, int quality, int bitDepth,
                                        int pngCompressionLevel,
                                        const QString& tiffCompression,
                                        const QVariantList& targetEntries);
  void ResetExportState();

  void FinishImport(const ImportResult& result);
  void FinishExport(const std::shared_ptr<std::vector<ExportResult>>& results, int skippedCount);
  void AddImportedEntries(const ImportLogSnapshot& snapshot);

  [[nodiscard]] auto CollectExportTargets(const QVariantList& targetEntries) const
      -> std::vector<ExportTarget>;
  auto BuildExportQueue(const std::vector<ExportTarget>& targets,
                        const std::filesystem::path& outputDir, ImageFormatType format,
                        bool resizeEnabled, int maxLengthSide, int quality,
                        ExportFormatOptions::BIT_DEPTH bitDepth, int pngCompressionLevel,
                        ExportFormatOptions::TIFF_COMPRESS tiffCompression)
      -> ExportQueueBuildResult;

  [[nodiscard]] bool export_inflight() const { return export_inflight_; }
  [[nodiscard]] auto current_import_job() const -> const std::shared_ptr<ImportJob>& {
    return current_import_job_;
  }
  [[nodiscard]] auto default_export_folder() const -> const QString& { return default_export_folder_; }
  [[nodiscard]] auto export_status() const -> const QString& { return export_status_; }
  [[nodiscard]] auto export_error_summary() const -> const QString& { return export_error_summary_; }
  [[nodiscard]] int  export_total() const { return export_total_; }
  [[nodiscard]] int  export_completed() const { return export_completed_; }
  [[nodiscard]] int  export_succeeded() const { return export_succeeded_; }
  [[nodiscard]] int  export_failed() const { return export_failed_; }
  [[nodiscard]] int  export_skipped() const { return export_skipped_; }
  [[nodiscard]] auto import_target_folder_id() const -> sl_element_id_t { return import_target_folder_id_; }
  [[nodiscard]] auto import_target_folder_path() const -> const std::filesystem::path& {
    return import_target_folder_path_;
  }

  void SetImportTarget(sl_element_id_t folderId, const std::filesystem::path& folderPath) {
    import_target_folder_id_   = folderId;
    import_target_folder_path_ = folderPath;
  }
  void ClearImportTarget() {
    import_target_folder_id_ = 0;
    import_target_folder_path_.clear();
  }

 private:
  void ResetExportProgressState(const QString& status);
  void SetExportFailureState(const QString& message);

  AlbumBackend& backend_;

  std::shared_ptr<ImportJob> current_import_job_{};
  bool    export_inflight_      = false;
  QString default_export_folder_{};
  QString export_status_        = "Ready to export.";
  QString export_error_summary_{};
  int     export_total_         = 0;
  int     export_completed_     = 0;
  int     export_succeeded_     = 0;
  int     export_failed_        = 0;
  int     export_skipped_       = 0;

  sl_element_id_t       import_target_folder_id_   = 0;
  std::filesystem::path import_target_folder_path_{};
};

}  // namespace puerhlab::ui
