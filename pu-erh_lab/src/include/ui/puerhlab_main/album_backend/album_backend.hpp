#pragma once

#include <QDate>
#include <QObject>
#include <QStringList>
#include <QVariantList>

#include <unordered_map>
#include <vector>

#include "ui/puerhlab_main/album_backend/filter_rule_model.hpp"
#include "ui/puerhlab_main/album_backend/album_types.hpp"
#include "ui/puerhlab_main/album_backend/project_handler.hpp"
#include "ui/puerhlab_main/album_backend/thumbnail_manager.hpp"
#include "ui/puerhlab_main/album_backend/folder_controller.hpp"
#include "ui/puerhlab_main/album_backend/filter_engine.hpp"
#include "ui/puerhlab_main/album_backend/import_export.hpp"
#include "ui/puerhlab_main/album_backend/editor_controller.hpp"

namespace puerhlab::ui {

class AlbumBackend final : public QObject {
  Q_OBJECT
  Q_PROPERTY(puerhlab::ui::FilterRuleModel* filterRules READ FilterRules CONSTANT)
  Q_PROPERTY(QVariantList fieldOptions READ FieldOptions CONSTANT)
  Q_PROPERTY(QVariantList thumbnails READ Thumbnails NOTIFY ThumbnailsChanged)
  Q_PROPERTY(QVariantList folders READ Folders NOTIFY FoldersChanged)
  Q_PROPERTY(uint currentFolderId READ CurrentFolderId NOTIFY FolderSelectionChanged)
  Q_PROPERTY(QString currentFolderPath READ CurrentFolderPath NOTIFY FolderSelectionChanged)
  Q_PROPERTY(int shownCount READ ShownCount NOTIFY CountsChanged)
  Q_PROPERTY(int totalCount READ TotalCount NOTIFY CountsChanged)
  Q_PROPERTY(QString filterInfo READ FilterInfo NOTIFY CountsChanged)
  Q_PROPERTY(QString sqlPreview READ SqlPreview NOTIFY SqlPreviewChanged)
  Q_PROPERTY(QString validationError READ ValidationError NOTIFY ValidationErrorChanged)
  Q_PROPERTY(bool serviceReady READ ServiceReady NOTIFY ServiceStateChanged)
  Q_PROPERTY(QString serviceMessage READ ServiceMessage NOTIFY ServiceStateChanged)
  Q_PROPERTY(bool projectLoading READ ProjectLoading NOTIFY ProjectLoadStateChanged)
  Q_PROPERTY(QString projectLoadingMessage READ ProjectLoadingMessage NOTIFY ProjectLoadStateChanged)
  Q_PROPERTY(QString taskStatus READ TaskStatus NOTIFY TaskStateChanged)
  Q_PROPERTY(int taskProgress READ TaskProgress NOTIFY TaskStateChanged)
  Q_PROPERTY(bool taskCancelVisible READ TaskCancelVisible NOTIFY TaskStateChanged)
  Q_PROPERTY(QString defaultExportFolder READ DefaultExportFolder CONSTANT)
  Q_PROPERTY(bool exportInFlight READ ExportInFlight NOTIFY ExportStateChanged)
  Q_PROPERTY(QString exportStatus READ ExportStatus NOTIFY ExportStateChanged)
  Q_PROPERTY(int exportTotal READ ExportTotal NOTIFY ExportStateChanged)
  Q_PROPERTY(int exportCompleted READ ExportCompleted NOTIFY ExportStateChanged)
  Q_PROPERTY(int exportSucceeded READ ExportSucceeded NOTIFY ExportStateChanged)
  Q_PROPERTY(int exportFailed READ ExportFailed NOTIFY ExportStateChanged)
  Q_PROPERTY(int exportSkipped READ ExportSkipped NOTIFY ExportStateChanged)
  Q_PROPERTY(QString exportErrorSummary READ ExportErrorSummary NOTIFY ExportStateChanged)
  Q_PROPERTY(bool editorActive READ EditorActive NOTIFY EditorStateChanged)
  Q_PROPERTY(bool editorBusy READ EditorBusy NOTIFY EditorStateChanged)
  Q_PROPERTY(uint editorElementId READ EditorElementId NOTIFY EditorStateChanged)
  Q_PROPERTY(QString editorTitle READ EditorTitle NOTIFY EditorStateChanged)
  Q_PROPERTY(QString editorStatus READ EditorStatus NOTIFY EditorStateChanged)
  Q_PROPERTY(QString editorPreviewUrl READ EditorPreviewUrl NOTIFY EditorPreviewChanged)
  Q_PROPERTY(QVariantList editorLutOptions READ EditorLutOptions NOTIFY EditorStateChanged)
  Q_PROPERTY(int editorLutIndex READ EditorLutIndex NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorExposure READ EditorExposure NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorContrast READ EditorContrast NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorSaturation READ EditorSaturation NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorTint READ EditorTint NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorBlacks READ EditorBlacks NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorWhites READ EditorWhites NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorShadows READ EditorShadows NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorHighlights READ EditorHighlights NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorSharpen READ EditorSharpen NOTIFY EditorStateChanged)
  Q_PROPERTY(double editorClarity READ EditorClarity NOTIFY EditorStateChanged)

 public:
  explicit AlbumBackend(QObject* parent = nullptr);
  ~AlbumBackend() override;

  // ── Q_PROPERTY getters ──────────────────────────────────────────────
  FilterRuleModel* FilterRules() { return &rule_model_; }
  QVariantList FieldOptions() const;
  QVariantList Thumbnails() const { return visible_thumbnails_; }
  QVariantList Folders() const { return folder_ctrl_.folders(); }
  uint CurrentFolderId() const { return static_cast<uint>(folder_ctrl_.current_folder_id()); }
  const QString& CurrentFolderPath() const { return folder_ctrl_.current_folder_path_text(); }
  int ShownCount() const { return static_cast<int>(visible_thumbnails_.size()); }
  int TotalCount() const;
  QString FilterInfo() const;
  const QString& SqlPreview() const { return sql_preview_; }
  const QString& ValidationError() const { return validation_error_; }
  bool ServiceReady() const { return service_ready_; }
  const QString& ServiceMessage() const { return service_message_; }
  bool ProjectLoading() const { return project_handler_.project_loading(); }
  const QString& ProjectLoadingMessage() const { return project_handler_.project_loading_message(); }
  const QString& TaskStatus() const { return task_status_; }
  int TaskProgress() const { return task_progress_; }
  bool TaskCancelVisible() const { return task_cancel_visible_; }
  const QString& DefaultExportFolder() const { return import_export_.default_export_folder(); }
  bool ExportInFlight() const { return import_export_.export_inflight(); }
  const QString& ExportStatus() const { return import_export_.export_status(); }
  int ExportTotal() const { return import_export_.export_total(); }
  int ExportCompleted() const { return import_export_.export_completed(); }
  int ExportSucceeded() const { return import_export_.export_succeeded(); }
  int ExportFailed() const { return import_export_.export_failed(); }
  int ExportSkipped() const { return import_export_.export_skipped(); }
  const QString& ExportErrorSummary() const { return import_export_.export_error_summary(); }
  bool EditorActive() const { return editor_.editor_active(); }
  bool EditorBusy() const { return editor_.editor_busy(); }
  uint EditorElementId() const { return static_cast<uint>(editor_.editor_element_id()); }
  const QString& EditorTitle() const { return editor_.editor_title(); }
  const QString& EditorStatus() const { return editor_.editor_status(); }
  const QString& EditorPreviewUrl() const { return editor_.editor_preview_url(); }
  QVariantList EditorLutOptions() const { return editor_.editor_lut_options(); }
  int EditorLutIndex() const { return editor_.editor_lut_index(); }
  double EditorExposure() const { return editor_.editor_state().exposure_; }
  double EditorContrast() const { return editor_.editor_state().contrast_; }
  double EditorSaturation() const { return editor_.editor_state().saturation_; }
  double EditorTint() const { return editor_.editor_state().tint_; }
  double EditorBlacks() const { return editor_.editor_state().blacks_; }
  double EditorWhites() const { return editor_.editor_state().whites_; }
  double EditorShadows() const { return editor_.editor_state().shadows_; }
  double EditorHighlights() const { return editor_.editor_state().highlights_; }
  double EditorSharpen() const { return editor_.editor_state().sharpen_; }
  double EditorClarity() const { return editor_.editor_state().clarity_; }

  Q_INVOKABLE void AddRule();
  Q_INVOKABLE void RemoveRule(int index);
  Q_INVOKABLE void SetRuleField(int index, int fieldValue);
  Q_INVOKABLE void SetRuleOp(int index, int opValue);
  Q_INVOKABLE void SetRuleValue(int index, const QString& value);
  Q_INVOKABLE void SetRuleValue2(int index, const QString& value);

  Q_INVOKABLE void ApplyFilters(int joinOpValue);
  Q_INVOKABLE void ClearFilters();
  Q_INVOKABLE QVariantList CompareOptionsForField(int fieldValue) const;
  Q_INVOKABLE QString PlaceholderForField(int fieldValue) const;
  Q_INVOKABLE void SelectFolder(uint folderId);
  Q_INVOKABLE void CreateFolder(const QString& folderName);
  Q_INVOKABLE void DeleteFolder(uint folderId);

  Q_INVOKABLE void StartImport(const QStringList& fileUrlsOrPaths);
  Q_INVOKABLE void CancelImport();
  Q_INVOKABLE bool LoadProject(const QString& metaFileUrlOrPath);
  Q_INVOKABLE bool CreateProjectInFolder(const QString& folderUrlOrPath);
  Q_INVOKABLE bool CreateProjectInFolderNamed(const QString& folderUrlOrPath,
                                              const QString& projectName);
  Q_INVOKABLE bool SaveProject();
  Q_INVOKABLE void StartExport(const QString& outputDirUrlOrPath);
  Q_INVOKABLE void StartExportWithOptions(const QString& outputDirUrlOrPath,
                                          const QString& formatName, bool resizeEnabled,
                                          int maxLengthSide, int quality, int bitDepth,
                                          int pngCompressionLevel,
                                          const QString& tiffCompression);
  Q_INVOKABLE void StartExportWithOptionsForTargets(const QString& outputDirUrlOrPath,
                                                    const QString& formatName,
                                                    bool resizeEnabled, int maxLengthSide,
                                                    int quality, int bitDepth,
                                                    int pngCompressionLevel,
                                                    const QString& tiffCompression,
                                                    const QVariantList& targetEntries);
  Q_INVOKABLE void ResetExportState();
  Q_INVOKABLE void OpenEditor(uint elementId, uint imageId);
  Q_INVOKABLE void CloseEditor();
  Q_INVOKABLE void ResetEditorAdjustments();
  Q_INVOKABLE void RequestEditorFullPreview();
  Q_INVOKABLE void SetEditorLutIndex(int index);
  Q_INVOKABLE void SetEditorExposure(double value);
  Q_INVOKABLE void SetEditorContrast(double value);
  Q_INVOKABLE void SetEditorSaturation(double value);
  Q_INVOKABLE void SetEditorTint(double value);
  Q_INVOKABLE void SetEditorBlacks(double value);
  Q_INVOKABLE void SetEditorWhites(double value);
  Q_INVOKABLE void SetEditorShadows(double value);
  Q_INVOKABLE void SetEditorHighlights(double value);
  Q_INVOKABLE void SetEditorSharpen(double value);
  Q_INVOKABLE void SetEditorClarity(double value);
  Q_INVOKABLE void SetThumbnailVisible(uint elementId, uint imageId, bool visible);

signals:
  void ThumbnailsChanged();
  void thumbnailsChanged();
  void ThumbnailUpdated(uint elementId, const QString& dataUrl);
  void thumbnailUpdated(uint elementId, const QString& dataUrl);
  void CountsChanged();
  void SqlPreviewChanged();
  void ValidationErrorChanged();
  void ServiceStateChanged();
  void TaskStateChanged();
  void ExportStateChanged();
  void exportStateChanged();
  void EditorStateChanged();
  void EditorPreviewChanged();
  void ProjectChanged();
  void projectChanged();
  void ProjectLoadStateChanged();
  void FoldersChanged();
  void FolderSelectionChanged();
  void folderSelectionChanged();

 private:
  friend class ProjectHandler;
  friend class ThumbnailManager;
  friend class FolderController;
  friend class FilterEngine;
  friend class ImportExportHandler;
  friend class EditorController;

  void SetServiceState(bool ready, const QString& message);
  void SetServiceMessageForCurrentProject(const QString& message);
  void ScheduleIdleTaskStateReset(int delayMs);
  void SetTaskState(const QString& status, int progress, bool cancelVisible);
  void AddOrUpdateAlbumItem(sl_element_id_t elementId, image_id_t imageId,
                            const file_name_t& fallbackName,
                            sl_element_id_t parentFolderId);

  // ── Helper modules ──────────────────────────────────────────────────
  ProjectHandler     project_handler_;
  ThumbnailManager   thumb_;
  FolderController   folder_ctrl_;
  FilterEngine       filter_;
  ImportExportHandler import_export_;
  EditorController   editor_;

  // ── Shared data (accessed by helpers via friend) ────────────────────
  FilterRuleModel                                     rule_model_;
  std::vector<AlbumItem>                              all_images_{};
  std::unordered_map<sl_element_id_t, size_t>         index_by_element_id_{};
  QVariantList                                        visible_thumbnails_{};
  QString                                             sql_preview_{};
  QString                                             validation_error_{};
  QString                                             service_message_ = "Initializing backend services...";
  bool                                                service_ready_   = false;
  QString                                             task_status_     = "No background tasks";
  int                                                 task_progress_   = 0;
  bool                                                task_cancel_visible_ = false;
};

}  // namespace puerhlab::ui
