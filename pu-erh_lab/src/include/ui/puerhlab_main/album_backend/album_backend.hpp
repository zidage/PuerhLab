#pragma once

#include <QDate>
#include <QObject>
#include <QStringList>
#include <QTimer>
#include <QVariantList>

#include <ctime>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ui/puerhlab_main/album_backend/filter_rule_model.hpp"
#include "app/export_service.hpp"
#include "app/history_mgmt_service.hpp"
#include "app/import_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "app/sleeve_filter_service.hpp"
#include "app/thumbnail_service.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "renderer/pipeline_task.hpp"

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

  FilterRuleModel* FilterRules() { return &rule_model_; }
  QVariantList FieldOptions() const;
  QVariantList Thumbnails() const { return visible_thumbnails_; }
  QVariantList Folders() const { return folders_; }
  uint CurrentFolderId() const { return static_cast<uint>(current_folder_id_); }
  const QString& CurrentFolderPath() const { return current_folder_path_text_; }
  int ShownCount() const { return static_cast<int>(visible_thumbnails_.size()); }
  int TotalCount() const;
  QString FilterInfo() const;
  const QString& SqlPreview() const { return sql_preview_; }
  const QString& ValidationError() const { return validation_error_; }
  bool ServiceReady() const { return service_ready_; }
  const QString& ServiceMessage() const { return service_message_; }
  bool ProjectLoading() const { return project_loading_; }
  const QString& ProjectLoadingMessage() const { return project_loading_message_; }
  const QString& TaskStatus() const { return task_status_; }
  int TaskProgress() const { return task_progress_; }
  bool TaskCancelVisible() const { return task_cancel_visible_; }
  const QString& DefaultExportFolder() const { return default_export_folder_; }
  bool ExportInFlight() const { return export_inflight_; }
  const QString& ExportStatus() const { return export_status_; }
  int ExportTotal() const { return export_total_; }
  int ExportCompleted() const { return export_completed_; }
  int ExportSucceeded() const { return export_succeeded_; }
  int ExportFailed() const { return export_failed_; }
  int ExportSkipped() const { return export_skipped_; }
  const QString& ExportErrorSummary() const { return export_error_summary_; }
  bool EditorActive() const { return editor_active_; }
  bool EditorBusy() const { return editor_busy_; }
  uint EditorElementId() const { return static_cast<uint>(editor_element_id_); }
  const QString& EditorTitle() const { return editor_title_; }
  const QString& EditorStatus() const { return editor_status_; }
  const QString& EditorPreviewUrl() const { return editor_preview_url_; }
  QVariantList EditorLutOptions() const { return editor_lut_options_; }
  int EditorLutIndex() const { return editor_lut_index_; }
  double EditorExposure() const { return editor_state_.exposure_; }
  double EditorContrast() const { return editor_state_.contrast_; }
  double EditorSaturation() const { return editor_state_.saturation_; }
  double EditorTint() const { return editor_state_.tint_; }
  double EditorBlacks() const { return editor_state_.blacks_; }
  double EditorWhites() const { return editor_state_.whites_; }
  double EditorShadows() const { return editor_state_.shadows_; }
  double EditorHighlights() const { return editor_state_.highlights_; }
  double EditorSharpen() const { return editor_state_.sharpen_; }
  double EditorClarity() const { return editor_state_.clarity_; }

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
  struct AlbumItem {
    sl_element_id_t element_id    = 0;
    sl_element_id_t parent_folder_id = 0;
    image_id_t      image_id      = 0;
    QString         file_name{};
    QString         camera_model{};
    QString         extension{};
    int             iso           = 0;
    double          aperture      = 0.0;
    double          focal_length  = 0.0;
    QDate           capture_date{};
    QDate           import_date{};
    int             rating        = 0;
    QString         tags{};
    QString         accent{};
    QString         thumb_data_url{};
  };

  struct BuildResult {
    std::optional<FilterNode> node{};
    QString                   error{};
  };

  struct EditorState {
    float       exposure_   = 1.0f;
    float       contrast_   = 1.0f;
    float       saturation_ = 0.0f;
    float       tint_       = 0.0f;
    float       blacks_     = 0.0f;
    float       whites_     = 0.0f;
    float       shadows_    = 0.0f;
    float       highlights_ = 0.0f;
    float       sharpen_    = 0.0f;
    float       clarity_    = 0.0f;
    std::string lut_path_{};
  };

  struct ExistingAlbumEntry {
    sl_element_id_t element_id_ = 0;
    sl_element_id_t parent_folder_id_ = 0;
    image_id_t      image_id_   = 0;
    file_name_t     file_name_{};
  };

  struct ExistingFolderEntry {
    sl_element_id_t      folder_id_   = 0;
    sl_element_id_t      parent_id_   = 0;
    file_name_t          folder_name_{};
    std::filesystem::path folder_path_{};
    int                  depth_       = 0;
  };

  struct ProjectSnapshot {
    std::vector<ExistingAlbumEntry>                    album_entries_{};
    std::vector<ExistingFolderEntry>                   folder_entries_{};
    std::unordered_map<sl_element_id_t, sl_element_id_t> folder_parent_by_id_{};
    std::unordered_map<sl_element_id_t, std::filesystem::path> folder_path_by_id_{};
  };

  using ExportTarget = std::pair<sl_element_id_t, image_id_t>;

  struct ExportQueueBuildResult {
    int     queued_count_  = 0;
    int     skipped_count_ = 0;
    QString first_error_{};
  };

  bool InitializeServices(const std::filesystem::path& dbPath,
                          const std::filesystem::path& metaPath,
                          ProjectOpenMode              openMode,
                          const std::filesystem::path& packagePath = {},
                          const std::filesystem::path& workspaceDir = {});
  bool PersistCurrentProjectState();
  bool PackageCurrentProjectFiles(QString* errorOut = nullptr) const;
  auto CollectProjectSnapshot(const std::shared_ptr<ProjectService>& project) const
      -> ProjectSnapshot;
  void ApplyLoadedProjectEntriesBatch();
  void SetProjectLoadingState(bool loading, const QString& message);
  void ClearProjectData();
  void RebuildFolderView();
  void ApplyFolderSelection(sl_element_id_t folderId, bool emitSignal);
  void RebuildThumbnailView(
      const std::optional<std::unordered_set<sl_element_id_t>>& allowedElementIds);
  void ReleaseVisibleThumbnailPins();
  auto CurrentFolderFsPath() const -> std::filesystem::path;
  void AddImportedEntries(const ImportLogSnapshot& snapshot);
  void AddOrUpdateAlbumItem(sl_element_id_t elementId, image_id_t imageId,
                            const file_name_t& fallbackName,
                            sl_element_id_t parentFolderId);
  void RequestThumbnail(sl_element_id_t elementId, image_id_t imageId);
  void UpdateThumbnailDataUrl(sl_element_id_t elementId, const QString& dataUrl);
  void FinishImport(const ImportResult& result);
  void FinishExport(const std::shared_ptr<std::vector<ExportResult>>& results, int skippedCount);
  void ReapplyCurrentFilters();
  void SetServiceState(bool ready, const QString& message);
  void SetServiceMessageForCurrentProject(const QString& message);
  void ScheduleIdleTaskStateReset(int delayMs);
  void SetExportFailureState(const QString& message);
  void ResetExportProgressState(const QString& status);
  auto CollectExportTargets(const QVariantList& targetEntries) const -> std::vector<ExportTarget>;
  auto BuildExportQueue(const std::vector<ExportTarget>& targets,
                        const std::filesystem::path&   outputDir,
                        ImageFormatType                format,
                        bool                           resizeEnabled,
                        int                            maxLengthSide,
                        int                            quality,
                        ExportFormatOptions::BIT_DEPTH bitDepth,
                        int                            pngCompressionLevel,
                        ExportFormatOptions::TIFF_COMPRESS tiffCompression)
      -> ExportQueueBuildResult;
  void InitializeEditorLuts();
  int LutIndexForPath(const std::string& lutPath) const;
  bool LoadEditorStateFromPipeline();
  void SetupEditorPipeline();
  void ApplyEditorStateToPipeline();
  void QueueEditorRender(RenderType renderType);
  void StartNextEditorRender();
  void PollEditorRender();
  void EnsureEditorPollTimer();
  void FinalizeEditorSession(bool persistChanges);
  bool UpdateEditorPreviewFromBuffer(const std::shared_ptr<ImageBuffer>& buffer);
  void SetEditorAdjustment(float& field, double value, double minValue, double maxValue);
  bool IsThumbnailPinned(sl_element_id_t elementId) const;

  BuildResult BuildFilterNode(FilterOp joinOp) const;
  std::optional<FilterValue> ParseFilterValue(FilterField field, const QString& text,
                                              QString& error) const;
  static std::optional<std::tm> ParseDate(const QString& text);
  bool IsImageInCurrentFolder(const AlbumItem& image) const;
  QString FormatFilterInfo(int shown, int total) const;
  QVariantMap MakeThumbMap(const AlbumItem& image, int index) const;
  void SetTaskState(const QString& status, int progress, bool cancelVisible);

  FilterRuleModel                                     rule_model_;
  std::vector<AlbumItem>                              all_images_{};
  std::unordered_map<sl_element_id_t, size_t>         index_by_element_id_{};
  QVariantList                                        visible_thumbnails_{};
  std::vector<ExistingFolderEntry>                    folder_entries_{};
  std::unordered_map<sl_element_id_t, sl_element_id_t> folder_parent_by_id_{};
  std::unordered_map<sl_element_id_t, std::filesystem::path> folder_path_by_id_{};
  QVariantList                                        folders_{};
  sl_element_id_t                                     current_folder_id_ = 0;
  QString                                             current_folder_path_text_ = "\\";
  std::optional<std::unordered_set<sl_element_id_t>>  active_filter_ids_{};
  FilterOp                                            last_join_op_ = FilterOp::AND;
  std::unordered_map<sl_element_id_t, uint32_t>       thumbnail_pin_ref_counts_{};

  std::shared_ptr<ProjectService>                     project_{};
  std::shared_ptr<PipelineMgmtService>                pipeline_service_{};
  std::shared_ptr<EditHistoryMgmtService>             history_service_{};
  std::shared_ptr<ThumbnailService>                   thumbnail_service_{};
  std::unique_ptr<SleeveFilterService>                filter_service_{};
  std::unique_ptr<ImportServiceImpl>                  import_service_{};
  std::shared_ptr<ExportService>                      export_service_{};
  std::shared_ptr<ImportJob>                          current_import_job_{};
  bool                                                export_inflight_ = false;
  QString                                             default_export_folder_{};
  QString                                             export_status_    = "Ready to export.";
  QString                                             export_error_summary_{};
  int                                                 export_total_     = 0;
  int                                                 export_completed_ = 0;
  int                                                 export_succeeded_ = 0;
  int                                                 export_failed_    = 0;
  int                                                 export_skipped_   = 0;
  std::filesystem::path                               db_path_{};
  std::filesystem::path                               meta_path_{};
  std::filesystem::path                               project_package_path_{};
  std::filesystem::path                               project_workspace_dir_{};

  QString                                             sql_preview_{};
  QString                                             validation_error_{};
  QString                                             service_message_ = "Initializing backend services...";
  bool                                                service_ready_   = false;
  bool                                                project_loading_ = false;
  QString                                             project_loading_message_{};
  uint64_t                                            project_load_request_id_ = 0;
  std::vector<ExistingAlbumEntry>                     pending_project_entries_{};
  std::vector<ExistingFolderEntry>                    pending_folder_entries_{};
  std::unordered_map<sl_element_id_t, sl_element_id_t> pending_folder_parent_by_id_{};
  std::unordered_map<sl_element_id_t, std::filesystem::path> pending_folder_path_by_id_{};
  size_t                                              pending_project_entry_index_ = 0;
  sl_element_id_t                                     import_target_folder_id_ = 0;
  std::filesystem::path                               import_target_folder_path_{};

  QString                                             task_status_         = "No background tasks";
  int                                                 task_progress_       = 0;
  bool                                                task_cancel_visible_ = false;

  bool                                                editor_active_ = false;
  bool                                                editor_busy_   = false;
  sl_element_id_t                                     editor_element_id_ = 0;
  image_id_t                                          editor_image_id_   = 0;
  QString                                             editor_title_{};
  QString                                             editor_status_ = "Select a photo to edit.";
  QString                                             editor_preview_url_{};
  QVariantList                                        editor_lut_options_{};
  std::vector<std::string>                            editor_lut_paths_{};
  int                                                 editor_lut_index_ = 0;
  EditorState                                         editor_state_{};
  EditorState                                         editor_initial_state_{};
  EditorState                                         editor_pending_state_{};
  RenderType                                          editor_pending_render_type_ = RenderType::FAST_PREVIEW;
  bool                                                editor_has_pending_render_  = false;
  bool                                                editor_render_inflight_     = false;
  std::shared_ptr<PipelineGuard>                      editor_pipeline_guard_{};
  std::shared_ptr<PipelineScheduler>                  editor_scheduler_{};
  PipelineTask                                        editor_base_task_{};
  QTimer*                                             editor_poll_timer_ = nullptr;
  std::optional<std::future<std::shared_ptr<ImageBuffer>>> editor_render_future_{};
};

}  // namespace puerhlab::ui
