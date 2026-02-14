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

#include "FilterRuleModel.h"
#include "app/export_service.hpp"
#include "app/history_mgmt_service.hpp"
#include "app/import_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "app/sleeve_filter_service.hpp"
#include "app/thumbnail_service.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "renderer/pipeline_task.hpp"

namespace puerhlab::demo {

class AlbumBackend final : public QObject {
  Q_OBJECT
  Q_PROPERTY(puerhlab::demo::FilterRuleModel* filterRules READ filterRules CONSTANT)
  Q_PROPERTY(QVariantList fieldOptions READ fieldOptions CONSTANT)
  Q_PROPERTY(QVariantList thumbnails READ thumbnails NOTIFY thumbnailsChanged)
  Q_PROPERTY(QVariantList folders READ folders NOTIFY foldersChanged)
  Q_PROPERTY(uint currentFolderId READ currentFolderId NOTIFY folderSelectionChanged)
  Q_PROPERTY(QString currentFolderPath READ currentFolderPath NOTIFY folderSelectionChanged)
  Q_PROPERTY(int shownCount READ shownCount NOTIFY countsChanged)
  Q_PROPERTY(int totalCount READ totalCount NOTIFY countsChanged)
  Q_PROPERTY(QString filterInfo READ filterInfo NOTIFY countsChanged)
  Q_PROPERTY(QString sqlPreview READ sqlPreview NOTIFY sqlPreviewChanged)
  Q_PROPERTY(QString validationError READ validationError NOTIFY validationErrorChanged)
  Q_PROPERTY(bool serviceReady READ serviceReady NOTIFY serviceStateChanged)
  Q_PROPERTY(QString serviceMessage READ serviceMessage NOTIFY serviceStateChanged)
  Q_PROPERTY(bool projectLoading READ projectLoading NOTIFY projectLoadStateChanged)
  Q_PROPERTY(QString projectLoadingMessage READ projectLoadingMessage NOTIFY projectLoadStateChanged)
  Q_PROPERTY(QString taskStatus READ taskStatus NOTIFY taskStateChanged)
  Q_PROPERTY(int taskProgress READ taskProgress NOTIFY taskStateChanged)
  Q_PROPERTY(bool taskCancelVisible READ taskCancelVisible NOTIFY taskStateChanged)
  Q_PROPERTY(QString defaultExportFolder READ defaultExportFolder CONSTANT)
  Q_PROPERTY(bool exportInFlight READ exportInFlight NOTIFY exportStateChanged)
  Q_PROPERTY(QString exportStatus READ exportStatus NOTIFY exportStateChanged)
  Q_PROPERTY(int exportTotal READ exportTotal NOTIFY exportStateChanged)
  Q_PROPERTY(int exportCompleted READ exportCompleted NOTIFY exportStateChanged)
  Q_PROPERTY(int exportSucceeded READ exportSucceeded NOTIFY exportStateChanged)
  Q_PROPERTY(int exportFailed READ exportFailed NOTIFY exportStateChanged)
  Q_PROPERTY(int exportSkipped READ exportSkipped NOTIFY exportStateChanged)
  Q_PROPERTY(QString exportErrorSummary READ exportErrorSummary NOTIFY exportStateChanged)
  Q_PROPERTY(bool editorActive READ editorActive NOTIFY editorStateChanged)
  Q_PROPERTY(bool editorBusy READ editorBusy NOTIFY editorStateChanged)
  Q_PROPERTY(uint editorElementId READ editorElementId NOTIFY editorStateChanged)
  Q_PROPERTY(QString editorTitle READ editorTitle NOTIFY editorStateChanged)
  Q_PROPERTY(QString editorStatus READ editorStatus NOTIFY editorStateChanged)
  Q_PROPERTY(QString editorPreviewUrl READ editorPreviewUrl NOTIFY editorPreviewChanged)
  Q_PROPERTY(QVariantList editorLutOptions READ editorLutOptions NOTIFY editorStateChanged)
  Q_PROPERTY(int editorLutIndex READ editorLutIndex NOTIFY editorStateChanged)
  Q_PROPERTY(double editorExposure READ editorExposure NOTIFY editorStateChanged)
  Q_PROPERTY(double editorContrast READ editorContrast NOTIFY editorStateChanged)
  Q_PROPERTY(double editorSaturation READ editorSaturation NOTIFY editorStateChanged)
  Q_PROPERTY(double editorTint READ editorTint NOTIFY editorStateChanged)
  Q_PROPERTY(double editorBlacks READ editorBlacks NOTIFY editorStateChanged)
  Q_PROPERTY(double editorWhites READ editorWhites NOTIFY editorStateChanged)
  Q_PROPERTY(double editorShadows READ editorShadows NOTIFY editorStateChanged)
  Q_PROPERTY(double editorHighlights READ editorHighlights NOTIFY editorStateChanged)
  Q_PROPERTY(double editorSharpen READ editorSharpen NOTIFY editorStateChanged)
  Q_PROPERTY(double editorClarity READ editorClarity NOTIFY editorStateChanged)

 public:
  explicit AlbumBackend(QObject* parent = nullptr);
  ~AlbumBackend() override;

  FilterRuleModel* filterRules() { return &rule_model_; }
  QVariantList fieldOptions() const;
  QVariantList thumbnails() const { return visible_thumbnails_; }
  QVariantList folders() const { return folders_; }
  uint currentFolderId() const { return static_cast<uint>(current_folder_id_); }
  const QString& currentFolderPath() const { return current_folder_path_text_; }
  int shownCount() const { return static_cast<int>(visible_thumbnails_.size()); }
  int totalCount() const;
  QString filterInfo() const;
  const QString& sqlPreview() const { return sql_preview_; }
  const QString& validationError() const { return validation_error_; }
  bool serviceReady() const { return service_ready_; }
  const QString& serviceMessage() const { return service_message_; }
  bool projectLoading() const { return project_loading_; }
  const QString& projectLoadingMessage() const { return project_loading_message_; }
  const QString& taskStatus() const { return task_status_; }
  int taskProgress() const { return task_progress_; }
  bool taskCancelVisible() const { return task_cancel_visible_; }
  const QString& defaultExportFolder() const { return default_export_folder_; }
  bool exportInFlight() const { return export_inflight_; }
  const QString& exportStatus() const { return export_status_; }
  int exportTotal() const { return export_total_; }
  int exportCompleted() const { return export_completed_; }
  int exportSucceeded() const { return export_succeeded_; }
  int exportFailed() const { return export_failed_; }
  int exportSkipped() const { return export_skipped_; }
  const QString& exportErrorSummary() const { return export_error_summary_; }
  bool editorActive() const { return editor_active_; }
  bool editorBusy() const { return editor_busy_; }
  uint editorElementId() const { return static_cast<uint>(editor_element_id_); }
  const QString& editorTitle() const { return editor_title_; }
  const QString& editorStatus() const { return editor_status_; }
  const QString& editorPreviewUrl() const { return editor_preview_url_; }
  QVariantList editorLutOptions() const { return editor_lut_options_; }
  int editorLutIndex() const { return editor_lut_index_; }
  double editorExposure() const { return editor_state_.exposure_; }
  double editorContrast() const { return editor_state_.contrast_; }
  double editorSaturation() const { return editor_state_.saturation_; }
  double editorTint() const { return editor_state_.tint_; }
  double editorBlacks() const { return editor_state_.blacks_; }
  double editorWhites() const { return editor_state_.whites_; }
  double editorShadows() const { return editor_state_.shadows_; }
  double editorHighlights() const { return editor_state_.highlights_; }
  double editorSharpen() const { return editor_state_.sharpen_; }
  double editorClarity() const { return editor_state_.clarity_; }

  Q_INVOKABLE void addRule();
  Q_INVOKABLE void removeRule(int index);
  Q_INVOKABLE void setRuleField(int index, int fieldValue);
  Q_INVOKABLE void setRuleOp(int index, int opValue);
  Q_INVOKABLE void setRuleValue(int index, const QString& value);
  Q_INVOKABLE void setRuleValue2(int index, const QString& value);

  Q_INVOKABLE void applyFilters(int joinOpValue);
  Q_INVOKABLE void clearFilters();
  Q_INVOKABLE QVariantList compareOptionsForField(int fieldValue) const;
  Q_INVOKABLE QString placeholderForField(int fieldValue) const;
  Q_INVOKABLE void selectFolder(uint folderId);
  Q_INVOKABLE void createFolder(const QString& folderName);
  Q_INVOKABLE void deleteFolder(uint folderId);

  Q_INVOKABLE void startImport(const QStringList& fileUrlsOrPaths);
  Q_INVOKABLE void cancelImport();
  Q_INVOKABLE bool loadProject(const QString& metaFileUrlOrPath);
  Q_INVOKABLE bool createProjectInFolder(const QString& folderUrlOrPath);
  Q_INVOKABLE bool createProjectInFolderNamed(const QString& folderUrlOrPath,
                                              const QString& projectName);
  Q_INVOKABLE bool saveProject();
  Q_INVOKABLE void startExport(const QString& outputDirUrlOrPath);
  Q_INVOKABLE void startExportWithOptions(const QString& outputDirUrlOrPath,
                                          const QString& formatName, bool resizeEnabled,
                                          int maxLengthSide, int quality, int bitDepth,
                                          int pngCompressionLevel,
                                          const QString& tiffCompression);
  Q_INVOKABLE void startExportWithOptionsForTargets(const QString& outputDirUrlOrPath,
                                                    const QString& formatName,
                                                    bool resizeEnabled, int maxLengthSide,
                                                    int quality, int bitDepth,
                                                    int pngCompressionLevel,
                                                    const QString& tiffCompression,
                                                    const QVariantList& targetEntries);
  Q_INVOKABLE void resetExportState();
  Q_INVOKABLE void openEditor(uint elementId, uint imageId);
  Q_INVOKABLE void closeEditor();
  Q_INVOKABLE void resetEditorAdjustments();
  Q_INVOKABLE void requestEditorFullPreview();
  Q_INVOKABLE void setEditorLutIndex(int index);
  Q_INVOKABLE void setEditorExposure(double value);
  Q_INVOKABLE void setEditorContrast(double value);
  Q_INVOKABLE void setEditorSaturation(double value);
  Q_INVOKABLE void setEditorTint(double value);
  Q_INVOKABLE void setEditorBlacks(double value);
  Q_INVOKABLE void setEditorWhites(double value);
  Q_INVOKABLE void setEditorShadows(double value);
  Q_INVOKABLE void setEditorHighlights(double value);
  Q_INVOKABLE void setEditorSharpen(double value);
  Q_INVOKABLE void setEditorClarity(double value);
  Q_INVOKABLE void setThumbnailVisible(uint elementId, uint imageId, bool visible);

signals:
  void thumbnailsChanged();
  void thumbnailUpdated(uint elementId, const QString& dataUrl);
  void countsChanged();
  void sqlPreviewChanged();
  void validationErrorChanged();
  void serviceStateChanged();
  void taskStateChanged();
  void exportStateChanged();
  void editorStateChanged();
  void editorPreviewChanged();
  void projectChanged();
  void projectLoadStateChanged();
  void foldersChanged();
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

  bool initializeServices(const std::filesystem::path& dbPath,
                          const std::filesystem::path& metaPath,
                          ProjectOpenMode              openMode,
                          const std::filesystem::path& packagePath = {},
                          const std::filesystem::path& workspaceDir = {});
  bool persistCurrentProjectState();
  bool packageCurrentProjectFiles(QString* errorOut = nullptr) const;
  auto collectProjectSnapshot(const std::shared_ptr<ProjectService>& project) const
      -> ProjectSnapshot;
  void applyLoadedProjectEntriesBatch();
  void setProjectLoadingState(bool loading, const QString& message);
  void clearProjectData();
  void rebuildFolderView();
  void applyFolderSelection(sl_element_id_t folderId, bool emitSignal);
  void rebuildThumbnailView(
      const std::optional<std::unordered_set<sl_element_id_t>>& allowedElementIds);
  void releaseVisibleThumbnailPins();
  auto currentFolderFsPath() const -> std::filesystem::path;
  void addImportedEntries(const ImportLogSnapshot& snapshot);
  void addOrUpdateAlbumItem(sl_element_id_t elementId, image_id_t imageId,
                            const file_name_t& fallbackName,
                            sl_element_id_t parentFolderId);
  void requestThumbnail(sl_element_id_t elementId, image_id_t imageId);
  void updateThumbnailDataUrl(sl_element_id_t elementId, const QString& dataUrl);
  void finishImport(const ImportResult& result);
  void finishExport(const std::shared_ptr<std::vector<ExportResult>>& results, int skippedCount);
  void reapplyCurrentFilters();
  void setServiceState(bool ready, const QString& message);
  void setServiceMessageForCurrentProject(const QString& message);
  void scheduleIdleTaskStateReset(int delayMs);
  void setExportFailureState(const QString& message);
  void resetExportProgressState(const QString& status);
  auto collectExportTargets(const QVariantList& targetEntries) const -> std::vector<ExportTarget>;
  auto buildExportQueue(const std::vector<ExportTarget>& targets,
                        const std::filesystem::path&   outputDir,
                        ImageFormatType                format,
                        bool                           resizeEnabled,
                        int                            maxLengthSide,
                        int                            quality,
                        ExportFormatOptions::BIT_DEPTH bitDepth,
                        int                            pngCompressionLevel,
                        ExportFormatOptions::TIFF_COMPRESS tiffCompression)
      -> ExportQueueBuildResult;
  void initializeEditorLuts();
  int lutIndexForPath(const std::string& lutPath) const;
  bool loadEditorStateFromPipeline();
  void setupEditorPipeline();
  void applyEditorStateToPipeline();
  void queueEditorRender(RenderType renderType);
  void startNextEditorRender();
  void pollEditorRender();
  void ensureEditorPollTimer();
  void finalizeEditorSession(bool persistChanges);
  bool updateEditorPreviewFromBuffer(const std::shared_ptr<ImageBuffer>& buffer);
  void setEditorAdjustment(float& field, double value, double minValue, double maxValue);
  bool isThumbnailPinned(sl_element_id_t elementId) const;

  BuildResult buildFilterNode(FilterOp joinOp) const;
  std::optional<FilterValue> parseFilterValue(FilterField field, const QString& text,
                                              QString& error) const;
  static std::optional<std::tm> parseDate(const QString& text);
  bool isImageInCurrentFolder(const AlbumItem& image) const;
  QString formatFilterInfo(int shown, int total) const;
  QVariantMap makeThumbMap(const AlbumItem& image, int index) const;
  void setTaskState(const QString& status, int progress, bool cancelVisible);

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

}  // namespace puerhlab::demo
