//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/album_backend/album_backend.hpp"

#include "app/project_package_service.hpp"
#include "edit/operators/utils/color_utils.hpp"
#include "ui/alcedo_main/album_backend/path_utils.hpp"

#include <QDesktopServices>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QInputDialog>
#include <QLineEdit>
#include <QSettings>
#include <QStandardPaths>
#include <QTimer>
#include <QUrl>

#include <algorithm>

#include "image/image.hpp"

namespace alcedo::ui {

using namespace album_util;

#define PL_TEXT(text, ...)                                                    \
  i18n::MakeLocalizedText(ALCEDO_I18N_CONTEXT,                              \
                          QT_TRANSLATE_NOOP(ALCEDO_I18N_CONTEXT, text)      \
                              __VA_OPT__(, ) __VA_ARGS__)

namespace {

constexpr auto kRecentProjectsKey = "projects/recent";
constexpr int  kMaxRecentProjects = 12;

auto IsHdrExportEotf(const ColorUtils::EOTF eotf) -> bool {
  return eotf == ColorUtils::EOTF::ST2084 || eotf == ColorUtils::EOTF::HLG;
}

auto NormalizeRecentProjectPath(const std::filesystem::path& projectPath) -> QString {
  if (projectPath.empty()) {
    return {};
  }
  return QDir::cleanPath(QDir::fromNativeSeparators(
      QFileInfo(PathToQString(projectPath.lexically_normal())).absoluteFilePath()));
}

auto BuildRecentProjectEntry(const QString& normalizedPath, qint64 lastOpenedMs) -> QVariantMap {
  const QFileInfo info(normalizedPath);
  QString         display_name = info.completeBaseName();
  if (display_name.isEmpty()) {
    display_name = info.fileName();
  }
  if (display_name.isEmpty()) {
    display_name = normalizedPath;
  }

  return QVariantMap{
      {"path", normalizedPath},
      {"name", display_name},
      {"folderPath", info.absolutePath()},
      {"lastOpenedMs", lastOpenedMs},
  };
}

}  // namespace

// ── Constructor / Destructor ────────────────────────────────────────────────

AlbumBackend::AlbumBackend(QObject* parent)
    : QObject(parent),
      project_handler_(*this),
      thumb_(*this),
      folder_ctrl_(*this),
      image_ctrl_(*this),
      stats_(*this),
      import_export_(*this),
      nikon_he_recovery_(*this),
      editor_(*this) {
  LoadRecentProjectsFromSettings();
  QObject::connect(&i18n::TranslationNotifier::Instance(), &i18n::TranslationNotifier::LanguageChanged,
                   this, &AlbumBackend::RefreshTranslations);
  editor_.InitializeEditorLuts();
  SetServiceState(false, PL_TEXT("Select a project: load a .alcd package or metadata JSON, "
                                 "or create a new packed project."));
  task_status_text_ = PL_TEXT("Open or create a project to begin.");
}

AlbumBackend::~AlbumBackend() {
  try {
    thumb_.ReleaseVisibleThumbnailPins();
    editor_.FinalizeEditorSession(true);
    auto job = import_export_.current_import_job();
    if (job) {
      job->canceled_.store(true);
    }
    auto psvc = project_handler_.pipeline_service();
    if (psvc) {
      psvc->Sync();
    }
    if (project_handler_.PersistCurrentProjectState()) {
      QString ignored_error;
      (void)project_handler_.PackageCurrentProjectFiles(&ignored_error);
    }
    CleanupWorkspaceDirectory(project_handler_.workspace_dir());
  } catch (...) {
  }
}

// ── Q_PROPERTY getters that compute ─────────────────────────────────────────

auto AlbumBackend::FilterInfo() const -> QString {
  return stats_.FormatPhotoInfo(ShownCount(), TotalCount());
}

int AlbumBackend::TotalCount() const {
  return static_cast<int>(view_state_.all_images_.size());
}

// ── Q_INVOKABLE: Folder delegation ──────────────────────────────────────────

void AlbumBackend::SelectFolder(uint folderId) {
  folder_ctrl_.SelectFolder(folderId);
}

void AlbumBackend::CreateFolder(const QString& folderName) { folder_ctrl_.CreateFolder(folderName); }
void AlbumBackend::DeleteFolder(uint folderId) { folder_ctrl_.DeleteFolder(folderId); }
auto AlbumBackend::DeleteImages(const QVariantList& targetEntries) -> QVariantMap {
  return image_ctrl_.DeleteImages(targetEntries);
}
auto AlbumBackend::GetImageDetails(uint elementId, uint imageId) -> QVariantMap {
  return image_ctrl_.GetImageDetails(elementId, imageId);
}
bool AlbumBackend::OpenDirectoryInFileManager(const QString& dirUrlOrPath) {
  const auto dir_path_opt = InputToPath(dirUrlOrPath);
  if (!dir_path_opt.has_value()) {
    SetServiceMessageForCurrentProject(PL_TEXT("Source directory is unavailable."));
    return false;
  }

  const std::filesystem::path dir_path = dir_path_opt.value().lexically_normal();
  std::error_code ec;
  if (!std::filesystem::exists(dir_path, ec) || ec || !std::filesystem::is_directory(dir_path, ec) ||
      ec) {
    SetServiceMessageForCurrentProject(PL_TEXT("Source directory is unavailable."));
    return false;
  }

  if (!QDesktopServices::openUrl(QUrl::fromLocalFile(PathToQString(dir_path)))) {
    SetServiceMessageForCurrentProject(PL_TEXT("Failed to open source directory."));
    return false;
  }

  return true;
}

// ── Q_INVOKABLE: Stats-bar filter ──────────────────────────────────────────

void AlbumBackend::ToggleStatsFilter(const QString& category, const QString& label) {
  stats_.ToggleFilter(category, label);
  stats_.RebuildThumbnailView();
  emit StatsFilterChanged();
}

void AlbumBackend::ClearStatsFilter() {
  stats_.ClearFilters();
  stats_.RebuildThumbnailView();
  emit StatsFilterChanged();
}

// ── Q_INVOKABLE: Import / export delegation ─────────────────────────────────

void AlbumBackend::StartImport(const QStringList& fileUrlsOrPaths) { import_export_.StartImport(fileUrlsOrPaths); }
void AlbumBackend::CancelImport() { import_export_.CancelImport(); }

void AlbumBackend::StartExport(const QString& outputDirUrlOrPath) {
  import_export_.StartExport(outputDirUrlOrPath);
}

void AlbumBackend::StartExportWithOptions(const QString& outputDirUrlOrPath,
                                          const QString& formatName,
                                          const QString& hdrExportMode, bool resizeEnabled,
                                          int maxLengthSide, int quality, int bitDepth,
                                          int pngCompressionLevel,
                                          const QString& tiffCompression) {
  import_export_.StartExportWithOptions(outputDirUrlOrPath, formatName, hdrExportMode,
                                        resizeEnabled, maxLengthSide, quality, bitDepth,
                                        pngCompressionLevel, tiffCompression);
}

void AlbumBackend::StartExportWithOptionsForTargets(const QString& outputDirUrlOrPath,
                                                    const QString& formatName,
                                                    const QString& hdrExportMode,
                                                    bool resizeEnabled, int maxLengthSide,
                                                    int quality, int bitDepth,
                                                    int pngCompressionLevel,
                                                    const QString& tiffCompression,
                                                    const QVariantList& targetEntries) {
  import_export_.StartExportWithOptionsForTargets(outputDirUrlOrPath, formatName, hdrExportMode,
                                                  resizeEnabled, maxLengthSide, quality, bitDepth,
                                                  pngCompressionLevel, tiffCompression,
                                                  targetEntries);
}

void AlbumBackend::ResetExportState() { import_export_.ResetExportState(); }
bool AlbumBackend::CanUseHdrExportForTargets(const QVariantList& targetEntries) const {
  if (project_handler_.project_loading()) {
    return false;
  }

  const auto& psvc = project_handler_.pipeline_service();
  auto        proj = project_handler_.project();
  if (!psvc || !proj) {
    return false;
  }

  const auto targets = import_export_.CollectExportTargets(targetEntries);
  if (targets.empty()) {
    return false;
  }

  for (const auto& [elementId, imageId] : targets) {
    (void)imageId;

    bool hdr_eotf = false;
    try {
      auto pipeline_guard = psvc->LoadPipeline(elementId);
      if (!pipeline_guard || !pipeline_guard->pipeline_) {
        return false;
      }
      hdr_eotf = IsHdrExportEotf(
          pipeline_guard->pipeline_->GetGlobalParams().to_output_params_.eotf_);
      psvc->SavePipeline(pipeline_guard);
    } catch (...) {
      return false;
    }

    if (!hdr_eotf) {
      return false;
    }
  }

  return true;
}
void AlbumBackend::BrowseNikonHeConverter() { nikon_he_recovery_.BrowseConverter(); }
void AlbumBackend::StartNikonHeConversion() { nikon_he_recovery_.StartConversion(); }
void AlbumBackend::ExitNikonHeRecovery() { nikon_he_recovery_.ExitRecovery(); }

// ── Q_INVOKABLE: Editor delegation ──────────────────────────────────────────

void AlbumBackend::OpenEditor(uint elementId, uint imageId) { editor_.OpenEditor(elementId, imageId); }
void AlbumBackend::CloseEditor() { editor_.CloseEditor(); }
void AlbumBackend::ResetEditorAdjustments() { editor_.ResetEditorAdjustments(); }
void AlbumBackend::RequestEditorFullPreview() { editor_.RequestEditorFullPreview(); }
void AlbumBackend::SetEditorLutIndex(int index) { editor_.SetEditorLutIndex(index); }
void AlbumBackend::SetEditorExposure(double value) { editor_.SetEditorExposure(value); }
void AlbumBackend::SetEditorContrast(double value) { editor_.SetEditorContrast(value); }
void AlbumBackend::SetEditorSaturation(double value) { editor_.SetEditorSaturation(value); }
void AlbumBackend::SetEditorTint(double value) { editor_.SetEditorTint(value); }
void AlbumBackend::SetEditorBlacks(double value) { editor_.SetEditorBlacks(value); }
void AlbumBackend::SetEditorWhites(double value) { editor_.SetEditorWhites(value); }
void AlbumBackend::SetEditorShadows(double value) { editor_.SetEditorShadows(value); }
void AlbumBackend::SetEditorHighlights(double value) { editor_.SetEditorHighlights(value); }
void AlbumBackend::SetEditorSharpen(double value) { editor_.SetEditorSharpen(value); }
void AlbumBackend::SetEditorClarity(double value) { editor_.SetEditorClarity(value); }

// ── Q_INVOKABLE: Thumbnail delegation ───────────────────────────────────────

void AlbumBackend::SetThumbnailVisible(uint elementId, uint imageId, bool visible) {
  thumb_.SetThumbnailVisible(elementId, imageId, visible);
}

// ── Q_INVOKABLE: Project I/O ────────────────────────────────────────────────

bool AlbumBackend::PromptAndLoadProject() {
  const QString start_dir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
  const QString selected_path =
      QFileDialog::getOpenFileName(nullptr, tr("Select Project Package or Metadata JSON"), start_dir,
                                   tr("Packed Project (*.alcd);;Project Metadata (*.json);;All Files (*)"));
  if (selected_path.isEmpty()) {
    return false;
  }
  return LoadProject(QUrl::fromLocalFile(selected_path).toString());
}

bool AlbumBackend::PromptAndCreateProject() {
  const QString start_dir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
  const QString selected_dir =
      QFileDialog::getExistingDirectory(nullptr, tr("Select Parent Folder for New Project"), start_dir,
                                        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
  if (selected_dir.isEmpty()) {
    return false;
  }

  bool accepted = false;
  QString project_name = QInputDialog::getText(
      nullptr, tr("Name New Project"), tr("Project name"), QLineEdit::Normal,
      QStringLiteral("album_editor_project"), &accepted);
  if (!accepted) {
    return false;
  }

  project_name = project_name.trimmed();
  if (project_name.isEmpty()) {
    SetServiceMessageForCurrentProject(PL_TEXT("Project name cannot be empty."));
    return false;
  }

  return CreateProjectInFolderNamed(QUrl::fromLocalFile(selected_dir).toString(), project_name);
}

bool AlbumBackend::LoadProject(const QString& metaFileUrlOrPath) {
  if (project_handler_.project_loading()) {
    SetServiceMessageForCurrentProject(PL_TEXT("A project load is already in progress."));
    return false;
  }

  const auto project_path_opt = InputToPath(metaFileUrlOrPath);
  if (!project_path_opt.has_value()) {
    SetServiceMessageForCurrentProject(PL_TEXT("Select a valid project file."));
    return false;
  }

  const auto project_path = project_path_opt.value();
  std::error_code ec;
  if (!std::filesystem::is_regular_file(project_path, ec) || ec) {
    RemoveRecentProject(project_path);
    SetServiceMessageForCurrentProject(PL_TEXT("Project file was not found."));
    return false;
  }

  ProjectPackageService package_service;

  if (package_service.IsPackedProjectPath(project_path) ||
      package_service.IsPackedProjectFile(project_path)) {
    const QString project_name = QFileInfo(PathToQString(project_path)).completeBaseName();
    std::filesystem::path workspace_dir;
    QString               workspace_error;
    if (!package_service.CreateProjectWorkspace(project_name, &workspace_dir, &workspace_error)) {
      SetServiceMessageForCurrentProject(
          workspace_error.isEmpty()
              ? PL_TEXT("Failed to prepare project temp workspace.")
              : PL_TEXT("%1", workspace_error));
      return false;
    }

    std::filesystem::path unpacked_db_path;
    std::filesystem::path unpacked_meta_path;
    QString               unpack_error;
    if (!package_service.UnpackProjectToWorkspace(project_path, workspace_dir, project_name,
                                                  &unpacked_db_path, &unpacked_meta_path,
                                                  &unpack_error)) {
      CleanupWorkspaceDirectory(workspace_dir);
      SetServiceMessageForCurrentProject(
          unpack_error.isEmpty() ? PL_TEXT("Failed to unpack project package.")
                                 : PL_TEXT("%1", unpack_error));
      return false;
    }

    return project_handler_.InitializeServices(unpacked_db_path, unpacked_meta_path,
                                               ProjectOpenMode::kLoadExisting,
                                               project_path, workspace_dir, project_path);
  }

  if (!package_service.IsMetadataJsonPath(project_path)) {
    SetServiceMessageForCurrentProject(
        PL_TEXT("Unsupported project format. Choose a .json or .alcd file."));
    return false;
  }

  const auto db_hint_path =
      project_path.parent_path() / (project_path.stem().wstring() + L".db");
  return project_handler_.InitializeServices(db_hint_path, project_path,
                                             ProjectOpenMode::kLoadExisting,
                                             package_service.BuildBundlePathFromMetaPath(project_path),
                                             {}, project_path);
}

bool AlbumBackend::CreateProjectInFolder(const QString& folderUrlOrPath) {
  return CreateProjectInFolderNamed(folderUrlOrPath, "album_editor_project");
}

bool AlbumBackend::CreateProjectInFolderNamed(const QString& folderUrlOrPath,
                                              const QString& projectName) {
  if (project_handler_.project_loading()) {
    SetServiceMessageForCurrentProject(PL_TEXT("A project load is already in progress."));
    return false;
  }

  const auto folder_path_opt = InputToPath(folderUrlOrPath);
  if (!folder_path_opt.has_value()) {
    SetServiceMessageForCurrentProject(PL_TEXT("Select a valid folder for the new project."));
    return false;
  }

  ProjectPackageService package_service;

  QString build_error;
  const auto packed_path_opt =
      package_service.BuildUniquePackedProjectPath(folder_path_opt.value(), projectName,
                                                   &build_error);
  if (!packed_path_opt.has_value()) {
    SetServiceMessageForCurrentProject(
        build_error.isEmpty() ? PL_TEXT("Failed to prepare project package path in selected folder.")
                              : PL_TEXT("%1", build_error));
    return false;
  }

  std::filesystem::path workspace_dir;
  QString               workspace_error;
  if (!package_service.CreateProjectWorkspace(projectName, &workspace_dir, &workspace_error)) {
    SetServiceMessageForCurrentProject(
        workspace_error.isEmpty() ? PL_TEXT("Failed to prepare project temp workspace.")
                                  : PL_TEXT("%1", workspace_error));
    return false;
  }

  const auto runtime_pair = package_service.BuildRuntimeProjectPair(workspace_dir, projectName);
  const bool started =
      project_handler_.InitializeServices(runtime_pair.first, runtime_pair.second,
                                          ProjectOpenMode::kCreateNew,
                                          packed_path_opt.value(), workspace_dir,
                                          packed_path_opt.value());
  if (!started) {
    CleanupWorkspaceDirectory(workspace_dir);
  }
  return started;
}

bool AlbumBackend::SaveProject() {
  if (project_handler_.project_loading()) {
    SetServiceMessageForCurrentProject(PL_TEXT("Please wait until project loading finishes."));
    return false;
  }

  if (!project_handler_.project() || project_handler_.meta_path().empty()) {
    SetServiceState(false, PL_TEXT("No project is loaded yet."));
    SetTaskState(PL_TEXT("No project to save."), 0, false);
    return false;
  }

  if (editor_.editor_active()) {
    editor_.FinalizeEditorSession(true);
  }

  if (!project_handler_.PersistCurrentProjectState()) {
    SetServiceMessageForCurrentProject(PL_TEXT("Project save failed."));
    SetTaskState(PL_TEXT("Project save failed."), 0, false);
    return false;
  }

  QString package_error;
  if (!project_handler_.PackageCurrentProjectFiles(&package_error)) {
    SetServiceMessageForCurrentProject(
        package_error.isEmpty() ? PL_TEXT("Project saved, but packing failed.")
                                : PL_TEXT("%1", package_error));
    SetTaskState(PL_TEXT("Project packing failed."), 0, false);
    return false;
  }

  SetServiceMessageForCurrentProject(
      project_handler_.package_path().empty()
          ? PL_TEXT("Project saved to %1", PathToQString(project_handler_.meta_path()))
          : PL_TEXT("Project saved and packed to %1",
                    PathToQString(project_handler_.package_path())));
  SetTaskState(project_handler_.package_path().empty() ? PL_TEXT("Project saved.")
                                                       : PL_TEXT("Project saved and packed."),
               100, false);
  ScheduleIdleTaskStateReset(1200);
  return true;
}

// ── Shared internal methods ─────────────────────────────────────────────────

void AlbumBackend::SetServiceState(bool ready, const i18n::LocalizedText& message) {
  if (service_ready_ == ready && service_message_text_.source_ == message.source_ &&
      service_message_text_.args_ == message.args_) {
    return;
  }
  service_ready_   = ready;
  service_message_text_ = message;
  emit ServiceStateChanged();
}

void AlbumBackend::SetServiceMessageForCurrentProject(const i18n::LocalizedText& message) {
  SetServiceState(project_handler_.project() != nullptr, message);
}

void AlbumBackend::ScheduleIdleTaskStateReset(int delayMs) {
  QTimer::singleShot(std::max(delayMs, 0), this, [this]() {
    if (!import_export_.export_inflight() && !task_cancel_visible_) {
      SetTaskState(PL_TEXT("No background tasks"), 0, false);
    }
  });
}

void AlbumBackend::SetTaskState(const i18n::LocalizedText& status, int progress, bool cancelVisible) {
  task_status_text_    = status;
  task_progress_       = std::clamp(progress, 0, 100);
  task_cancel_visible_ = cancelVisible;
  emit TaskStateChanged();
}

void AlbumBackend::RefreshTranslations() {
  if (!folder_ctrl_.folder_entries().empty()) {
    folder_ctrl_.RebuildFolderView();
  }
  if (!view_state_.all_images_.empty()) {
    stats_.RebuildThumbnailView();
  }
  stats_.RefreshStats();
  emit ServiceStateChanged();
  emit TaskStateChanged();
  emit ProjectLoadStateChanged();
  emit ImportStateChanged();
  emit importStateChanged();
  emit ExportStateChanged();
  emit exportStateChanged();
  emit RecentProjectsChanged();
  emit EditorStateChanged();
}

void AlbumBackend::LoadRecentProjectsFromSettings() {
  const QVariantList stored_entries = QSettings{}.value(QLatin1String(kRecentProjectsKey)).toList();

  QVariantList normalized_entries;
  QStringList  seen_paths;
  bool         changed = false;

  for (const QVariant& entry_variant : stored_entries) {
    const QVariantMap entry_map = entry_variant.toMap();
    const QString raw_path = entry_map.value(QStringLiteral("path")).toString().trimmed();
    if (raw_path.isEmpty()) {
      changed = true;
      continue;
    }

    const QString normalized_path =
        NormalizeRecentProjectPath(std::filesystem::path(raw_path.toStdWString()));
    if (normalized_path.isEmpty() || seen_paths.contains(normalized_path)) {
      changed = true;
      continue;
    }

    const QFileInfo info(normalized_path);
    if (!info.exists() || !info.isFile()) {
      changed = true;
      continue;
    }

    const qint64 last_opened_ms =
        entry_map.value(QStringLiteral("lastOpenedMs")).toLongLong();
    normalized_entries.push_back(
        BuildRecentProjectEntry(normalized_path, last_opened_ms > 0 ? last_opened_ms : 0));
    seen_paths.push_back(normalized_path);

    if (normalized_entries.size() >= kMaxRecentProjects) {
      if (stored_entries.size() > normalized_entries.size()) {
        changed = true;
      }
      break;
    }
  }

  recent_projects_ = normalized_entries;
  if (changed || stored_entries != normalized_entries) {
    PersistRecentProjects();
  }
}

void AlbumBackend::PersistRecentProjects() const {
  QSettings{}.setValue(QLatin1String(kRecentProjectsKey), recent_projects_);
}

void AlbumBackend::RegisterRecentProject(const std::filesystem::path& projectPath) {
  const QString normalized_path = NormalizeRecentProjectPath(projectPath);
  if (normalized_path.isEmpty()) {
    return;
  }

  QVariantList next_entries;
  next_entries.push_back(
      BuildRecentProjectEntry(normalized_path, QDateTime::currentDateTimeUtc().toMSecsSinceEpoch()));

  for (const QVariant& entry_variant : recent_projects_) {
    const QVariantMap entry_map = entry_variant.toMap();
    const QString entry_path = entry_map.value(QStringLiteral("path")).toString();
    if (entry_path.isEmpty() || entry_path == normalized_path) {
      continue;
    }
    next_entries.push_back(entry_map);
    if (next_entries.size() >= kMaxRecentProjects) {
      break;
    }
  }

  if (recent_projects_ == next_entries) {
    return;
  }

  recent_projects_ = next_entries;
  PersistRecentProjects();
  emit RecentProjectsChanged();
}

void AlbumBackend::RemoveRecentProject(const std::filesystem::path& projectPath) {
  const QString normalized_path = NormalizeRecentProjectPath(projectPath);
  if (normalized_path.isEmpty()) {
    return;
  }

  QVariantList next_entries;
  next_entries.reserve(recent_projects_.size());
  for (const QVariant& entry_variant : recent_projects_) {
    const QVariantMap entry_map = entry_variant.toMap();
    if (entry_map.value(QStringLiteral("path")).toString() == normalized_path) {
      continue;
    }
    next_entries.push_back(entry_map);
  }

  if (recent_projects_ == next_entries) {
    return;
  }

  recent_projects_ = next_entries;
  PersistRecentProjects();
  emit RecentProjectsChanged();
}

void AlbumBackend::ReloadFolderTree(const std::filesystem::path& preferredFolderPath) {
  auto proj = project_handler_.project();
  if (!proj) {
    folder_ctrl_.ClearState();
    emit FoldersChanged();
    emit FolderSelectionChanged();
    emit folderSelectionChanged();
    return;
  }

  auto browse = proj->GetAlbumBrowseService();
  if (!browse) {
    return;
  }

  folder_ctrl_.ReloadTree(preferredFolderPath.empty() ? folder_ctrl_.current_folder_path()
                                                      : preferredFolderPath);
}

void AlbumBackend::ReloadCurrentFolder() {
  thumb_.ReleaseVisibleThumbnailPins();

  view_state_.all_images_.clear();
  view_state_.visible_thumbnails_.clear();
  emit ThumbnailsChanged();
  emit thumbnailsChanged();
  emit CountsChanged();

  auto proj = project_handler_.project();
  if (!proj) {
    stats_.RebuildThumbnailView();
    stats_.RefreshStats();
    return;
  }

  auto browse = proj->GetAlbumBrowseService();
  if (!browse) {
    stats_.RebuildThumbnailView();
    stats_.RefreshStats();
    return;
  }

  const auto files = browse->ListFilesInFolder(folder_ctrl_.CurrentFolderFsPath());
  for (const auto& file : files) {
    AddOrUpdateAlbumItem(file.element_id_, file.image_id_, file.file_name_, file.file_path_);
  }

  stats_.RebuildThumbnailView();
  stats_.RefreshStats();
}

void AlbumBackend::AddOrUpdateAlbumItem(sl_element_id_t elementId, image_id_t imageId,
                                        const file_name_t& fallbackName,
                                        const std::filesystem::path& filePath) {
  AlbumItem* item = FindAlbumItem(elementId);

  if (!item) {
    AlbumItem next;
    next.element_id = elementId;
    next.image_id   = imageId;
    next.file_path_ = filePath;
    next.file_name  = WStringToQString(fallbackName);
    next.extension  = ExtensionFromFileName(next.file_name);
    next.accent     = AccentForIndex(view_state_.all_images_.size());

    view_state_.all_images_.push_back(std::move(next));
    item = &view_state_.all_images_.back();
  }

  if (!item) return;

  item->element_id       = elementId;
  item->image_id         = imageId;
  item->file_path_       = filePath;

  auto proj = project_handler_.project();
  if (proj) {
    try {
      proj->GetImagePoolService()->Read<void>(
          imageId,
          [item](std::shared_ptr<Image> image) {
            if (!image) return;
            if (!image->image_name_.empty()) {
              item->file_name = WStringToQString(image->image_name_);
            }
            if (!image->image_path_.empty()) {
              item->extension = ExtensionUpper(image->image_path_);
            }

            const auto& exif    = image->exif_display_;
            item->camera_model  = QString::fromUtf8(exif.model_.c_str());
            item->lens          = QString::fromUtf8(exif.lens_.c_str());
            item->iso           = static_cast<int>(exif.iso_);
            item->aperture      = static_cast<double>(exif.aperture_);
            item->focal_length  = static_cast<double>(exif.focal_);
            item->rating        = exif.rating_;
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

auto AlbumBackend::FindAlbumItem(sl_element_id_t elementId) -> AlbumItem* {
  for (auto& item : view_state_.all_images_) {
    if (item.element_id == elementId) {
      return &item;
    }
  }
  return nullptr;
}

auto AlbumBackend::FindAlbumItem(sl_element_id_t elementId) const -> const AlbumItem* {
  for (const auto& item : view_state_.all_images_) {
    if (item.element_id == elementId) {
      return &item;
    }
  }
  return nullptr;
}

}  // namespace alcedo::ui

#undef PL_TEXT
