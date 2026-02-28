#include "ui/puerhlab_main/album_backend/album_backend.hpp"

#include "ui/puerhlab_main/album_backend/path_utils.hpp"
#include "ui/puerhlab_main/album_backend/packed_project.hpp"

#include <QDir>
#include <QFileInfo>
#include <QTimer>

#include <algorithm>

#include "image/image.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filesystem.hpp"

namespace puerhlab::ui {

using namespace album_util;
using namespace packed_proj;

// ── Constructor / Destructor ────────────────────────────────────────────────

AlbumBackend::AlbumBackend(QObject* parent)
    : QObject(parent),
      project_handler_(*this),
      thumb_(*this),
      folder_ctrl_(*this),
      filter_(*this),
      import_export_(*this),
      editor_(*this),
      rule_model_(this) {
  editor_.InitializeEditorLuts();
  SetServiceState(
      false,
      "Select a project: load a .puerhproj package or metadata JSON, or create a new packed project.");
  task_status_ = "Open or create a project to begin.";
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

auto AlbumBackend::FieldOptions() const -> QVariantList {
  return rule_model_.FieldOptions();
}

auto AlbumBackend::FilterInfo() const -> QString {
  return filter_.FormatFilterInfo(ShownCount(), TotalCount());
}

int AlbumBackend::TotalCount() const {
  int count = 0;
  for (const auto& image : all_images_) {
    if (filter_.IsImageInCurrentFolder(image)) {
      ++count;
    }
  }
  return count;
}

// ── Q_INVOKABLE: Filter-rule forwarding ─────────────────────────────────────

void AlbumBackend::AddRule() { rule_model_.AddRule(); }
void AlbumBackend::RemoveRule(int index) { rule_model_.RemoveRule(index); }
void AlbumBackend::SetRuleField(int index, int fieldValue) { rule_model_.SetField(index, fieldValue); }
void AlbumBackend::SetRuleOp(int index, int opValue) { rule_model_.SetOp(index, opValue); }
void AlbumBackend::SetRuleValue(int index, const QString& value) { rule_model_.SetValue(index, value); }
void AlbumBackend::SetRuleValue2(int index, const QString& value) { rule_model_.SetValue2(index, value); }

auto AlbumBackend::CompareOptionsForField(int fieldValue) const -> QVariantList {
  return FilterRuleModel::CompareOptionsForField(static_cast<FilterField>(fieldValue));
}

auto AlbumBackend::PlaceholderForField(int fieldValue) const -> QString {
  return FilterRuleModel::PlaceholderForField(static_cast<FilterField>(fieldValue));
}

// ── Q_INVOKABLE: Filter / folder delegation ─────────────────────────────────

void AlbumBackend::ApplyFilters(int joinOpValue) { filter_.ApplyFilters(joinOpValue); }
void AlbumBackend::ClearFilters() { filter_.ClearFilters(); }

void AlbumBackend::SelectFolder(uint folderId) {
  if (project_handler_.project_loading() || !project_handler_.project()) return;

  folder_ctrl_.ApplyFolderSelection(static_cast<sl_element_id_t>(folderId), true);
  filter_.ReapplyCurrentFilters();
}

void AlbumBackend::CreateFolder(const QString& folderName) { folder_ctrl_.CreateFolder(folderName); }
void AlbumBackend::DeleteFolder(uint folderId) { folder_ctrl_.DeleteFolder(folderId); }

// ── Q_INVOKABLE: Import / export delegation ─────────────────────────────────

void AlbumBackend::StartImport(const QStringList& fileUrlsOrPaths) { import_export_.StartImport(fileUrlsOrPaths); }
void AlbumBackend::CancelImport() { import_export_.CancelImport(); }

void AlbumBackend::StartExport(const QString& outputDirUrlOrPath) {
  import_export_.StartExport(outputDirUrlOrPath);
}

void AlbumBackend::StartExportWithOptions(const QString& outputDirUrlOrPath,
                                          const QString& formatName, bool resizeEnabled,
                                          int maxLengthSide, int quality, int bitDepth,
                                          int pngCompressionLevel,
                                          const QString& tiffCompression) {
  import_export_.StartExportWithOptions(outputDirUrlOrPath, formatName, resizeEnabled,
                                        maxLengthSide, quality, bitDepth, pngCompressionLevel,
                                        tiffCompression);
}

void AlbumBackend::StartExportWithOptionsForTargets(const QString& outputDirUrlOrPath,
                                                    const QString& formatName,
                                                    bool resizeEnabled, int maxLengthSide,
                                                    int quality, int bitDepth,
                                                    int pngCompressionLevel,
                                                    const QString& tiffCompression,
                                                    const QVariantList& targetEntries) {
  import_export_.StartExportWithOptionsForTargets(outputDirUrlOrPath, formatName, resizeEnabled,
                                                  maxLengthSide, quality, bitDepth,
                                                  pngCompressionLevel, tiffCompression,
                                                  targetEntries);
}

void AlbumBackend::ResetExportState() { import_export_.ResetExportState(); }

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

bool AlbumBackend::LoadProject(const QString& metaFileUrlOrPath) {
  if (project_handler_.project_loading()) {
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

    return project_handler_.InitializeServices(unpacked_db_path, unpacked_meta_path,
                                               ProjectOpenMode::kLoadExisting,
                                               project_path, workspace_dir);
  }

  if (!IsMetadataJsonPath(project_path)) {
    SetServiceMessageForCurrentProject(
        "Unsupported project format. Choose a .json or .puerhproj file.");
    return false;
  }

  const auto db_hint_path =
      project_path.parent_path() / (project_path.stem().wstring() + L".db");
  return project_handler_.InitializeServices(db_hint_path, project_path,
                                             ProjectOpenMode::kLoadExisting,
                                             BuildBundlePathFromMetaPath(project_path), {});
}

bool AlbumBackend::CreateProjectInFolder(const QString& folderUrlOrPath) {
  return CreateProjectInFolderNamed(folderUrlOrPath, "album_editor_project");
}

bool AlbumBackend::CreateProjectInFolderNamed(const QString& folderUrlOrPath,
                                              const QString& projectName) {
  if (project_handler_.project_loading()) {
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
      project_handler_.InitializeServices(runtime_pair.first, runtime_pair.second,
                                          ProjectOpenMode::kCreateNew,
                                          packed_path_opt.value(), workspace_dir);
  if (!started) {
    CleanupWorkspaceDirectory(workspace_dir);
  }
  return started;
}

bool AlbumBackend::SaveProject() {
  if (project_handler_.project_loading()) {
    SetServiceMessageForCurrentProject("Please wait until project loading finishes.");
    return false;
  }

  if (!project_handler_.project() || project_handler_.meta_path().empty()) {
    SetServiceState(false, "No project is loaded yet.");
    SetTaskState("No project to save.", 0, false);
    return false;
  }

  if (editor_.editor_active()) {
    editor_.FinalizeEditorSession(true);
  }

  if (!project_handler_.PersistCurrentProjectState()) {
    SetServiceMessageForCurrentProject("Project save failed.");
    SetTaskState("Project save failed.", 0, false);
    return false;
  }

  QString package_error;
  if (!project_handler_.PackageCurrentProjectFiles(&package_error)) {
    SetServiceMessageForCurrentProject(
        package_error.isEmpty() ? "Project saved, but packing failed." : package_error);
    SetTaskState("Project packing failed.", 0, false);
    return false;
  }

  SetServiceMessageForCurrentProject(
      project_handler_.package_path().empty()
          ? QString("Project saved to %1").arg(PathToQString(project_handler_.meta_path()))
          : QString("Project saved and packed to %1").arg(PathToQString(project_handler_.package_path())));
  SetTaskState(project_handler_.package_path().empty()
                   ? "Project saved."
                   : "Project saved and packed.",
               100, false);
  ScheduleIdleTaskStateReset(1200);
  return true;
}

// ── Shared internal methods ─────────────────────────────────────────────────

void AlbumBackend::SetServiceState(bool ready, const QString& message) {
  if (service_ready_ == ready && service_message_ == message) return;
  service_ready_   = ready;
  service_message_ = message;
  emit ServiceStateChanged();
}

void AlbumBackend::SetServiceMessageForCurrentProject(const QString& message) {
  SetServiceState(project_handler_.project() != nullptr, message);
}

void AlbumBackend::ScheduleIdleTaskStateReset(int delayMs) {
  QTimer::singleShot(std::max(delayMs, 0), this, [this]() {
    if (!import_export_.export_inflight() && !task_cancel_visible_) {
      SetTaskState("No background tasks", 0, false);
    }
  });
}

void AlbumBackend::SetTaskState(const QString& status, int progress, bool cancelVisible) {
  task_status_         = status;
  task_progress_       = std::clamp(progress, 0, 100);
  task_cancel_visible_ = cancelVisible;
  emit TaskStateChanged();
}

void AlbumBackend::AddOrUpdateAlbumItem(sl_element_id_t elementId, image_id_t imageId,
                                        const file_name_t& fallbackName,
                                        sl_element_id_t parentFolderId) {
  AlbumItem* item = nullptr;

  if (const auto it = index_by_element_id_.find(elementId); it != index_by_element_id_.end()) {
    item = &all_images_[it->second];
  } else {
    AlbumItem next;
    next.element_id = elementId;
    next.image_id   = imageId;
    next.file_name  = WStringToQString(fallbackName);
    next.extension  = ExtensionFromFileName(next.file_name);
    next.accent     = AccentForIndex(all_images_.size());

    all_images_.push_back(std::move(next));
    index_by_element_id_[elementId] = all_images_.size() - 1;
    item = &all_images_.back();
  }

  if (!item) return;

  item->element_id       = elementId;
  item->image_id         = imageId;
  item->parent_folder_id = parentFolderId;

  auto proj = project_handler_.project();
  if (proj) {
    try {
      const auto infoOpt =
          proj->GetSleeveService()->Read<std::optional<std::pair<QString, QDate>>>(
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
                  element = fs.Get(elementId);
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

}  // namespace puerhlab::ui
