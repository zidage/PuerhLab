//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QMetaObject>
#include <QObject>
#include <QPointer>
#include <QStringList>
#include <QThreadPool>
#include <QVariantList>
#include <QVariantMap>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "app/album_browse_service.hpp"
#include "app/editor_adjustment_context.hpp"
#include "app/image_pool_service.hpp"
#include "app/thumbnail_service.hpp"
#include "image/image.hpp"
#include "image/image_buffer.hpp"
#include "sleeve/album_query.hpp"
#include "storage/mapper/image/image_search_columns.hpp"
#include "type/supported_file_type.hpp"
#include "ui/alcedo_main/album_backend/path_utils.hpp"
#include "ui/alcedo_main/automation/automation_command_support.hpp"
#include "ui/alcedo_main/automation/automation_library_commands.hpp"

namespace alcedo::automation {
namespace {

constexpr int kMaxThumbnailLongEdge = 2048;
/// The most photos of one library.info or library.thumbnails request.
constexpr int kMaxBatchElementIds   = 1000;
/// The thumbnail renders that library.thumbnails keeps requested at the same time.
constexpr int kThumbnailsInFlight   = 4;

struct SortFieldName {
  const char*    name;
  AlbumSortField field;
};

constexpr std::array<SortFieldName, 8> kSortFields = {{
    {"none", AlbumSortField::kNone},
    {"capture_time", AlbumSortField::kCaptureTime},
    {"import_time", AlbumSortField::kImportTime},
    {"camera_model", AlbumSortField::kCameraModel},
    {"lens", AlbumSortField::kLens},
    {"rating", AlbumSortField::kRating},
    {"labels", AlbumSortField::kLabels},
    {"edit_time", AlbumSortField::kEditTime},
}};

auto                                   PathText(const std::filesystem::path& path) -> QString {
  return QString::fromStdWString(path.generic_wstring());
}

auto ToPath(const QString& text) -> std::filesystem::path {
  return std::filesystem::path(text.toStdWString());
}

auto RootFolderId(ProjectService& project) -> std::optional<sl_element_id_t> {
  const auto sleeve = project.GetSleeveService();
  if (!sleeve) {
    return std::nullopt;
  }
  const auto root = sleeve->ResolveFolder(ui::album_util::RootFsPath());
  if (!root) {
    return std::nullopt;
  }
  return root->element_id_;
}

/// Depth-first folder list below @p folder_path, parents before children.
void AppendFolders(const AlbumBrowseService& browse, const std::filesystem::path& folder_path,
                   sl_element_id_t parent_id, int depth, QJsonArray* folders) {
  for (const auto& folder : browse.ListFolders(folder_path)) {
    folders->push_back(QJsonObject{{"folder_id", static_cast<double>(folder.folder_id_)},
                                   {"parent_id", static_cast<double>(parent_id)},
                                   {"name", QString::fromStdWString(folder.folder_name_)},
                                   {"path", PathText(folder.folder_path_)},
                                   {"depth", depth}});
    AppendFolders(browse, folder.folder_path_, folder.folder_id_, depth + 1, folders);
  }
}

auto ThumbnailTierFor(int long_edge) -> ThumbnailResolution {
  if (long_edge <= 256) return ThumbnailResolution::k256;
  if (long_edge <= 512) return ThumbnailResolution::k512;
  if (long_edge <= 1024) return ThumbnailResolution::k1024;
  return ThumbnailResolution::k2048;
}

/// A thumbnail PNG that WriteThumbnailPng wrote, or the reason why it wrote none.
struct ThumbnailFile {
  QString path_;
  int     width_  = 0;
  int     height_ = 0;
  QString error_;
};

/// Renders the library thumbnail of @p row with @p thumbnails, scales it to @p long_edge, and
/// writes it to the PNG file @p out. Calls @p done on the GUI thread with the file or the error.
void WriteThumbnailPng(const std::shared_ptr<ThumbnailService>& thumbnails,
                       const SearchResultRow& row, const QString& out, int long_edge,
                       std::function<void(ThumbnailFile)> done) {
  std::weak_ptr<ThumbnailService> weak_service = thumbnails;
  // The render answers on a worker thread; the conversion and the answer run on this one.
  CallbackDispatcher              dispatcher   = [](std::function<void()> callback) {
    QMetaObject::invokeMethod(QCoreApplication::instance(), std::move(callback),
                                             Qt::QueuedConnection);
  };
  thumbnails->GetThumbnailDetailed(
      row.file_id_, row.image_id_,
      [weak_service, out, long_edge, done = std::move(done)](ThumbnailRequestResult result) {
        QImage  image;
        QString failure = QString::fromStdString(result.message);
        if (result.status == ThumbnailRequestStatus::kReady && result.guard &&
            result.guard->thumbnail_buffer_) {
          auto* buffer = result.guard->thumbnail_buffer_.get();
          if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
            buffer->SyncToCPU();
          }
          if (buffer->cpu_data_valid_) {
            image = ui::album_util::MatRgba32fToQImageCopy(buffer->GetCPUData());
          }
        }
        if (result.guard) {
          if (auto service = weak_service.lock()) {
            service->ReleaseThumbnail(result.key);
          }
        }
        ThumbnailFile file;
        if (image.isNull()) {
          file.error_ = failure.isEmpty()
                            ? QStringLiteral("The thumbnail render returned no image.")
                            : failure;
          done(std::move(file));
          return;
        }
        if (std::max(image.width(), image.height()) > long_edge) {
          image = image.scaled(long_edge, long_edge, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        }
        if (!image.save(out, "PNG")) {
          file.error_ = QStringLiteral("The PNG file cannot be written: %1").arg(out);
          done(std::move(file));
          return;
        }
        file.path_   = out;
        file.width_  = image.width();
        file.height_ = image.height();
        done(std::move(file));
      },
      /*pin_if_found=*/true, dispatcher, ThumbnailTierFor(long_edge));
}

/// One library.thumbnails request: the photos, the next photo to request, and the item of each
/// photo. GUI thread only. The thumbnail callbacks share it; it ends with the last callback.
struct ThumbnailBatch {
  std::vector<SearchResultRow>    rows_;
  std::vector<QJsonObject>        items_;
  QString                         out_dir_;
  int                             long_edge_ = 0;
  std::size_t                     next_      = 0;
  std::size_t                     finished_  = 0;
  int                             in_flight_ = 0;
  std::weak_ptr<ThumbnailService> thumbnails_;
  QPointer<AutomationCommandWait> wait_;
};

/// Requests the next thumbnails of @p batch until kThumbnailsInFlight renders run. Answers the
/// request when the last thumbnail is written or has failed. Stops requesting when the request
/// was answered by its time limit.
void RequestNextThumbnails(const std::shared_ptr<ThumbnailBatch>& batch) {
  while (batch->in_flight_ < kThumbnailsInFlight && batch->next_ < batch->rows_.size()) {
    if (batch->wait_ == nullptr || batch->wait_->IsAnswered()) {
      return;
    }
    const auto thumbnails = batch->thumbnails_.lock();
    if (!thumbnails) {
      batch->wait_->SendOwnerError(AutomationErrorCode::Failed,
                                   QStringLiteral("The thumbnail service closed."));
      return;
    }
    const std::size_t      index = batch->next_++;
    const SearchResultRow& row   = batch->rows_[index];
    const QString out = QDir(batch->out_dir_).filePath(QStringLiteral("%1.png").arg(row.file_id_));
    ++batch->in_flight_;
    WriteThumbnailPng(thumbnails, row, out, batch->long_edge_, [batch, index](ThumbnailFile file) {
      --batch->in_flight_;
      ++batch->finished_;
      QJsonObject& item = batch->items_[index];
      if (file.error_.isEmpty()) {
        item.insert("path", file.path_);
        item.insert("width", file.width_);
        item.insert("height", file.height_);
      } else {
        item.insert("error", file.error_);
      }
      if (batch->wait_ == nullptr || batch->wait_->IsAnswered()) {
        return;
      }
      if (batch->finished_ < batch->rows_.size()) {
        RequestNextThumbnails(batch);
        return;
      }
      QJsonArray items;
      int        failed = 0;
      for (const QJsonObject& written : batch->items_) {
        failed += written.contains("error") ? 1 : 0;
        items.push_back(written);
      }
      batch->wait_->SendResult(
          QJsonObject{{"items", items},
                      {"written_count", static_cast<int>(batch->items_.size()) - failed},
                      {"failed_count", failed}});
    });
  }
}

/// The metadata keeps focal length and aperture as float, so 6.3f reads as 6.3000001907 when
/// widened. Two decimals are finer than any lens or aperture marking; the Image search columns
/// round the same way.
auto RoundedLensValue(float value) -> double {
  return std::round(static_cast<double>(value) * 100.0) / 100.0;
}

/// The library.info item of @p row: the library columns, the source file, and the capture
/// values of the Image metadata that the import read. A value that the metadata does not have
/// is left out.
auto PhotoInfoItem(ImagePoolService* image_pool, const SearchResultRow& row) -> QJsonObject {
  QJsonObject item{{"element_id", static_cast<double>(row.file_id_)},
                   {"image_id", static_cast<double>(row.image_id_)},
                   {"file_name", QString::fromStdString(row.file_name_)},
                   {"rating", row.rating_}};
  if (image_pool == nullptr) {
    return item;
  }
  const auto insert_text = [&item](const char* name, const std::string& text) {
    if (!text.empty()) {
      item.insert(QLatin1String(name), QString::fromStdString(text));
    }
  };
  try {
    image_pool->Read<void>(row.image_id_, [&](std::shared_ptr<Image> image) {
      if (!image) {
        return;
      }
      if (!image->image_path_.empty()) {
        item.insert("file_path", PathText(image->image_path_));
      }
      if (!image->has_exif_display_.load()) {
        return;
      }
      const ExifDisplayMetaData& metadata = image->exif_display_;
      if (const auto capture = ParseCaptureDateTime(metadata.date_time_str_)) {
        item.insert("capture_time", QString::fromStdString(capture->date_time_));
      }
      insert_text("camera_make", metadata.make_);
      insert_text("camera_model", metadata.model_);
      insert_text("lens", metadata.lens_);
      const EditorImageExifDisplay exposure = ReadEditorImageExifDisplay(metadata);
      if (exposure.focal_mm.has_value() && std::isfinite(*exposure.focal_mm)) {
        item.insert("focal_mm", RoundedLensValue(*exposure.focal_mm));
      }
      if (exposure.aperture.has_value() && std::isfinite(*exposure.aperture)) {
        item.insert("aperture", RoundedLensValue(*exposure.aperture));
      }
      if (exposure.shutter_speed.has_value()) {
        const auto [numerator, denominator] = *exposure.shutter_speed;
        item.insert("shutter",
                    QString::fromStdString(FormatEditorImageExifDisplay(exposure).shutter));
        item.insert("shutter_seconds",
                    static_cast<double>(numerator) / static_cast<double>(denominator));
      }
      if (exposure.iso.has_value()) {
        item.insert("iso", static_cast<double>(*exposure.iso));
      }
      if (metadata.width_ > 0 && metadata.height_ > 0) {
        item.insert("width", static_cast<double>(metadata.width_));
        item.insert("height", static_cast<double>(metadata.height_));
      }
    });
  } catch (const std::exception& e) {
    // The library columns above still name the photo; the item says why the rest is missing.
    item.insert("metadata_error", QString::fromUtf8(e.what()));
  }
  return item;
}

/// Starts the import of @p paths and answers with its task id, or with the owner message when
/// ImportExportHandler does not start it.
void StartImportAndAnswer(ui::ApplicationModuleHost* host, AutomationTaskTracker* tracker,
                          const std::vector<image_path_t>& paths, ImportCategoryMask allowed,
                          AutomationReply& reply) {
  if (paths.empty()) {
    // An import of zero supported files finishes with counts, not with an error.
    reply.SendResult(QJsonObject{{"task_id", tracker->RecordEmptyImport()}});
    return;
  }
  host->import_export()->StartImportPaths(paths, false, allowed);
  if (!host->import_export()->ImportRunning()) {
    SendAutomationOwnerError(reply, AutomationErrorCode::Rejected, host->project()->TaskStatus());
    return;
  }
  reply.SendResult(QJsonObject{{"task_id", tracker->RunningImportId()}});
}

void ImportFolder(ui::ApplicationModuleHost* host, AutomationTaskTracker* tracker,
                  const QString& folder, bool recursive, AutomationReply reply) {
  if (!recursive) {
    std::vector<image_path_t> paths;
    const QFileInfoList       entries =
        QDir(folder).entryInfoList(QDir::Files | QDir::NoDotAndDotDot, QDir::Name);
    for (const QFileInfo& entry : entries) {
      const auto path = ToPath(entry.absoluteFilePath());
      if (CategoryForPath(path) != ImportFileCategory::Other) {
        paths.push_back(path);
      }
    }
    StartImportAndAnswer(host, tracker, paths, kAllImportCategories, reply);
    return;
  }
  // The recursive folder import of the GUI: the folder scan model, then StartFolderImport.
  ui::FolderImportScanModel* scan = host->import_export()->FolderScan();
  auto*                      wait = new AutomationCommandWait(std::move(reply), host);
  QObject::connect(
      scan, &ui::FolderImportScanModel::ScanStateChanged, wait, [host, tracker, scan, wait]() {
        if (scan->Scanning()) {
          return;
        }
        QObject::disconnect(scan, nullptr, wait, nullptr);
        if (!scan->ScanFinished() || !scan->FolderValid()) {
          wait->SendOwnerError(AutomationErrorCode::Failed,
                               QStringLiteral("The folder scan did not finish."));
          return;
        }
        if (scan->AllowedFileCount() == 0) {
          (void)scan->TakeFilePaths();
          wait->SendResult(QJsonObject{{"task_id", tracker->RecordEmptyImport()}});
          return;
        }
        host->import_export()->StartFolderImport();
        if (!host->import_export()->ImportRunning()) {
          wait->SendOwnerError(AutomationErrorCode::Rejected, host->project()->TaskStatus());
          return;
        }
        wait->SendResult(QJsonObject{{"task_id", tracker->RunningImportId()}});
      });
  scan->Start(folder);
}

auto ItemSchema() -> QJsonObject {
  return AutomationObjectSchema(QJsonObject{{"element_id", QJsonObject{{"type", "integer"}}},
                                            {"image_id", QJsonObject{{"type", "integer"}}},
                                            {"file_name", QJsonObject{{"type", "string"}}},
                                            {"rating", QJsonObject{{"type", "integer"}}}},
                                QJsonArray{"element_id", "image_id", "file_name", "rating"});
}

/// ImageController target entries (elementId, imageId, fileName) of @p rows.
auto TargetEntries(const std::vector<SearchResultRow>& rows) -> QVariantList {
  QVariantList entries;
  for (const auto& row : rows) {
    entries.push_back(
        QVariantMap{{QStringLiteral("elementId"), static_cast<uint>(row.file_id_)},
                    {QStringLiteral("imageId"), static_cast<uint>(row.image_id_)},
                    {QStringLiteral("fileName"), QString::fromStdString(row.file_name_)}});
  }
  return entries;
}

auto SelectionItems(const ui::LibrarySelection& selection) -> QJsonArray {
  QJsonArray items;
  for (const QVariant& value : selection.SelectedItems()) {
    const QVariantMap item = value.toMap();
    items.push_back(QJsonObject{
        {"element_id", static_cast<double>(item.value(QStringLiteral("elementId")).toUInt())},
        {"image_id", static_cast<double>(item.value(QStringLiteral("imageId")).toUInt())},
        {"file_name", item.value(QStringLiteral("fileName")).toString()}});
  }
  return items;
}

auto ElementIdsParam() -> QJsonObject {
  return QJsonObject{{"type", "array"},
                     {"items", QJsonObject{{"type", "integer"}, {"minimum", 1}}},
                     {"minItems", 1}};
}

/// `element_ids` of library.info and library.thumbnails: at most kMaxBatchElementIds ids.
auto BatchElementIdsParam() -> QJsonObject {
  QJsonObject param = ElementIdsParam();
  param.insert("maxItems", kMaxBatchElementIds);
  return param;
}

auto InfoItemSchema() -> QJsonObject {
  const QJsonObject text{{"type", "string"}};
  const QJsonObject integer{{"type", "integer"}};
  return AutomationObjectSchema(
      QJsonObject{
          {"element_id", integer},
          {"image_id", integer},
          {"file_name", text},
          {"rating", integer},
          {"file_path",
           QJsonObject{{"type", "string"}, {"description", "The source file of the photo."}}},
          {"capture_time",
           QJsonObject{{"type", "string"},
                       {"description", "YYYY-MM-DD HH:MM:SS by the camera clock."}}},
          {"camera_make", text},
          {"camera_model", text},
          {"lens", text},
          {"focal_mm",
           QJsonObject{{"type", "number"}, {"description", "The real focal length in mm."}}},
          {"aperture", QJsonObject{{"type", "number"}, {"description", "The f-number."}}},
          {"shutter", QJsonObject{{"type", "string"},
                                  {"description", "The exposure time, for example 1/250s."}}},
          {"shutter_seconds", QJsonObject{{"type", "number"}}},
          {"iso", integer},
          {"width",
           QJsonObject{{"type", "integer"}, {"description", "The image width in pixels."}}},
          {"height", integer},
          {"metadata_error",
           QJsonObject{{"type", "string"},
                       {"description",
                        "Why the metadata cannot be read; the item then has only the library "
                        "columns."}}}},
      QJsonArray{"element_id", "image_id", "file_name", "rating"});
}

void RegisterOrFail(AutomationCommandRegistry& registry, AutomationCommandSpec spec, bool* ok,
                    QString* error) {
  if (*ok && !registry.Register(std::move(spec), error)) {
    *ok = false;
  }
}

}  // namespace

auto RegisterAutomationLibraryCommands(AutomationCommandRegistry& registry,
                                       ui::ApplicationModuleHost* host,
                                       AutomationTaskTracker* tracker, const QString& host_mode,
                                       QString* error) -> bool {
  if (host == nullptr || tracker == nullptr) {
    if (error != nullptr) {
      *error = QStringLiteral("the library commands need a host and a task tracker");
    }
    return false;
  }
  bool                  ok = true;

  AutomationCommandSpec import_files;
  import_files.method      = QStringLiteral("library.import");
  import_files.description = QStringLiteral(
      "Imports the files 'paths', or the supported files of 'folder' (with its subfolders when "
      "'recursive', the default), into the current folder of the open project. Returns a task "
      "id for tasks.wait as soon as the import starts.");
  import_files.changes_state = true;
  import_files.params_schema = AutomationClosedParamsSchema(QJsonObject{
      {"paths", QJsonObject{{"type", "array"},
                            {"items", QJsonObject{{"type", "string"}}},
                            {"minItems", 1},
                            {"description", "Files to import."}}},
      {"folder", QJsonObject{{"type", "string"}, {"description", "A folder to import."}}},
      {"recursive", QJsonObject{{"type", "boolean"},
                                {"default", true},
                                {"description", "Include the subfolders of 'folder'."}}}});
  import_files.result_schema = AutomationObjectSchema(
      QJsonObject{{"task_id", QJsonObject{{"type", "string"}}}}, QJsonArray{"task_id"});
  import_files.handler = [host, tracker](const QJsonObject& params, AutomationReply reply) {
    const bool has_paths  = params.contains("paths");
    const bool has_folder = params.contains("folder");
    if (has_paths == has_folder) {
      SendAutomationParamError(reply, QString(),
                               QStringLiteral("give exactly one of paths and folder"));
      return;
    }
    std::vector<image_path_t> paths;
    if (has_paths) {
      const QJsonArray values = params.value("paths").toArray();
      for (qsizetype index = 0; index < values.size(); ++index) {
        const QString path = values.at(index).toString();
        if (!QFileInfo(path).isFile()) {
          SendAutomationParamError(reply, QStringLiteral("/paths/%1").arg(index),
                                   QStringLiteral("the file does not exist: %1").arg(path));
          return;
        }
        paths.push_back(ToPath(QFileInfo(path).absoluteFilePath()));
      }
    }
    const QString folder = params.value("folder").toString();
    if (has_folder && !QFileInfo(folder).isDir()) {
      SendAutomationParamError(reply, QStringLiteral("/folder"),
                               QStringLiteral("the folder does not exist: %1").arg(folder));
      return;
    }
    if (!AutomationEnteredProject(host, reply)) {
      return;
    }
    if (host->import_export()->ImportRunning()) {
      SendAutomationOwnerError(reply, AutomationErrorCode::Busy,
                               QStringLiteral("An import is already running."));
      return;
    }
    if (has_paths) {
      // The caller named these files, so every importable category is allowed.
      StartImportAndAnswer(host, tracker, paths, kAllImportCategories, reply);
      return;
    }
    ImportFolder(host, tracker, folder, params.value("recursive").toBool(true), std::move(reply));
  };
  RegisterOrFail(registry, std::move(import_files), &ok, error);

  AutomationCommandSpec folders;
  folders.method      = QStringLiteral("library.folders");
  folders.description = QStringLiteral(
      "Returns the folder tree of the open project as a list, parents before children. The "
      "root folder is first.");
  folders.params_schema = AutomationClosedParamsSchema();
  folders.result_schema = AutomationObjectSchema(
      QJsonObject{{"folders", QJsonObject{{"type", "array"}}}}, QJsonArray{"folders"});
  folders.handler = [host](const QJsonObject&, AutomationReply reply) {
    const auto project = AutomationEnteredProject(host, reply);
    if (!project) {
      return;
    }
    const auto browse  = project->GetAlbumBrowseService();
    const auto root_id = RootFolderId(*project);
    if (!browse || !root_id.has_value()) {
      SendAutomationOwnerError(reply, AutomationErrorCode::Failed,
                               QStringLiteral("The folder tree cannot be read."));
      return;
    }
    QJsonArray list{QJsonObject{{"folder_id", static_cast<double>(*root_id)},
                                {"parent_id", 0},
                                {"name", ""},
                                {"path", PathText(ui::album_util::RootFsPath())},
                                {"depth", 0}}};
    AppendFolders(*browse, ui::album_util::RootFsPath(), *root_id, 1, &list);
    reply.SendResult(QJsonObject{{"folders", list}});
  };
  RegisterOrFail(registry, std::move(folders), &ok, error);

  AutomationCommandSpec list;
  list.method      = QStringLiteral("library.list");
  list.description = QStringLiteral(
      "Returns one page of the photos of a folder (the root folder by default) in the order "
      "of 'sort', then by element id. 'total' is the photo count of the folder.");
  QJsonArray sort_names;
  for (const auto& sort : kSortFields) {
    sort_names.push_back(QString::fromLatin1(sort.name));
  }
  list.params_schema = AutomationClosedParamsSchema(QJsonObject{
      {"folder_id", QJsonObject{{"type", "integer"},
                                {"minimum", 0},
                                {"description", "A folder id from library.folders."}}},
      {"offset", QJsonObject{{"type", "integer"}, {"minimum", 0}, {"default", 0}}},
      {"limit", QJsonObject{{"type", "integer"},
                            {"minimum", 1},
                            {"maximum", static_cast<double>(kMaxAlbumQueryPageRows)},
                            {"default", 100}}},
      {"sort", QJsonObject{{"type", "string"}, {"enum", sort_names}, {"default", "none"}}},
      {"descending", QJsonObject{{"type", "boolean"}, {"default", false}}}});
  list.result_schema = AutomationObjectSchema(
      QJsonObject{{"total", QJsonObject{{"type", "integer"}}},
                  {"offset", QJsonObject{{"type", "integer"}}},
                  {"items", QJsonObject{{"type", "array"}, {"items", ItemSchema()}}}},
      QJsonArray{"total", "offset", "items"});
  list.handler = [host](const QJsonObject& params, AutomationReply reply) {
    const auto project = AutomationEnteredProject(host, reply);
    if (!project) {
      return;
    }
    const auto                     browse = project->GetAlbumBrowseService();
    std::optional<sl_element_id_t> folder_id;
    if (params.contains("folder_id")) {
      folder_id = static_cast<sl_element_id_t>(params.value("folder_id").toDouble());
    } else {
      folder_id = RootFolderId(*project);
    }
    if (!browse || !folder_id.has_value()) {
      SendAutomationOwnerError(reply, AutomationErrorCode::Failed,
                               QStringLiteral("The library cannot be read."));
      return;
    }
    AlbumQueryOptions options;
    const QString     sort = params.value("sort").toString(QStringLiteral("none"));
    for (const auto& entry : kSortFields) {
      if (sort == QLatin1String(entry.name)) {
        options.sort_field_ = entry.field;
      }
    }
    options.sort_direction_ = params.value("descending").toBool(false) ? SortDirection::kDescending
                                                                       : SortDirection::kAscending;
    const int64_t  offset   = static_cast<int64_t>(params.value("offset").toDouble(0));
    const int64_t  limit    = static_cast<int64_t>(params.value("limit").toDouble(100));
    AlbumQueryRead read;
    read.offset_ = offset;
    read.limit_  = limit;
    try {
      const AlbumQueryResult result = browse->ReadAlbumQuery(
          *folder_id, std::nullopt, options, host->stats()->ActiveSemanticModelKey(), read);
      QJsonArray items;
      for (const auto& row : result.rows_) {
        items.push_back(QJsonObject{{"element_id", static_cast<double>(row.photo_.file_id_)},
                                    {"image_id", static_cast<double>(row.photo_.image_id_)},
                                    {"file_name", QString::fromStdString(row.photo_.file_name_)},
                                    {"rating", row.photo_.rating_}});
      }
      reply.SendResult(QJsonObject{{"total", static_cast<double>(result.unique_file_count_)},
                                   {"offset", static_cast<double>(offset)},
                                   {"items", items}});
    } catch (const std::invalid_argument& e) {
      SendAutomationParamError(reply, QString(), QString::fromUtf8(e.what()));
    } catch (const std::exception& e) {
      SendAutomationOwnerError(reply, AutomationErrorCode::Failed, QString::fromUtf8(e.what()));
    }
  };
  RegisterOrFail(registry, std::move(list), &ok, error);

  AutomationCommandSpec thumbnail;
  thumbnail.method      = QStringLiteral("library.thumbnail");
  thumbnail.description = QStringLiteral(
      "Renders the library thumbnail of a photo and writes it as a PNG file whose long edge is "
      "'long_edge'.");
  thumbnail.params_schema = AutomationClosedParamsSchema(
      QJsonObject{{"element_id", QJsonObject{{"type", "integer"}, {"minimum", 1}}},
                  {"out", QJsonObject{{"type", "string"}, {"description", "The PNG file."}}},
                  {"long_edge", QJsonObject{{"type", "integer"},
                                            {"minimum", 16},
                                            {"maximum", kMaxThumbnailLongEdge},
                                            {"default", 512}}}},
      QJsonArray{"element_id", "out"});
  thumbnail.result_schema =
      AutomationObjectSchema(QJsonObject{{"path", QJsonObject{{"type", "string"}}},
                                         {"width", QJsonObject{{"type", "integer"}}},
                                         {"height", QJsonObject{{"type", "integer"}}}},
                             QJsonArray{"path", "width", "height"});
  thumbnail.handler = [host](const QJsonObject& params, AutomationReply reply) {
    const auto project = AutomationEnteredProject(host, reply);
    if (!project) {
      return;
    }
    const auto    element_id = static_cast<sl_element_id_t>(params.value("element_id").toDouble());
    const int     long_edge  = params.value("long_edge").toInt(512);
    const QString out        = QFileInfo(params.value("out").toString()).absoluteFilePath();
    const auto    browse     = project->GetAlbumBrowseService();
    auto          thumbnails = host->project()->handler().thumbnail_service();
    if (!browse || !thumbnails) {
      SendAutomationOwnerError(reply, AutomationErrorCode::Failed,
                               QStringLiteral("The thumbnail service is not available."));
      return;
    }
    std::vector<SearchResultRow> rows;
    try {
      const std::array<sl_element_id_t, 1> ids{element_id};
      rows = browse->ReadAlbumFileRows(ids);
    } catch (const std::exception& e) {
      SendAutomationOwnerError(reply, AutomationErrorCode::Failed, QString::fromUtf8(e.what()));
      return;
    }
    if (rows.empty() || rows.front().image_id_ == 0) {
      SendAutomationParamError(reply, QStringLiteral("/element_id"),
                               QStringLiteral("unknown element id: %1").arg(element_id));
      return;
    }
    if (!QDir().mkpath(QFileInfo(out).absolutePath())) {
      SendAutomationParamError(reply, QStringLiteral("/out"),
                               QStringLiteral("the folder of the file cannot be created"));
      return;
    }

    QPointer<AutomationCommandWait> wait = new AutomationCommandWait(std::move(reply), host);
    WriteThumbnailPng(thumbnails, rows.front(), out, long_edge, [wait](ThumbnailFile file) {
      if (wait == nullptr) {
        return;
      }
      if (!file.error_.isEmpty()) {
        wait->SendOwnerError(AutomationErrorCode::Failed, file.error_);
        return;
      }
      wait->SendResult(
          QJsonObject{{"path", file.path_}, {"width", file.width_}, {"height", file.height_}});
    });
  };
  RegisterOrFail(registry, std::move(thumbnail), &ok, error);

  AutomationCommandSpec thumbnails;
  thumbnails.method      = QStringLiteral("library.thumbnails");
  thumbnails.description = QStringLiteral(
      "Renders the library thumbnails of 'element_ids' and writes each as the PNG file "
      "<element_id>.png in 'out_dir', with the long edge 'long_edge'. Answers when every "
      "thumbnail is written or has failed: each item has 'path', 'width', and 'height', or "
      "'error'.");
  thumbnails.params_schema = AutomationClosedParamsSchema(
      QJsonObject{
          {"element_ids", BatchElementIdsParam()},
          {"out_dir", QJsonObject{{"type", "string"}, {"description", "Folder of the PNG files."}}},
          {"long_edge", QJsonObject{{"type", "integer"},
                                    {"minimum", 16},
                                    {"maximum", kMaxThumbnailLongEdge},
                                    {"default", 512}}}},
      QJsonArray{"element_ids", "out_dir"});
  thumbnails.result_schema = AutomationObjectSchema(
      QJsonObject{
          {"items",
           QJsonObject{{"type", "array"},
                       {"items", AutomationObjectSchema(
                                     QJsonObject{{"element_id", QJsonObject{{"type", "integer"}}},
                                                 {"path", QJsonObject{{"type", "string"}}},
                                                 {"width", QJsonObject{{"type", "integer"}}},
                                                 {"height", QJsonObject{{"type", "integer"}}},
                                                 {"error", QJsonObject{{"type", "string"}}}},
                                     QJsonArray{"element_id"})}}},
          {"written_count", QJsonObject{{"type", "integer"}}},
          {"failed_count", QJsonObject{{"type", "integer"}}}},
      QJsonArray{"items", "written_count", "failed_count"});
  thumbnails.handler = [host](const QJsonObject& params, AutomationReply reply) {
    const auto project = AutomationEnteredProject(host, reply);
    if (!project) {
      return;
    }
    const QString out_dir = QFileInfo(params.value("out_dir").toString()).absoluteFilePath();
    auto          service = host->project()->handler().thumbnail_service();
    if (!service) {
      SendAutomationOwnerError(reply, AutomationErrorCode::Failed,
                               QStringLiteral("The thumbnail service is not available."));
      return;
    }
    auto rows = ResolveAutomationElementRows(*project, params, reply);
    if (!rows.has_value()) {
      return;
    }
    if (!QDir().mkpath(out_dir)) {
      SendAutomationParamError(reply, QStringLiteral("/out_dir"),
                               QStringLiteral("the folder cannot be created"));
      return;
    }
    auto batch         = std::make_shared<ThumbnailBatch>();
    batch->rows_       = std::move(*rows);
    batch->out_dir_    = out_dir;
    batch->long_edge_  = params.value("long_edge").toInt(512);
    batch->thumbnails_ = service;
    for (const SearchResultRow& row : batch->rows_) {
      batch->items_.push_back(QJsonObject{{"element_id", static_cast<double>(row.file_id_)}});
    }
    batch->wait_ = new AutomationCommandWait(std::move(reply), host);
    RequestNextThumbnails(batch);
  };
  RegisterOrFail(registry, std::move(thumbnails), &ok, error);

  AutomationCommandSpec info;
  info.method      = QStringLiteral("library.info");
  info.description = QStringLiteral(
      "Returns the source file and the capture values of 'element_ids', in that order. The "
      "capture values come from the metadata that the import read: capture_time, camera_make, "
      "camera_model, lens, focal_mm, aperture, shutter, shutter_seconds, iso, width, and "
      "height. A value that the metadata does not have is left out of the item.");
  info.params_schema = AutomationClosedParamsSchema(
      QJsonObject{{"element_ids", BatchElementIdsParam()}}, QJsonArray{"element_ids"});
  info.result_schema = AutomationObjectSchema(
      QJsonObject{{"items", QJsonObject{{"type", "array"}, {"items", InfoItemSchema()}}}},
      QJsonArray{"items"});
  info.handler = [host](const QJsonObject& params, AutomationReply reply) {
    const auto project = AutomationEnteredProject(host, reply);
    if (!project) {
      return;
    }
    auto rows = ResolveAutomationElementRows(*project, params, reply);
    if (!rows.has_value()) {
      return;
    }
    // A cold image pool reads each Image from the database. The library view reads it on its
    // query worker for the same reason: up to 1000 reads on the GUI thread stall the window.
    QPointer<AutomationCommandWait> wait = new AutomationCommandWait(std::move(reply), host);
    QThreadPool::globalInstance()->start(
        [wait, image_pool = project->GetImagePoolService(), rows = std::move(*rows)]() {
          QJsonArray items;
          for (const SearchResultRow& row : rows) {
            items.push_back(PhotoInfoItem(image_pool.get(), row));
          }
          QMetaObject::invokeMethod(
              QCoreApplication::instance(),
              [wait, items = std::move(items)]() {
                if (wait != nullptr) {
                  wait->SendResult(QJsonObject{{"items", items}});
                }
              },
              Qt::QueuedConnection);
        });
  };
  RegisterOrFail(registry, std::move(info), &ok, error);

  const bool selection_available    = host_mode != QStringLiteral("headless");
  const auto selection_items_schema = AutomationObjectSchema(
      QJsonObject{{"items", QJsonObject{{"type", "array"}}}}, QJsonArray{"items"});

  AutomationCommandSpec selection_get;
  selection_get.method      = QStringLiteral("library.selection.get");
  selection_get.description = QStringLiteral(
      "Returns the library selection of the GUI. The headless host has no selection.");
  selection_get.params_schema = AutomationClosedParamsSchema();
  selection_get.result_schema = selection_items_schema;
  selection_get.handler = [host, selection_available](const QJsonObject&, AutomationReply reply) {
    if (!selection_available) {
      SendAutomationOwnerError(reply, AutomationErrorCode::NotReady,
                               QStringLiteral("The library selection exists only in the GUI."));
      return;
    }
    reply.SendResult(QJsonObject{{"items", SelectionItems(*host->library()->selection())}});
  };
  RegisterOrFail(registry, std::move(selection_get), &ok, error);

  AutomationCommandSpec selection_set;
  selection_set.method      = QStringLiteral("library.selection.set");
  selection_set.description = QStringLiteral(
      "Replaces the library selection of the GUI with 'element_ids'. The headless host has no "
      "selection.");
  selection_set.changes_state = true;
  selection_set.params_schema = AutomationClosedParamsSchema(
      QJsonObject{{"element_ids",
                   QJsonObject{{"type", "array"},
                               {"items", QJsonObject{{"type", "integer"}, {"minimum", 1}}}}}},
      QJsonArray{"element_ids"});
  selection_set.result_schema = selection_items_schema;
  selection_set.handler       = [host, selection_available](const QJsonObject& params,
                                                      AutomationReply    reply) {
    if (!selection_available) {
      SendAutomationOwnerError(reply, AutomationErrorCode::NotReady,
                                     QStringLiteral("The library selection exists only in the GUI."));
      return;
    }
    const auto project = AutomationEnteredProject(host, reply);
    if (!project) {
      return;
    }
    const auto rows = ResolveAutomationElementRows(*project, params, reply);
    if (!rows.has_value()) {
      return;
    }
    std::vector<ui::LibrarySelection::SelectedImage> images;
    for (const auto& row : *rows) {
      images.push_back(ui::LibrarySelection::SelectedImage{
          row.file_id_, row.image_id_, QString::fromStdString(row.file_name_), false});
    }
    host->library()->selection()->Replace(std::move(images));
    reply.SendResult(QJsonObject{{"items", SelectionItems(*host->library()->selection())}});
  };
  RegisterOrFail(registry, std::move(selection_set), &ok, error);

  AutomationCommandSpec rate;
  rate.method      = QStringLiteral("library.rate");
  rate.description = QStringLiteral(
      "Sets the star rating (0 to 5) of 'element_ids'. One photo is rated at once and the "
      "result holds 'applied_count'. Several photos use the batch save, and the result holds "
      "the 'task_id' of the save task.");
  rate.changes_state = true;
  rate.params_schema = AutomationClosedParamsSchema(
      QJsonObject{{"element_ids", ElementIdsParam()},
                  {"rating", QJsonObject{{"type", "integer"}, {"minimum", 0}, {"maximum", 5}}}},
      QJsonArray{"element_ids", "rating"});
  rate.result_schema = AutomationObjectSchema();
  rate.handler       = [host](const QJsonObject& params, AutomationReply reply) {
    const auto project = AutomationEnteredProject(host, reply);
    if (!project) {
      return;
    }
    const auto rows = ResolveAutomationElementRows(*project, params, reply);
    if (!rows.has_value()) {
      return;
    }
    const int         rating = params.value("rating").toInt();
    const QVariantMap result = host->library_mutations()->RateTargets(TargetEntries(*rows), rating);
    const QString     message = result.value(QStringLiteral("message")).toString();
    if (result.value(QStringLiteral("batch")).toBool()) {
      if (!result.value(QStringLiteral("started")).toBool()) {
        SendAutomationOwnerError(reply, AutomationErrorCode::Rejected, message);
        return;
      }
      reply.SendResult(QJsonObject{{"task_id", result.value(QStringLiteral("taskId")).toString()},
                                         {"rated_count", static_cast<int>(rows->size())},
                                         {"rating", rating}});
      return;
    }
    if (!result.value(QStringLiteral("success")).toBool()) {
      SendAutomationOwnerError(reply, AutomationErrorCode::Rejected, message);
      return;
    }
    reply.SendResult(QJsonObject{{"applied_count", 1},
                                       {"rating", result.value(QStringLiteral("rating")).toInt()}});
  };
  RegisterOrFail(registry, std::move(rate), &ok, error);

  AutomationCommandSpec remove;
  remove.method      = QStringLiteral("library.delete");
  remove.description = QStringLiteral(
      "Deletes 'element_ids'. With scope 'project' the photos leave the project; this needs the "
      "root folder as the current folder. With scope 'album' they leave the current album.");
  remove.changes_state = true;
  remove.params_schema = AutomationClosedParamsSchema(
      QJsonObject{
          {"element_ids", ElementIdsParam()},
          {"scope", QJsonObject{{"type", "string"}, {"enum", QJsonArray{"project", "album"}}}}},
      QJsonArray{"element_ids", "scope"});
  remove.result_schema =
      AutomationObjectSchema(QJsonObject{{"deleted_ids", QJsonObject{{"type", "array"}}},
                                         {"failed_ids", QJsonObject{{"type", "array"}}}},
                             QJsonArray{"deleted_ids", "failed_ids"});
  remove.handler = [host](const QJsonObject& params, AutomationReply reply) {
    const auto project = AutomationEnteredProject(host, reply);
    if (!project) {
      return;
    }
    const auto rows = ResolveAutomationElementRows(*project, params, reply);
    if (!rows.has_value()) {
      return;
    }
    ui::LibraryMutationOperations* mutations = host->library_mutations();
    const QString                  scope     = params.value("scope").toString();
    if (scope != mutations->DeleteScope()) {
      SendAutomationOwnerError(
          reply, AutomationErrorCode::Rejected,
          QStringLiteral("The current folder gives the delete scope '%1', not '%2'.")
              .arg(mutations->DeleteScope(), scope));
      return;
    }
    const QVariantMap result = mutations->DeleteTargets(TargetEntries(*rows));
    const QJsonArray  deleted =
        QJsonArray::fromVariantList(result.value(QStringLiteral("deletedElementIds")).toList());
    if (deleted.isEmpty()) {
      SendAutomationOwnerError(reply, AutomationErrorCode::Failed,
                               result.value(QStringLiteral("message")).toString());
      return;
    }
    reply.SendResult(QJsonObject{
        {"deleted_ids", deleted},
        {"failed_ids",
         QJsonArray::fromVariantList(result.value(QStringLiteral("failedElementIds")).toList())},
        {"message", result.value(QStringLiteral("message")).toString()}});
  };
  RegisterOrFail(registry, std::move(remove), &ok, error);

  return ok;
}

}  // namespace alcedo::automation
