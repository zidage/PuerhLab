#include "AlbumBackend.h"

#include <QApplication>
#include <QBuffer>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QMetaObject>
#include <QPointer>
#include <QStandardPaths>
#include <QTimer>
#include <QUrl>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cwctype>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include <opencv2/opencv.hpp>

#include "EditorDialog.h"
#include "app/render_service.hpp"
#include "image/image.hpp"
#include "io/image/image_loader.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "type/supported_file_type.hpp"

namespace puerhlab::demo {
namespace {

using namespace std::chrono_literals;

auto WStringToQString(const std::wstring& value) -> QString {
  return QString::fromStdWString(value);
}

auto PathToQString(const std::filesystem::path& path) -> QString {
#if defined(_WIN32)
  return QString::fromStdWString(path.wstring());
#else
  return QString::fromUtf8(path.string().c_str());
#endif
}

auto InputToPath(const QString& raw) -> std::optional<std::filesystem::path> {
  const QString trimmed = raw.trimmed();
  if (trimmed.isEmpty()) {
    return std::nullopt;
  }

  const QUrl maybe_url(trimmed);
  if (maybe_url.isValid() && maybe_url.scheme() == QStringLiteral("file")) {
    const QString local = maybe_url.toLocalFile();
    if (local.isEmpty()) {
      return std::nullopt;
    }
#if defined(_WIN32)
    return std::filesystem::path(local.toStdWString());
#else
    return std::filesystem::path(local.toStdString());
#endif
  }

#if defined(_WIN32)
  return std::filesystem::path(trimmed.toStdWString());
#else
  return std::filesystem::path(trimmed.toStdString());
#endif
}

auto DateFromTimeT(std::time_t value) -> QDate {
  if (value <= 0) {
    return {};
  }

  std::tm tm{};
#if defined(_WIN32)
  if (localtime_s(&tm, &value) != 0) {
    return {};
  }
#else
  if (localtime_r(&value, &tm) == nullptr) {
    return {};
  }
#endif

  const QDate date(tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
  return date.isValid() ? date : QDate();
}

auto DateFromExifString(const std::string& value) -> QDate {
  if (value.empty()) {
    return {};
  }

  const QString text = QString::fromUtf8(value.c_str()).trimmed();
  const QDateTime dt = QDateTime::fromString(text, "yyyy-MM-dd HH:mm:ss");
  if (dt.isValid()) {
    return dt.date();
  }
  const QDate date = QDate::fromString(text, "yyyy-MM-dd");
  return date.isValid() ? date : QDate();
}

auto ExtensionUpper(const std::filesystem::path& path) -> QString {
  const QString ext = PathToQString(path.extension());
  if (ext.isEmpty()) {
    return QString();
  }
  return ext.startsWith('.') ? ext.mid(1).toUpper() : ext.toUpper();
}

auto ExtensionFromFileName(const QString& name) -> QString {
  const int dot = name.lastIndexOf('.');
  if (dot < 0 || dot + 1 >= name.size()) {
    return QString();
  }
  return name.mid(dot + 1).toUpper();
}

auto DataUrlFromImage(const QImage& image) -> QString {
  if (image.isNull()) {
    return QString();
  }

  QByteArray bytes;
  QBuffer    buffer(&bytes);
  if (!buffer.open(QIODevice::WriteOnly)) {
    return QString();
  }
  if (!image.save(&buffer, "PNG")) {
    return QString();
  }
  return QStringLiteral("data:image/png;base64,") + QString::fromLatin1(bytes.toBase64());
}

auto MatRgba32fToQImageCopy(const cv::Mat& rgba32fOrU8) -> QImage {
  if (rgba32fOrU8.empty()) {
    return {};
  }

  cv::Mat rgba8;
  if (rgba32fOrU8.type() == CV_32FC4) {
    rgba32fOrU8.convertTo(rgba8, CV_8UC4, 255.0);
  } else if (rgba32fOrU8.type() == CV_8UC4) {
    rgba8 = rgba32fOrU8;
  } else {
    cv::Mat tmp;
    rgba32fOrU8.convertTo(tmp, CV_8UC4);
    rgba8 = tmp;
  }

  if (!rgba8.isContinuous()) {
    rgba8 = rgba8.clone();
  }

  QImage img(rgba8.data, rgba8.cols, rgba8.rows, static_cast<int>(rgba8.step),
             QImage::Format_RGBA8888);
  return img.copy();
}

auto ExtensionForExportFormat(ImageFormatType format) -> std::string {
  switch (format) {
    case ImageFormatType::JPEG:
      return ".jpg";
    case ImageFormatType::PNG:
      return ".png";
    case ImageFormatType::TIFF:
      return ".tiff";
    case ImageFormatType::WEBP:
      return ".webp";
    case ImageFormatType::EXR:
      return ".exr";
    default:
      return ".jpg";
  }
}

auto FormatFromName(const QString& value) -> ImageFormatType {
  const QString upper = value.trimmed().toUpper();
  if (upper == "PNG") {
    return ImageFormatType::PNG;
  }
  if (upper == "TIFF") {
    return ImageFormatType::TIFF;
  }
  if (upper == "WEBP") {
    return ImageFormatType::WEBP;
  }
  if (upper == "EXR") {
    return ImageFormatType::EXR;
  }
  return ImageFormatType::JPEG;
}

auto BitDepthFromInt(int value) -> ExportFormatOptions::BIT_DEPTH {
  if (value == 8) {
    return ExportFormatOptions::BIT_DEPTH::BIT_8;
  }
  if (value == 32) {
    return ExportFormatOptions::BIT_DEPTH::BIT_32;
  }
  return ExportFormatOptions::BIT_DEPTH::BIT_16;
}

auto TiffCompressFromName(const QString& value) -> ExportFormatOptions::TIFF_COMPRESS {
  const QString upper = value.trimmed().toUpper();
  if (upper == "LZW") {
    return ExportFormatOptions::TIFF_COMPRESS::LZW;
  }
  if (upper == "ZIP") {
    return ExportFormatOptions::TIFF_COMPRESS::ZIP;
  }
  return ExportFormatOptions::TIFF_COMPRESS::NONE;
}

auto ExportPathForOptions(const std::filesystem::path& srcPath, const std::filesystem::path& outDir,
                          sl_element_id_t elementId, image_id_t imageId,
                          ImageFormatType format) -> std::filesystem::path {
  std::wstring stem = srcPath.stem().wstring();
  if (stem.empty()) {
    stem = L"image";
  }
  const std::string suffix = "_" + std::to_string(static_cast<uint64_t>(elementId)) + "_" +
                             std::to_string(static_cast<uint64_t>(imageId));
  const std::string ext = ExtensionForExportFormat(format);
  return outDir / (stem + std::wstring(suffix.begin(), suffix.end()) +
                   std::wstring(ext.begin(), ext.end()));
}

auto ListCubeLutsInDir(const std::filesystem::path& dir) -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> files;
  std::error_code                    ec;
  if (!std::filesystem::exists(dir, ec) || ec) {
    return files;
  }

  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto ext = entry.path().extension().wstring();
    std::wstring normalized = ext;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::towlower);
    if (normalized == L".cube") {
      files.push_back(entry.path());
    }
  }

  std::sort(files.begin(), files.end(),
            [](const std::filesystem::path& a, const std::filesystem::path& b) {
              return a.filename().wstring() < b.filename().wstring();
            });
  return files;
}

auto NearlyEqual(float a, float b) -> bool {
  return std::abs(a - b) <= 1e-6f;
}

auto ClampToRange(double value, double minValue, double maxValue) -> float {
  return static_cast<float>(std::clamp(value, minValue, maxValue));
}

}  // namespace

AlbumBackend::AlbumBackend(QObject* parent) : QObject(parent), rule_model_(this) {
  const QString pictures = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
  default_export_folder_ = pictures.isEmpty() ? QDir::currentPath() : pictures;

  initializeServices();
  initializeEditorLuts();
  rebuildThumbnailView(std::nullopt);
  applyFilters(static_cast<int>(FilterOp::AND));
}

AlbumBackend::~AlbumBackend() {
  try {
    finalizeEditorSession(true);
    if (current_import_job_) {
      current_import_job_->canceled_.store(true);
    }
    if (pipeline_service_) {
      pipeline_service_->Sync();
    }
    if (project_) {
      project_->GetImagePoolService()->SyncWithStorage();
      project_->SaveProject(meta_path_);
    }
  } catch (...) {
  }
}

auto AlbumBackend::fieldOptions() const -> QVariantList {
  return rule_model_.fieldOptions();
}

auto AlbumBackend::filterInfo() const -> QString {
  return formatFilterInfo(shownCount(), totalCount());
}

void AlbumBackend::addRule() {
  rule_model_.addRule();
}

void AlbumBackend::removeRule(int index) {
  rule_model_.removeRule(index);
}

void AlbumBackend::setRuleField(int index, int fieldValue) {
  rule_model_.setField(index, fieldValue);
}

void AlbumBackend::setRuleOp(int index, int opValue) {
  rule_model_.setOp(index, opValue);
}

void AlbumBackend::setRuleValue(int index, const QString& value) {
  rule_model_.setValue(index, value);
}

void AlbumBackend::setRuleValue2(int index, const QString& value) {
  rule_model_.setValue2(index, value);
}

void AlbumBackend::applyFilters(int joinOpValue) {
  auto parsedJoin = static_cast<FilterOp>(joinOpValue);
  if (parsedJoin != FilterOp::AND && parsedJoin != FilterOp::OR) {
    parsedJoin = FilterOp::AND;
  }

  last_join_op_      = parsedJoin;

  const BuildResult result = buildFilterNode(parsedJoin);
  if (!result.error.isEmpty()) {
    if (validation_error_ != result.error) {
      validation_error_ = result.error;
      emit validationErrorChanged();
    }
    return;
  }

  if (!validation_error_.isEmpty()) {
    validation_error_.clear();
    emit validationErrorChanged();
  }

  QString nextSql;
  if (result.node.has_value()) {
    nextSql = QString::fromStdWString(FilterSQLCompiler::Compile(result.node.value()));
  }
  if (sql_preview_ != nextSql) {
    sql_preview_ = nextSql;
    emit sqlPreviewChanged();
  }

  if (!result.node.has_value()) {
    active_filter_ids_.reset();
    rebuildThumbnailView(std::nullopt);
    return;
  }

  if (!filter_service_) {
    active_filter_ids_ = std::unordered_set<sl_element_id_t>{};
    rebuildThumbnailView(active_filter_ids_);
    return;
  }

  try {
    const auto filterId = filter_service_->CreateFilterCombo(result.node.value());
    const auto idsOpt   = filter_service_->ApplyFilterOn(filterId, 0);
    filter_service_->RemoveFilterCombo(filterId);

    std::unordered_set<sl_element_id_t> nextIds;
    if (idsOpt.has_value()) {
      nextIds.reserve(idsOpt->size() * 2 + 1);
      for (const auto id : idsOpt.value()) {
        nextIds.insert(id);
      }
    }

    active_filter_ids_ = std::move(nextIds);
    rebuildThumbnailView(active_filter_ids_);
  } catch (const std::exception& e) {
    const QString error = QString("Filter execution failed: %1").arg(QString::fromUtf8(e.what()));
    if (validation_error_ != error) {
      validation_error_ = error;
      emit validationErrorChanged();
    }
    active_filter_ids_ = std::unordered_set<sl_element_id_t>{};
    rebuildThumbnailView(active_filter_ids_);
  }
}

void AlbumBackend::clearFilters() {
  rule_model_.clearAndReset();
  last_join_op_ = FilterOp::AND;

  if (!validation_error_.isEmpty()) {
    validation_error_.clear();
    emit validationErrorChanged();
  }
  if (!sql_preview_.isEmpty()) {
    sql_preview_.clear();
    emit sqlPreviewChanged();
  }

  active_filter_ids_.reset();
  rebuildThumbnailView(std::nullopt);
}

auto AlbumBackend::compareOptionsForField(int fieldValue) const -> QVariantList {
  return FilterRuleModel::compareOptionsForField(static_cast<FilterField>(fieldValue));
}

auto AlbumBackend::placeholderForField(int fieldValue) const -> QString {
  return FilterRuleModel::placeholderForField(static_cast<FilterField>(fieldValue));
}

void AlbumBackend::startImport(const QStringList& fileUrlsOrPaths) {
  if (!import_service_) {
    setTaskState("Import service is unavailable.", 0, false);
    return;
  }
  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    setTaskState("Import already running.", task_progress_, true);
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
    setTaskState("No supported files selected.", 0, false);
    return;
  }

  auto job            = std::make_shared<ImportJob>();
  current_import_job_ = job;

  setTaskState(QString("Importing %1 file(s)...").arg(static_cast<int>(paths.size())), 0, true);

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
          self->setTaskState(
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
          self->finishImport(result);
        },
        Qt::QueuedConnection);
  };

  try {
    ImportOptions options;
    current_import_job_ = import_service_->ImportToFolder(paths, image_path_t{}, options, job);
  } catch (const std::exception& e) {
    current_import_job_.reset();
    setTaskState(QString("Import failed: %1").arg(QString::fromUtf8(e.what())), 0, false);
  }
}

void AlbumBackend::cancelImport() {
  if (!current_import_job_) {
    return;
  }
  current_import_job_->canceled_.store(true);
  setTaskState("Cancelling import...", task_progress_, true);
}

void AlbumBackend::startExport(const QString& outputDirUrlOrPath) {
  startExportWithOptionsForTargets(outputDirUrlOrPath, "JPEG", false, 4096, 95, 16, 5, "NONE",
                                   {});
}

void AlbumBackend::startExportWithOptions(const QString& outputDirUrlOrPath,
                                          const QString& formatName, bool resizeEnabled,
                                          int maxLengthSide, int quality, int bitDepth,
                                          int pngCompressionLevel,
                                          const QString& tiffCompression) {
  startExportWithOptionsForTargets(outputDirUrlOrPath, formatName, resizeEnabled, maxLengthSide,
                                   quality, bitDepth, pngCompressionLevel, tiffCompression, {});
}

void AlbumBackend::startExportWithOptionsForTargets(const QString& outputDirUrlOrPath,
                                                    const QString& formatName,
                                                    bool resizeEnabled, int maxLengthSide,
                                                    int quality, int bitDepth,
                                                    int pngCompressionLevel,
                                                    const QString& tiffCompression,
                                                    const QVariantList& targetEntries) {
  const auto fail_with_status = [this](const QString& message) {
    export_status_ = message;
    emit exportStateChanged();
    setTaskState(message, 0, false);
  };

  if (!export_service_ || !project_) {
    fail_with_status("Export service is unavailable.");
    return;
  }
  if (export_inflight_) {
    fail_with_status("Export already running.");
    return;
  }

  export_status_        = "Preparing export queue...";
  export_error_summary_.clear();
  export_total_         = 0;
  export_completed_     = 0;
  export_succeeded_     = 0;
  export_failed_        = 0;
  export_skipped_       = 0;
  emit exportStateChanged();

  const auto outDirOpt = InputToPath(outputDirUrlOrPath);
  if (!outDirOpt.has_value()) {
    fail_with_status("No export folder selected.");
    return;
  }

  std::error_code ec;
  if (!std::filesystem::exists(outDirOpt.value(), ec)) {
    std::filesystem::create_directories(outDirOpt.value(), ec);
  }
  if (ec || !std::filesystem::is_directory(outDirOpt.value(), ec) || ec) {
    fail_with_status("Export folder is invalid.");
    return;
  }

  std::vector<std::pair<sl_element_id_t, image_id_t>> targets;
  if (!targetEntries.empty()) {
    targets.reserve(static_cast<size_t>(targetEntries.size()));
    std::unordered_set<uint64_t> dedupe{};
    dedupe.reserve(static_cast<size_t>(targetEntries.size()) * 2 + 1);
    for (const QVariant& entry : targetEntries) {
      const QVariantMap map = entry.toMap();
      const auto elementId =
          static_cast<sl_element_id_t>(map.value("elementId").toUInt());
      const auto imageId =
          static_cast<image_id_t>(map.value("imageId").toUInt());
      if (elementId == 0 || imageId == 0) {
        continue;
      }
      const uint64_t key = (static_cast<uint64_t>(elementId) << 32U) |
                           static_cast<uint64_t>(imageId);
      if (dedupe.insert(key).second) {
        targets.emplace_back(elementId, imageId);
      }
    }
  } else {
    targets.reserve(static_cast<size_t>(visible_thumbnails_.size()));
    for (const QVariant& entry : visible_thumbnails_) {
      const auto map = entry.toMap();
      const auto elementId = static_cast<sl_element_id_t>(map.value("elementId").toUInt());
      const auto imageId   = static_cast<image_id_t>(map.value("imageId").toUInt());
      if (elementId == 0 || imageId == 0) {
        continue;
      }
      targets.emplace_back(elementId, imageId);
    }
  }

  if (targets.empty()) {
    fail_with_status("No images to export.");
    return;
  }

  const ImageFormatType format         = FormatFromName(formatName);
  const bool            resizeEnabled_ = resizeEnabled;
  const int             maxLengthSide_ = std::clamp(maxLengthSide, 256, 16384);
  const int             quality_       = std::clamp(quality, 1, 100);
  const auto            bitDepth_      = BitDepthFromInt(bitDepth);
  const int             pngLevel_      = std::clamp(pngCompressionLevel, 0, 9);
  const auto            tiffCompress_  = TiffCompressFromName(tiffCompression);

  export_service_->ClearAllExportTasks();

  size_t  queuedCount   = 0;
  int     skippedCount  = 0;
  QString firstError{};
  for (const auto& [elementId, imageId] : targets) {
    try {
      const auto srcPath = project_->GetImagePoolService()->Read<std::filesystem::path>(
          imageId, [](std::shared_ptr<Image> image) {
            return image ? image->image_path_ : image_path_t{};
          });

      if (srcPath.empty()) {
        ++skippedCount;
        if (firstError.isEmpty()) {
          firstError = "Image source path is empty.";
        }
        continue;
      }

      ExportTask task;
      task.sleeve_id_                 = elementId;
      task.image_id_                  = imageId;
      task.options_.format_           = format;
      task.options_.resize_enabled_   = resizeEnabled_;
      task.options_.max_length_side_  = resizeEnabled_ ? maxLengthSide_ : 0;
      task.options_.quality_          = quality_;
      task.options_.bit_depth_        = bitDepth_;
      task.options_.compression_level_ = pngLevel_;
      task.options_.tiff_compress_     = tiffCompress_;
      task.options_.export_path_ =
          ExportPathForOptions(srcPath, outDirOpt.value(), elementId, imageId, format);

      export_service_->EnqueueExportTask(task);
      ++queuedCount;
    } catch (const std::exception& e) {
      ++skippedCount;
      if (firstError.isEmpty()) {
        firstError = QString::fromUtf8(e.what());
      }
    } catch (...) {
      ++skippedCount;
      if (firstError.isEmpty()) {
        firstError = "Unknown error while preparing export task.";
      }
    }
  }

  if (queuedCount == 0) {
    export_status_ = "No export tasks were queued.";
    if (!firstError.isEmpty()) {
      export_error_summary_ = firstError;
    }
    emit exportStateChanged();
    setTaskState("No valid export tasks could be created.", 0, false);
    return;
  }

  export_inflight_ = true;
  export_total_    = static_cast<int>(queuedCount);
  export_skipped_  = skippedCount;
  if (skippedCount > 0) {
    export_status_ = QString("Exporting %1 image(s). Skipped %2 invalid item(s).")
                         .arg(static_cast<int>(queuedCount))
                         .arg(skippedCount);
  } else {
    export_status_ = QString("Exporting %1 image(s)...").arg(static_cast<int>(queuedCount));
  }
  emit exportStateChanged();
  setTaskState(export_status_, 0, false);

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
              emit self->exportStateChanged();

              const int percent =
                  self->export_total_ > 0 ? (self->export_completed_ * 100) / self->export_total_ : 0;
              self->setTaskState(self->export_status_, percent, false);
            },
            Qt::QueuedConnection);
      },
      [self, skippedCount](std::shared_ptr<std::vector<ExportResult>> results) {
        if (!self) {
          return;
        }

        QMetaObject::invokeMethod(
            self,
            [self, results, skippedCount]() {
              if (!self) {
                return;
              }
              self->finishExport(results, skippedCount);
            },
            Qt::QueuedConnection);
      });
}

void AlbumBackend::resetExportState() {
  if (export_inflight_) {
    return;
  }
  export_status_        = "Ready to export.";
  export_error_summary_.clear();
  export_total_         = 0;
  export_completed_     = 0;
  export_succeeded_     = 0;
  export_failed_        = 0;
  export_skipped_       = 0;
  emit exportStateChanged();
}

void AlbumBackend::openEditor(uint elementId, uint imageId) {
  if (!pipeline_service_ || !project_ || !history_service_) {
    editor_status_ = "Editor service is unavailable.";
    emit editorStateChanged();
    return;
  }

  const auto nextElementId = static_cast<sl_element_id_t>(elementId);
  const auto nextImageId   = static_cast<image_id_t>(imageId);
  if (nextElementId == 0 || nextImageId == 0) {
    return;
  }

  finalizeEditorSession(true);

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
    emit editorStateChanged();

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
      requestThumbnail(nextElementId, nextImageId);
    }

    editor_status_ = "Editor closed. Changes saved.";
  } catch (const std::exception& e) {
    editor_status_ = QString("Failed to open editor: %1").arg(QString::fromUtf8(e.what()));
  }

  if (!editor_preview_url_.isEmpty()) {
    editor_preview_url_.clear();
    emit editorPreviewChanged();
  }
  editor_active_     = false;
  editor_busy_       = false;
  editor_element_id_ = 0;
  editor_image_id_   = 0;
  editor_title_.clear();
  emit editorStateChanged();
}

void AlbumBackend::closeEditor() {
  finalizeEditorSession(true);
}

void AlbumBackend::resetEditorAdjustments() {
  if (!editor_active_) {
    return;
  }
  editor_state_     = editor_initial_state_;
  editor_lut_index_ = lutIndexForPath(editor_state_.lut_path_);
  emit editorStateChanged();
  queueEditorRender(RenderType::FULL_RES_PREVIEW);
}

void AlbumBackend::requestEditorFullPreview() {
  if (!editor_active_) {
    return;
  }
  queueEditorRender(RenderType::FULL_RES_PREVIEW);
}

void AlbumBackend::setEditorLutIndex(int index) {
  if (!editor_active_ || index < 0 || index >= static_cast<int>(editor_lut_paths_.size())) {
    return;
  }
  if (editor_lut_index_ == index) {
    return;
  }
  editor_lut_index_       = index;
  editor_state_.lut_path_ = editor_lut_paths_[static_cast<size_t>(index)];
  emit editorStateChanged();
  queueEditorRender(RenderType::FAST_PREVIEW);
}

void AlbumBackend::setEditorExposure(double value) {
  setEditorAdjustment(editor_state_.exposure_, value, -10.0, 10.0);
}

void AlbumBackend::setEditorContrast(double value) {
  setEditorAdjustment(editor_state_.contrast_, value, -100.0, 100.0);
}

void AlbumBackend::setEditorSaturation(double value) {
  setEditorAdjustment(editor_state_.saturation_, value, -100.0, 100.0);
}

void AlbumBackend::setEditorTint(double value) {
  setEditorAdjustment(editor_state_.tint_, value, -100.0, 100.0);
}

void AlbumBackend::setEditorBlacks(double value) {
  setEditorAdjustment(editor_state_.blacks_, value, -100.0, 100.0);
}

void AlbumBackend::setEditorWhites(double value) {
  setEditorAdjustment(editor_state_.whites_, value, -100.0, 100.0);
}

void AlbumBackend::setEditorShadows(double value) {
  setEditorAdjustment(editor_state_.shadows_, value, -100.0, 100.0);
}

void AlbumBackend::setEditorHighlights(double value) {
  setEditorAdjustment(editor_state_.highlights_, value, -100.0, 100.0);
}

void AlbumBackend::setEditorSharpen(double value) {
  setEditorAdjustment(editor_state_.sharpen_, value, -100.0, 100.0);
}

void AlbumBackend::setEditorClarity(double value) {
  setEditorAdjustment(editor_state_.clarity_, value, -100.0, 100.0);
}

void AlbumBackend::initializeServices() {
  try {
    db_path_   = std::filesystem::temp_directory_path() / "album_editor_qml_demo.db";
    meta_path_ = std::filesystem::temp_directory_path() / "album_editor_qml_demo.json";

    std::error_code ec;
    std::filesystem::remove(db_path_, ec);
    ec.clear();
    std::filesystem::remove(meta_path_, ec);

    project_ = std::make_shared<ProjectService>(db_path_, meta_path_);
    pipeline_service_ = std::make_shared<PipelineMgmtService>(project_->GetStorageService());
    history_service_ = std::make_shared<EditHistoryMgmtService>(project_->GetStorageService());
    thumbnail_service_ = std::make_shared<ThumbnailService>(project_->GetSleeveService(),
                                                            project_->GetImagePoolService(),
                                                            pipeline_service_);
    filter_service_ = std::make_unique<SleeveFilterService>(project_->GetStorageService());
    import_service_ = std::make_unique<ImportServiceImpl>(project_->GetSleeveService(),
                                                          project_->GetImagePoolService());
    export_service_ = std::make_shared<ExportService>(project_->GetSleeveService(),
                                                      project_->GetImagePoolService(),
                                                      pipeline_service_);

    service_ready_   = true;
    service_message_ = QString("ProjectService initialized (%1)").arg(PathToQString(db_path_));
  } catch (const std::exception& e) {
    service_ready_   = false;
    service_message_ = QString("Service init failed: %1").arg(QString::fromUtf8(e.what()));
  }

  emit serviceStateChanged();
}

void AlbumBackend::rebuildThumbnailView(
    const std::optional<std::unordered_set<sl_element_id_t>>& allowedElementIds) {
  QVariantList next;
  next.reserve(static_cast<qsizetype>(all_images_.size()));

  int index = 0;
  for (const AlbumItem& image : all_images_) {
    if (allowedElementIds.has_value() && !allowedElementIds->contains(image.element_id)) {
      continue;
    }
    next.push_back(makeThumbMap(image, index++));
  }

  visible_thumbnails_ = std::move(next);
  emit thumbnailsChanged();
  emit countsChanged();
}

void AlbumBackend::addImportedEntries(const ImportLogSnapshot& snapshot) {
  std::unordered_set<image_id_t> metadataOk;
  metadataOk.reserve(snapshot.metadata_ok_.size() * 2 + 1);
  for (const auto id : snapshot.metadata_ok_) {
    metadataOk.insert(id);
  }

  for (const auto& created : snapshot.created_) {
    if (!metadataOk.empty() && !metadataOk.contains(created.image_id_)) {
      continue;
    }
    addOrUpdateAlbumItem(created.element_id_, created.image_id_, created.file_name_);
  }
}

void AlbumBackend::addOrUpdateAlbumItem(sl_element_id_t elementId, image_id_t imageId,
                                        const file_name_t& fallbackName) {
  AlbumItem* item = nullptr;

  if (const auto it = index_by_element_id_.find(elementId); it != index_by_element_id_.end()) {
    item = &all_images_[it->second];
  } else {
    AlbumItem next;
    next.element_id   = elementId;
    next.image_id     = imageId;
    next.file_name    = WStringToQString(fallbackName);
    next.extension    = ExtensionFromFileName(next.file_name);

    static const std::array<QString, 6> accents = {
        "#5AA2FF",
        "#4CC9A6",
        "#F7B267",
        "#E08BFF",
        "#7AD1FF",
        "#9BD65B",
    };
    next.accent = accents[all_images_.size() % accents.size()];

    all_images_.push_back(std::move(next));
    index_by_element_id_[elementId] = all_images_.size() - 1;
    item = &all_images_.back();
  }

  if (!item) {
    return;
  }

  item->element_id = elementId;
  item->image_id   = imageId;

  if (project_) {
    try {
      const auto infoOpt = project_->GetSleeveService()->Read<std::optional<std::pair<QString, QDate>>>(
          [elementId](FileSystem& fs) -> std::optional<std::pair<QString, QDate>> {
            const auto element = fs.Get(elementId);
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
      project_->GetImagePoolService()->Read<void>(
          imageId,
          [item](std::shared_ptr<Image> image) {
            if (!image) {
              return;
            }

            if (!image->image_name_.empty()) {
              item->file_name = WStringToQString(image->image_name_);
            }
            if (!image->image_path_.empty()) {
              item->extension = ExtensionUpper(image->image_path_);
            }

            const auto& exif = image->exif_display_;
            item->camera_model = QString::fromUtf8(exif.model_.c_str());
            item->iso          = static_cast<int>(exif.iso_);
            item->aperture     = static_cast<double>(exif.aperture_);
            item->focal_length = static_cast<double>(exif.focal_);
            item->rating       = exif.rating_;
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

  requestThumbnail(elementId, imageId);
}

void AlbumBackend::requestThumbnail(sl_element_id_t elementId, image_id_t imageId) {
  if (!thumbnail_service_) {
    return;
  }

  auto                 service = thumbnail_service_;
  QPointer<AlbumBackend> self(this);

  CallbackDispatcher dispatcher = [](std::function<void()> fn) {
    auto* app = QCoreApplication::instance();
    if (!app) {
      fn();
      return;
    }
    QMetaObject::invokeMethod(app, std::move(fn), Qt::QueuedConnection);
  };

  service->GetThumbnail(
      elementId, imageId,
      [self, service, elementId](std::shared_ptr<ThumbnailGuard> guard) {
        if (!self || !guard || !guard->thumbnail_buffer_) {
          return;
        }

        std::thread([self, service, elementId, guard = std::move(guard)]() mutable {
          QString dataUrl;
          try {
            auto* buffer = guard->thumbnail_buffer_.get();
            if (buffer) {
              if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
                buffer->SyncToCPU();
              }
              if (buffer->cpu_data_valid_) {
                QImage image = MatRgba32fToQImageCopy(buffer->GetCPUData());
                if (!image.isNull()) {
                  QImage scaled = image.scaled(220, 160, Qt::KeepAspectRatio,
                                               Qt::SmoothTransformation);
                  dataUrl = DataUrlFromImage(scaled);
                }
              }
            }
          } catch (...) {
          }

          if (self) {
            QMetaObject::invokeMethod(
                self,
                [self, elementId, dataUrl]() {
                  if (!self) {
                    return;
                  }
                  self->updateThumbnailDataUrl(elementId, dataUrl);
                },
                Qt::QueuedConnection);
          }

          try {
            if (service) {
              service->ReleaseThumbnail(elementId);
            }
          } catch (...) {
          }
        }).detach();
      },
      true, dispatcher);
}

void AlbumBackend::updateThumbnailDataUrl(sl_element_id_t elementId, const QString& dataUrl) {
  if (dataUrl.isEmpty()) {
    return;
  }

  const auto it = index_by_element_id_.find(elementId);
  if (it == index_by_element_id_.end()) {
    return;
  }

  auto& item = all_images_[it->second];
  if (item.thumb_data_url == dataUrl) {
    return;
  }

  item.thumb_data_url = dataUrl;

  for (qsizetype i = 0; i < visible_thumbnails_.size(); ++i) {
    QVariantMap row = visible_thumbnails_.at(i).toMap();
    if (static_cast<sl_element_id_t>(row.value("elementId").toUInt()) != elementId) {
      continue;
    }
    row.insert("thumbUrl", dataUrl);
    visible_thumbnails_[i] = row;
    break;
  }

  emit thumbnailUpdated(static_cast<uint>(elementId), dataUrl);
}

void AlbumBackend::finishImport(const ImportResult& result) {
  const auto importJob = current_import_job_;
  current_import_job_.reset();

  if (!importJob || !importJob->import_log_) {
    setTaskState("Import finished but no log snapshot is available.", 0, false);
    return;
  }

  const auto snapshot = importJob->import_log_->Snapshot();

  try {
    if (import_service_) {
      import_service_->SyncImports(snapshot, image_path_t{});
    }
    if (project_) {
      project_->GetSleeveService()->Sync();
      project_->GetImagePoolService()->SyncWithStorage();
      project_->SaveProject(meta_path_);
    }
  } catch (...) {
  }

  addImportedEntries(snapshot);
  reapplyCurrentFilters();

  setTaskState(QString("Import complete: %1 imported, %2 failed")
                   .arg(result.imported_)
                   .arg(result.failed_),
               100, false);

  QTimer::singleShot(1800, this, [this]() {
    if (!export_inflight_ && !task_cancel_visible_) {
      setTaskState("No background tasks", 0, false);
    }
  });
}

void AlbumBackend::finishExport(const std::shared_ptr<std::vector<ExportResult>>& results,
                                int skippedCount) {
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

  const int total  = ok + fail;
  export_total_    = std::max(export_total_, total);
  export_completed_ = total;
  export_succeeded_ = ok;
  export_failed_    = fail;
  export_skipped_   = skippedCount;
  export_error_summary_.clear();
  if (!errors.isEmpty()) {
    export_error_summary_ = errors.join('\n');
  }

  export_status_ = QString("Export complete. Written %1/%2 image(s), failed %3.")
                       .arg(ok)
                       .arg(total)
                       .arg(fail);
  if (skippedCount > 0) {
    export_status_ += QString(" Skipped %1 invalid item(s).").arg(skippedCount);
  }
  emit exportStateChanged();

  setTaskState(QString("Export complete: %1 ok, %2 failed").arg(ok).arg(fail), 100, false);

  QTimer::singleShot(1800, this, [this]() {
    if (!task_cancel_visible_) {
      setTaskState("No background tasks", 0, false);
    }
  });
}

void AlbumBackend::reapplyCurrentFilters() {
  applyFilters(static_cast<int>(last_join_op_));
}

auto AlbumBackend::buildFilterNode(FilterOp joinOp) const -> BuildResult {
  std::optional<FilterNode> rules_node;
  std::vector<FilterNode>   conditions;

  for (const auto& rule : rule_model_.rules()) {
    if (rule.value.trimmed().isEmpty()) {
      continue;
    }

    QString error;
    const auto value_opt = parseFilterValue(rule.field, rule.value, error);
    if (!value_opt.has_value()) {
      return BuildResult{.node = std::nullopt, .error = error};
    }

    FieldCondition condition{
        .field_        = rule.field,
        .op_           = rule.op,
        .value_        = value_opt.value(),
        .second_value_ = std::nullopt,
    };

    if (rule.op == CompareOp::BETWEEN) {
      if (rule.value2.trimmed().isEmpty()) {
        return BuildResult{.node = std::nullopt, .error = "BETWEEN requires two values."};
      }
      const auto second_opt = parseFilterValue(rule.field, rule.value2, error);
      if (!second_opt.has_value()) {
        return BuildResult{.node = std::nullopt, .error = error};
      }
      condition.second_value_ = second_opt.value();
    }

    conditions.push_back(FilterNode{
        FilterNode::Type::Condition, {}, {}, std::move(condition), std::nullopt});
  }

  if (!conditions.empty()) {
    if (conditions.size() == 1) {
      rules_node = conditions.front();
    } else {
      rules_node = FilterNode{
          FilterNode::Type::Logical, joinOp, std::move(conditions), {}, std::nullopt};
    }
  }

  if (rules_node.has_value()) {
    return BuildResult{.node = rules_node, .error = QString()};
  }
  return BuildResult{.node = std::nullopt, .error = QString()};
}

auto AlbumBackend::parseFilterValue(FilterField field, const QString& text, QString& error) const
    -> std::optional<FilterValue> {
  const QString trimmed = text.trimmed();
  const auto    kind    = FilterRuleModel::kindForField(field);

  if (kind == FilterValueKind::String) {
    return FilterValue{trimmed.toStdWString()};
  }

  if (kind == FilterValueKind::Int64) {
    bool       ok = false;
    const auto v  = trimmed.toLongLong(&ok);
    if (!ok) {
      error = "Expected an integer value.";
      return std::nullopt;
    }
    return FilterValue{static_cast<int64_t>(v)};
  }

  if (kind == FilterValueKind::Double) {
    bool       ok = false;
    const auto v  = trimmed.toDouble(&ok);
    if (!ok) {
      error = "Expected a numeric value.";
      return std::nullopt;
    }
    return FilterValue{v};
  }

  const auto date_opt = parseDate(trimmed);
  if (!date_opt.has_value()) {
    error = "Expected a date in YYYY-MM-DD format.";
    return std::nullopt;
  }
  return FilterValue{date_opt.value()};
}

auto AlbumBackend::parseDate(const QString& text) -> std::optional<std::tm> {
  const QStringList parts = text.trimmed().split('-', Qt::SkipEmptyParts);
  if (parts.size() != 3) {
    return std::nullopt;
  }

  bool      ok_year = false;
  bool      ok_mon  = false;
  bool      ok_day  = false;
  const int year    = parts[0].toInt(&ok_year);
  const int month   = parts[1].toInt(&ok_mon);
  const int day     = parts[2].toInt(&ok_day);
  if (!ok_year || !ok_mon || !ok_day) {
    return std::nullopt;
  }

  const QDate date(year, month, day);
  if (!date.isValid()) {
    return std::nullopt;
  }

  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon  = month - 1;
  tm.tm_mday = day;
  return tm;
}

auto AlbumBackend::formatFilterInfo(int shown, int total) const -> QString {
  if (total <= 0) {
    return "No images loaded.";
  }
  if (shown == total) {
    return QString("Showing %1 images").arg(total);
  }
  return QString("Showing %1 of %2").arg(shown).arg(total);
}

auto AlbumBackend::makeThumbMap(const AlbumItem& image, int index) const -> QVariantMap {
  static const std::array<QString, 6> accents = {
      "#5AA2FF",
      "#4CC9A6",
      "#F7B267",
      "#E08BFF",
      "#7AD1FF",
      "#9BD65B",
  };

  const QString aperture = image.aperture > 0.0 ? QString::number(image.aperture, 'f', 1) : "--";
  const QString focal    = image.focal_length > 0.0 ? QString::number(image.focal_length, 'f', 0) : "--";

  return QVariantMap{
      {"elementId", static_cast<uint>(image.element_id)},
      {"imageId", static_cast<uint>(image.image_id)},
      {"fileName", image.file_name.isEmpty() ? "(unnamed)" : image.file_name},
      {"cameraModel", image.camera_model.isEmpty() ? "Unknown" : image.camera_model},
      {"extension", image.extension.isEmpty() ? "--" : image.extension},
      {"iso", image.iso},
      {"aperture", aperture},
      {"focalLength", focal},
      {"captureDate", image.capture_date.isValid() ? image.capture_date.toString("yyyy-MM-dd") : "--"},
      {"rating", image.rating},
      {"tags", image.tags},
      {"accent", image.accent.isEmpty() ? accents[static_cast<size_t>(index) % accents.size()] : image.accent},
      {"thumbUrl", image.thumb_data_url},
  };
}

void AlbumBackend::initializeEditorLuts() {
  editor_lut_paths_.clear();
  editor_lut_options_.clear();

  editor_lut_paths_.push_back("");
  editor_lut_options_.push_back(QVariantMap{{"text", "None"}, {"value", 0}});

  // Prefer LUTs next to the executable (installed layout), fall back to source tree.
  const auto appLutsDir = std::filesystem::path(
      QCoreApplication::applicationDirPath().toStdWString()) / "LUTs";
  const auto srcLutsDir = std::filesystem::path(CONFIG_PATH) / "LUTs";
  const auto lutsDir    = std::filesystem::is_directory(appLutsDir) ? appLutsDir : srcLutsDir;
  const auto lutFiles   = ListCubeLutsInDir(lutsDir);
  for (const auto& path : lutFiles) {
    editor_lut_paths_.push_back(path.generic_string());
    editor_lut_options_.push_back(
        QVariantMap{{"text", QString::fromStdString(path.filename().string())},
                    {"value", static_cast<int>(editor_lut_paths_.size() - 1)}});
  }

  editor_lut_index_ = lutIndexForPath(editor_state_.lut_path_);
  if (!editor_lut_paths_.empty()) {
    editor_state_.lut_path_ = editor_lut_paths_[static_cast<size_t>(editor_lut_index_)];
  }
}

auto AlbumBackend::lutIndexForPath(const std::string& lutPath) const -> int {
  if (editor_lut_paths_.empty()) {
    return 0;
  }

  if (lutPath.empty()) {
    return 0;
  }

  for (size_t i = 0; i < editor_lut_paths_.size(); ++i) {
    if (editor_lut_paths_[i] == lutPath) {
      return static_cast<int>(i);
    }
  }

  const auto target = std::filesystem::path(lutPath).filename().wstring();
  for (size_t i = 0; i < editor_lut_paths_.size(); ++i) {
    if (std::filesystem::path(editor_lut_paths_[i]).filename().wstring() == target) {
      return static_cast<int>(i);
    }
  }
  return 0;
}

auto AlbumBackend::loadEditorStateFromPipeline() -> bool {
  auto exec = editor_pipeline_guard_ ? editor_pipeline_guard_->pipeline_ : nullptr;
  if (!exec) {
    return false;
  }

  auto ReadFloat = [](const PipelineStage& stage, OperatorType type,
                      const char* key) -> std::optional<float> {
    const auto op = stage.GetOperator(type);
    if (!op.has_value() || op.value() == nullptr) {
      return std::nullopt;
    }
    const auto j = op.value()->ExportOperatorParams();
    if (!j.contains("params")) {
      return std::nullopt;
    }
    const auto& params = j["params"];
    if (!params.contains(key)) {
      return std::nullopt;
    }
    try {
      return params[key].get<float>();
    } catch (...) {
      return std::nullopt;
    }
  };

  auto ReadNestedFloat = [](const PipelineStage& stage, OperatorType type, const char* key1,
                            const char* key2) -> std::optional<float> {
    const auto op = stage.GetOperator(type);
    if (!op.has_value() || op.value() == nullptr) {
      return std::nullopt;
    }
    const auto j = op.value()->ExportOperatorParams();
    if (!j.contains("params")) {
      return std::nullopt;
    }
    const auto& params = j["params"];
    if (!params.contains(key1)) {
      return std::nullopt;
    }
    const auto& inner = params[key1];
    if (!inner.contains(key2)) {
      return std::nullopt;
    }
    try {
      return inner[key2].get<float>();
    } catch (...) {
      return std::nullopt;
    }
  };

  auto ReadString = [](const PipelineStage& stage, OperatorType type,
                       const char* key) -> std::optional<std::string> {
    const auto op = stage.GetOperator(type);
    if (!op.has_value() || op.value() == nullptr) {
      return std::nullopt;
    }
    const auto j = op.value()->ExportOperatorParams();
    if (!j.contains("params")) {
      return std::nullopt;
    }
    const auto& params = j["params"];
    if (!params.contains(key)) {
      return std::nullopt;
    }
    try {
      return params[key].get<std::string>();
    } catch (...) {
      return std::nullopt;
    }
  };

  const auto& basic  = exec->GetStage(PipelineStageName::Basic_Adjustment);
  const auto& color  = exec->GetStage(PipelineStageName::Color_Adjustment);
  const auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);

  if (!basic.GetOperator(OperatorType::EXPOSURE).has_value()) {
    return false;
  }

  EditorState loaded;
  if (const auto v = ReadFloat(basic, OperatorType::EXPOSURE, "exposure"); v.has_value()) {
    loaded.exposure_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::CONTRAST, "contrast"); v.has_value()) {
    loaded.contrast_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::BLACK, "black"); v.has_value()) {
    loaded.blacks_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::WHITE, "white"); v.has_value()) {
    loaded.whites_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::SHADOWS, "shadows"); v.has_value()) {
    loaded.shadows_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::HIGHLIGHTS, "highlights"); v.has_value()) {
    loaded.highlights_ = v.value();
  }
  if (const auto v = ReadFloat(color, OperatorType::SATURATION, "saturation"); v.has_value()) {
    loaded.saturation_ = v.value();
  }
  if (const auto v = ReadFloat(color, OperatorType::TINT, "tint"); v.has_value()) {
    loaded.tint_ = v.value();
  }
  if (const auto v = ReadNestedFloat(detail, OperatorType::SHARPEN, "sharpen", "offset");
      v.has_value()) {
    loaded.sharpen_ = v.value();
  }
  if (const auto v = ReadFloat(detail, OperatorType::CLARITY, "clarity"); v.has_value()) {
    loaded.clarity_ = v.value();
  }
  if (const auto lut = ReadString(color, OperatorType::LMT, "ocio_lmt");
      lut.has_value() && !lut->empty()) {
    loaded.lut_path_ = *lut;
  } else {
    loaded.lut_path_.clear();
  }

  editor_state_     = loaded;
  editor_lut_index_ = lutIndexForPath(editor_state_.lut_path_);
  if (!editor_lut_paths_.empty()) {
    editor_state_.lut_path_ = editor_lut_paths_[static_cast<size_t>(editor_lut_index_)];
  }

  return true;
}

void AlbumBackend::setupEditorPipeline() {
  if (!editor_pipeline_guard_ || !editor_pipeline_guard_->pipeline_ || !project_) {
    throw std::runtime_error("Editor services are unavailable.");
  }

  auto imageDesc = project_->GetImagePoolService()->Read<std::shared_ptr<Image>>(
      editor_image_id_, [](const std::shared_ptr<Image>& img) { return img; });
  auto bytes = ByteBufferLoader::LoadFromImage(imageDesc);
  if (!bytes) {
    throw std::runtime_error("Failed to load image bytes.");
  }

  editor_base_task_                    = PipelineTask{};
  editor_base_task_.input_             = std::make_shared<ImageBuffer>(std::move(*bytes));
  editor_base_task_.pipeline_executor_ = editor_pipeline_guard_->pipeline_;
  editor_base_task_.options_.is_blocking_     = true;
  editor_base_task_.options_.is_callback_     = false;
  editor_base_task_.options_.is_seq_callback_ = false;
  editor_base_task_.options_.task_priority_   = 0;
  editor_base_task_.options_.render_desc_.render_type_ = RenderType::FAST_PREVIEW;

  auto exec = editor_pipeline_guard_->pipeline_;
  auto& loading = exec->GetStage(PipelineStageName::Image_Loading);

  nlohmann::json decodeParams;
#ifdef HAVE_CUDA
  decodeParams["raw"]["cuda"] = true;
#else
  decodeParams["raw"]["cuda"] = false;
#endif
  decodeParams["raw"]["highlights_reconstruct"] = true;
  decodeParams["raw"]["use_camera_wb"]          = true;
  decodeParams["raw"]["user_wb"]                = 7600.f;
  decodeParams["raw"]["backend"]                = "puerh";
  loading.SetOperator(OperatorType::RAW_DECODE, decodeParams);

  exec->SetExecutionStages();
}

void AlbumBackend::applyEditorStateToPipeline() {
  if (!editor_pipeline_guard_ || !editor_pipeline_guard_->pipeline_) {
    return;
  }

  auto exec          = editor_pipeline_guard_->pipeline_;
  auto& globalParams = exec->GetGlobalParams();

  auto& basic        = exec->GetStage(PipelineStageName::Basic_Adjustment);
  basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", editor_state_.exposure_}}, globalParams);
  basic.SetOperator(OperatorType::CONTRAST, {{"contrast", editor_state_.contrast_}}, globalParams);
  basic.SetOperator(OperatorType::BLACK, {{"black", editor_state_.blacks_}}, globalParams);
  basic.SetOperator(OperatorType::WHITE, {{"white", editor_state_.whites_}}, globalParams);
  basic.SetOperator(OperatorType::SHADOWS, {{"shadows", editor_state_.shadows_}}, globalParams);
  basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", editor_state_.highlights_}},
                    globalParams);

  auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
  color.SetOperator(OperatorType::SATURATION, {{"saturation", editor_state_.saturation_}},
                    globalParams);
  color.SetOperator(OperatorType::TINT, {{"tint", editor_state_.tint_}}, globalParams);
  color.SetOperator(OperatorType::LMT, {{"ocio_lmt", editor_state_.lut_path_}}, globalParams);

  auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
  detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", editor_state_.sharpen_}}}},
                     globalParams);
  detail.SetOperator(OperatorType::CLARITY, {{"clarity", editor_state_.clarity_}}, globalParams);

  editor_pipeline_guard_->dirty_ = true;
}

void AlbumBackend::queueEditorRender(RenderType renderType) {
  if (!editor_active_ || !editor_scheduler_ || !editor_pipeline_guard_) {
    return;
  }
  editor_pending_state_       = editor_state_;
  editor_pending_render_type_ = renderType;
  editor_has_pending_render_  = true;

  if (!editor_busy_) {
    editor_busy_ = true;
    emit editorStateChanged();
  }

  if (!editor_render_inflight_) {
    startNextEditorRender();
  }
}

void AlbumBackend::startNextEditorRender() {
  if (!editor_has_pending_render_ || !editor_scheduler_ || !editor_pipeline_guard_ ||
      !editor_base_task_.pipeline_executor_) {
    return;
  }

  editor_has_pending_render_ = false;
  editor_state_              = editor_pending_state_;

  try {
    applyEditorStateToPipeline();
  } catch (...) {
    editor_status_ = "Failed to apply editor pipeline state.";
    editor_busy_   = false;
    emit editorStateChanged();
    return;
  }

  PipelineTask task                       = editor_base_task_;
  task.options_.render_desc_.render_type_ = editor_pending_render_type_;
  task.options_.is_blocking_              = true;
  task.options_.is_callback_              = false;
  task.options_.is_seq_callback_          = false;
  task.options_.task_priority_            = 0;

  auto promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
  auto future  = promise->get_future();
  task.result_ = std::move(promise);

  editor_render_inflight_ = true;
  editor_status_          = "Rendering preview...";
  emit editorStateChanged();

  editor_scheduler_->ScheduleTask(std::move(task));
  editor_render_future_ = std::move(future);
  ensureEditorPollTimer();
  if (editor_poll_timer_ && !editor_poll_timer_->isActive()) {
    editor_poll_timer_->start();
  }
}

void AlbumBackend::pollEditorRender() {
  if (!editor_render_future_.has_value()) {
    if (editor_poll_timer_ && editor_poll_timer_->isActive() && !editor_render_inflight_) {
      editor_poll_timer_->stop();
    }
    return;
  }

  if (editor_render_future_->wait_for(0ms) != std::future_status::ready) {
    return;
  }

  std::shared_ptr<ImageBuffer> result;
  try {
    result = editor_render_future_->get();
  } catch (...) {
    result.reset();
  }
  editor_render_future_.reset();
  editor_render_inflight_ = false;

  if (!updateEditorPreviewFromBuffer(result)) {
    editor_status_ = "Preview render did not produce an image.";
  } else {
    editor_status_ = "Preview ready.";
  }

  if (editor_has_pending_render_) {
    startNextEditorRender();
    return;
  }

  editor_busy_ = false;
  emit editorStateChanged();

  if (editor_poll_timer_ && editor_poll_timer_->isActive()) {
    editor_poll_timer_->stop();
  }
}

void AlbumBackend::ensureEditorPollTimer() {
  if (editor_poll_timer_) {
    return;
  }
  editor_poll_timer_ = new QTimer(this);
  editor_poll_timer_->setInterval(16);
  connect(editor_poll_timer_, &QTimer::timeout, this, [this]() { pollEditorRender(); });
}

void AlbumBackend::finalizeEditorSession(bool persistChanges) {
  if (!editor_pipeline_guard_) {
    editor_active_ = false;
    editor_busy_   = false;
    return;
  }

  if (editor_render_future_.has_value()) {
    try {
      editor_render_future_->wait();
      auto last = editor_render_future_->get();
      (void)updateEditorPreviewFromBuffer(last);
    } catch (...) {
    }
    editor_render_future_.reset();
  }

  editor_has_pending_render_ = false;
  editor_render_inflight_    = false;
  if (editor_poll_timer_ && editor_poll_timer_->isActive()) {
    editor_poll_timer_->stop();
  }

  const auto finishedElement = editor_element_id_;
  const auto finishedImage   = editor_image_id_;

  if (pipeline_service_) {
    try {
      if (persistChanges) {
        applyEditorStateToPipeline();
        editor_pipeline_guard_->dirty_ = true;
      } else {
        editor_pipeline_guard_->dirty_ = false;
      }
      pipeline_service_->SavePipeline(editor_pipeline_guard_);
      if (persistChanges) {
        pipeline_service_->Sync();
      }
    } catch (...) {
    }
  }

  if (persistChanges && project_) {
    try {
      project_->GetImagePoolService()->SyncWithStorage();
      project_->SaveProject(meta_path_);
    } catch (...) {
    }
  }

  if (persistChanges && thumbnail_service_ && finishedElement != 0 && finishedImage != 0) {
    try {
      thumbnail_service_->InvalidateThumbnail(finishedElement);
    } catch (...) {
    }
    requestThumbnail(finishedElement, finishedImage);
  }

  editor_pipeline_guard_.reset();
  editor_base_task_   = PipelineTask{};
  editor_active_      = false;
  editor_busy_        = false;
  editor_element_id_  = 0;
  editor_image_id_    = 0;
  editor_title_.clear();
  editor_status_      = persistChanges ? "Edits saved." : "Editor closed.";
  if (!editor_preview_url_.isEmpty()) {
    editor_preview_url_.clear();
    emit editorPreviewChanged();
  }
  emit editorStateChanged();
}

auto AlbumBackend::updateEditorPreviewFromBuffer(const std::shared_ptr<ImageBuffer>& buffer) -> bool {
  if (!buffer) {
    return false;
  }

  QString dataUrl;
  try {
    if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
      buffer->SyncToCPU();
    }
    if (!buffer->cpu_data_valid_) {
      return false;
    }

    QImage image = MatRgba32fToQImageCopy(buffer->GetCPUData());
    if (image.isNull()) {
      return false;
    }
    QImage scaled = image.scaled(1180, 760, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    dataUrl = DataUrlFromImage(scaled);
  } catch (...) {
    return false;
  }

  if (dataUrl.isEmpty()) {
    return false;
  }

  if (editor_preview_url_ != dataUrl) {
    editor_preview_url_ = dataUrl;
    emit editorPreviewChanged();
  }
  return true;
}

void AlbumBackend::setEditorAdjustment(float& field, double value, double minValue, double maxValue) {
  if (!editor_active_) {
    return;
  }
  const float clamped = ClampToRange(value, minValue, maxValue);
  if (NearlyEqual(field, clamped)) {
    return;
  }
  field = clamped;
  emit editorStateChanged();
  queueEditorRender(RenderType::FAST_PREVIEW);
}

void AlbumBackend::setTaskState(const QString& status, int progress, bool cancelVisible) {
  task_status_         = status;
  task_progress_       = std::clamp(progress, 0, 100);
  task_cancel_visible_ = cancelVisible;
  emit taskStateChanged();
}

}  // namespace puerhlab::demo
