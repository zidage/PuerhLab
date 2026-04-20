//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/album_backend/image_controller.hpp"

#include "ui/alcedo_main/album_backend/album_backend.hpp"

#include <QCoreApplication>
#include <QStringList>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <numeric>
#include <unordered_set>

#include <json.hpp>

#include "ui/alcedo_main/album_backend/path_utils.hpp"

namespace alcedo::ui {

#define PL_TEXT(text, ...)                                                    \
  i18n::MakeLocalizedText(ALCEDO_I18N_CONTEXT,                              \
                          QT_TRANSLATE_NOOP(ALCEDO_I18N_CONTEXT, text)      \
                              __VA_OPT__(, ) __VA_ARGS__)

namespace {
using json = nlohmann::json;

auto ToVariantIdList(const std::vector<sl_element_id_t>& ids) -> QVariantList {
  QVariantList out;
  out.reserve(static_cast<qsizetype>(ids.size()));
  for (const auto id : ids) {
    out.push_back(static_cast<uint>(id));
  }
  return out;
}

auto DashValue() -> QString {
  return QString::fromUtf8("\u2014");
}

auto ToDisplayText(const std::string& value) -> QString {
  const QString text = QString::fromUtf8(value.c_str()).trimmed();
  return text.isEmpty() ? DashValue() : text;
}

auto ToOptionalDisplayText(const std::string& value) -> QString {
  return QString::fromUtf8(value.c_str()).trimmed();
}

auto FormatUnsigned(uint64_t value) -> QString {
  return value > 0 ? QString::number(value) : DashValue();
}

auto FormatFixed(double value, int precision, const QString& prefix = QString{},
                 const QString& suffix = QString{}) -> QString {
  if (!std::isfinite(value) || value <= 0.0) {
    return DashValue();
  }
  return prefix + QString::number(value, 'f', precision) + suffix;
}

auto FormatRating(int value) -> QString {
  return value > 0 ? QStringLiteral("%1/5").arg(value) : DashValue();
}

auto JsonNumberOrZero(const json& metadata, const char* key) -> double {
  if (!metadata.contains(key)) {
    return 0.0;
  }
  const auto& value = metadata.at(key);
  return value.is_number() ? value.get<double>() : 0.0;
}

auto JsonUnsignedOrZero(const json& metadata, const char* key) -> uint32_t {
  if (!metadata.contains(key)) {
    return 0;
  }
  const auto& value = metadata.at(key);
  return value.is_number_unsigned() ? value.get<uint32_t>()
         : value.is_number_integer() ? static_cast<uint32_t>(std::max<int64_t>(value.get<int64_t>(), 0))
                                     : 0;
}

auto JsonStringOrEmpty(const json& metadata, const char* key) -> std::string {
  if (!metadata.contains(key)) {
    return {};
  }
  const auto& value = metadata.at(key);
  return value.is_string() ? value.get<std::string>() : std::string{};
}

auto FormatAspectRatio(uint32_t width, uint32_t height) -> QString {
  if (width == 0 || height == 0) {
    return DashValue();
  }

  const double longer_edge  = static_cast<double>(std::max(width, height));
  const double shorter_edge = static_cast<double>(std::min(width, height));
  if (shorter_edge <= 0.0) {
    return DashValue();
  }

  const double normalized_ratio = longer_edge / shorter_edge;

  struct CommonAspectRatio {
    double      ratio;
    const char* label;
  };
  constexpr std::array<CommonAspectRatio, 4> kCommonAspectRatios = {
      CommonAspectRatio{3.0 / 2.0, "3:2"},
      CommonAspectRatio{1.85, "1.85:1"},
      CommonAspectRatio{1.79, "1.79:1"},
      CommonAspectRatio{4.0 / 3.0, "4:3"},
  };
  constexpr double kCommonRatioTolerance = 0.03;

  const auto nearest =
      std::min_element(kCommonAspectRatios.begin(), kCommonAspectRatios.end(),
                       [normalized_ratio](const CommonAspectRatio& lhs,
                                          const CommonAspectRatio& rhs) {
                         return std::abs(normalized_ratio - lhs.ratio) <
                                std::abs(normalized_ratio - rhs.ratio);
                       });
  if (nearest != kCommonAspectRatios.end() &&
      std::abs(normalized_ratio - nearest->ratio) <= kCommonRatioTolerance) {
    return QString::fromLatin1(nearest->label);
  }

  return QStringLiteral("%1:1").arg(QString::number(normalized_ratio, 'f', 2));
}

auto FormatDimensions(uint32_t width, uint32_t height) -> QString {
  if (width == 0 || height == 0) {
    return DashValue();
  }
  return QStringLiteral("%1 × %2 px").arg(width).arg(height);
}

auto FormatShutterSpeed(const json& metadata) -> QString {
  if (!metadata.contains("ShutterSpeed")) {
    return DashValue();
  }
  const auto& value = metadata.at("ShutterSpeed");
  if (!value.is_array() || value.size() < 2 || !value[0].is_number_integer() ||
      !value[1].is_number_integer()) {
    return DashValue();
  }

  const int64_t numerator   = value[0].get<int64_t>();
  const int64_t denominator = value[1].get<int64_t>();
  if (numerator <= 0 || denominator <= 0) {
    return DashValue();
  }
  if (denominator == 1) {
    return QStringLiteral("%1 s").arg(numerator);
  }
  return QStringLiteral("%1/%2 s").arg(numerator).arg(denominator);
}

auto MakeDetailsRow(const QString& section, const QString& label, const QString& value,
                    bool emphasized = false, const QString& actionId = QString{},
                    const QString& actionValue = QString{},
                    const QString& actionTooltip = QString{}) -> QVariantMap {
  return QVariantMap{{"section", section},
                     {"label", label},
                     {"value", value},
                     {"emphasized", emphasized},
                     {"actionId", actionId},
                     {"actionValue", actionValue},
                     {"actionTooltip", actionTooltip}};
}

void AppendDetailsRow(QVariantList& rows, const QString& section, const QString& label,
                      const QString& value, bool emphasized = false,
                      const QString& actionId = QString{},
                      const QString& actionValue = QString{},
                      const QString& actionTooltip = QString{}) {
  rows.push_back(
      MakeDetailsRow(section, label, value, emphasized, actionId, actionValue, actionTooltip));
}

struct SourceDirectoryInfo {
  QString displayText = DashValue();
  QString pathText{};
  bool    canOpen = false;
};

auto ResolveSourceDirectory(const std::shared_ptr<Image>& image) -> SourceDirectoryInfo {
  if (!image || image->image_path_.empty()) {
    return {};
  }

  const std::filesystem::path directory = image->image_path_.parent_path().lexically_normal();
  if (directory.empty()) {
    return {};
  }

  const QString pathText = album_util::PathToQString(directory);
  if (pathText.trimmed().isEmpty()) {
    return {};
  }

  return SourceDirectoryInfo{pathText, pathText, true};
}

auto ComposeSubtitle(const json& metadata) -> QString {
  const QString camera = ToOptionalDisplayText(JsonStringOrEmpty(metadata, "Model"));
  const QString lens   = ToOptionalDisplayText(JsonStringOrEmpty(metadata, "Lens"));

  QStringList parts;
  if (!camera.isEmpty()) {
    parts.push_back(camera);
  }
  if (!lens.isEmpty()) {
    parts.push_back(lens);
  }
  return parts.join(QStringLiteral(" · "));
}

auto ResolveTitle(const AlbumItem* item, const std::shared_ptr<Image>& image) -> QString {
  if (item && !item->file_name.trimmed().isEmpty()) {
    return item->file_name.trimmed();
  }
  if (image && !image->image_name_.empty()) {
    const QString from_image = album_util::WStringToQString(image->image_name_).trimmed();
    if (!from_image.isEmpty()) {
      return from_image;
    }
  }
  return Tr("(unnamed)");
}

auto ParseExifDisplayJson(const std::shared_ptr<Image>& image) -> json {
  if (!image) {
    return json::object();
  }
  try {
    const std::string exif_text = image->ExifToJson();
    if (exif_text.empty()) {
      return json::object();
    }
    const json parsed = json::parse(exif_text, nullptr, false);
    return parsed.is_discarded() ? json::object() : parsed;
  } catch (...) {
    return json::object();
  }
}

auto BuildDetailsResult(const AlbumItem* item, const std::shared_ptr<Image>& image) -> QVariantMap {
  const json metadata = ParseExifDisplayJson(image);
  const QString section_capture  = Tr("Capture");
  const QString section_gear     = Tr("Gear");
  const QString section_exposure = Tr("Exposure");
  const QString section_storage  = Tr("Storage");
  const uint32_t width           = JsonUnsignedOrZero(metadata, "ImageWidth");
  const uint32_t height          = JsonUnsignedOrZero(metadata, "ImageHeight");
  const SourceDirectoryInfo source_directory = ResolveSourceDirectory(image);

  QVariantList rows;
  rows.reserve(15);

  AppendDetailsRow(rows, section_capture, Tr("Original Size"), FormatDimensions(width, height), true);
  AppendDetailsRow(rows, section_capture, Tr("Original Aspect Ratio"),
                   FormatAspectRatio(width, height));
  AppendDetailsRow(rows, section_capture, Tr("Captured At"),
                   ToDisplayText(JsonStringOrEmpty(metadata, "DateTimeString")));

  AppendDetailsRow(rows, section_gear, Tr("Camera Brand"),
                   ToDisplayText(JsonStringOrEmpty(metadata, "Make")));
  AppendDetailsRow(rows, section_gear, Tr("Camera Model"),
                   ToDisplayText(JsonStringOrEmpty(metadata, "Model")), true);
  AppendDetailsRow(rows, section_gear, Tr("Lens Brand"),
                   ToDisplayText(JsonStringOrEmpty(metadata, "LensMake")));
  AppendDetailsRow(rows, section_gear, Tr("Lens Model"),
                   ToDisplayText(JsonStringOrEmpty(metadata, "Lens")), true);

  AppendDetailsRow(rows, section_exposure, Tr("Aperture"),
                   FormatFixed(JsonNumberOrZero(metadata, "Aperture"), 1, "f/"));
  AppendDetailsRow(rows, section_exposure, Tr("Shutter"), FormatShutterSpeed(metadata));
  AppendDetailsRow(rows, section_exposure, Tr("ISO"),
                   FormatUnsigned(JsonUnsignedOrZero(metadata, "ISO")));
  AppendDetailsRow(rows, section_exposure, Tr("Focal Length"),
                   FormatFixed(JsonNumberOrZero(metadata, "FocalLength"), 0, QString{}, " mm"));
  AppendDetailsRow(rows, section_exposure, Tr("35mm Equivalent"),
                   FormatFixed(JsonNumberOrZero(metadata, "FocalLength35mm"), 0, QString{},
                               " mm"));
  AppendDetailsRow(rows, section_exposure, Tr("Focus Distance"),
                   FormatFixed(JsonNumberOrZero(metadata, "FocusDistanceM"), 2, QString{}, " m"));
  AppendDetailsRow(rows, section_exposure, Tr("Rating"),
                   FormatRating(static_cast<int>(JsonUnsignedOrZero(metadata, "Rating"))));
  AppendDetailsRow(rows, section_storage, Tr("Source Directory"), source_directory.displayText,
                   false, source_directory.canOpen ? QStringLiteral("open-directory") : QString{},
                   source_directory.pathText,
                   source_directory.canOpen ? Tr("Open in file manager") : QString{});

  return QVariantMap{{"success", true},
                     {"message", QString{}},
                     {"title", ResolveTitle(item, image)},
                     {"subtitle", ComposeSubtitle(metadata)},
                     {"rows", rows}};
}
}  // namespace

ImageController::ImageController(AlbumBackend& backend) : backend_(backend) {}

auto ImageController::CollectDeleteTargets(const QVariantList& targetEntries) const
    -> std::vector<DeleteTarget> {
  std::vector<DeleteTarget> targets;
  targets.reserve(static_cast<size_t>(targetEntries.size()));

  std::unordered_set<sl_element_id_t> seen_element_ids;
  seen_element_ids.reserve(static_cast<size_t>(targetEntries.size()) * 2 + 1);

  for (const QVariant& row_var : targetEntries) {
    const QVariantMap row = row_var.toMap();
    const auto element_id = static_cast<sl_element_id_t>(row.value("elementId").toUInt());
    if (element_id == 0 || !seen_element_ids.insert(element_id).second) {
      continue;
    }

    DeleteTarget target;
    target.element_id_ = element_id;
    target.image_id_   = static_cast<image_id_t>(row.value("imageId").toUInt());

    if (const auto* item = backend_.FindAlbumItem(element_id); item) {
      if (target.image_id_ == 0) {
        target.image_id_ = item->image_id;
      }
      target.file_path_ = item->file_path_;
    }

    targets.push_back(target);
  }

  return targets;
}

auto ImageController::DeleteImages(const QVariantList& targetEntries) -> QVariantMap {
  const auto delete_result = DeleteTargets(CollectDeleteTargets(targetEntries));

  QVariantMap result{{"success", false},
                     {"deletedCount", 0},
                     {"failedCount", 0},
                     {"deletedElementIds", QVariantList{}},
                     {"failedElementIds", QVariantList{}},
                     {"message", delete_result.message_}};

  result["success"]           = delete_result.success_;
  result["deletedCount"]      = delete_result.deleted_count_;
  result["failedCount"]       = delete_result.failed_count_;
  result["deletedElementIds"] = ToVariantIdList(delete_result.deleted_element_ids_);
  result["failedElementIds"]  = ToVariantIdList(delete_result.failed_element_ids_);
  return result;
}

auto ImageController::DeleteTargets(const std::vector<DeleteTarget>& targets)
    -> DeleteExecutionResult {
  DeleteExecutionResult result;

  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    const auto msg = PL_TEXT("Project is loading. Please wait.");
    backend_.SetTaskState(msg, 0, false);
    result.message_ = msg.Render();
    return result;
  }
  if (!ph.project()) {
    const auto msg = PL_TEXT("No project is loaded.");
    backend_.SetTaskState(msg, 0, false);
    result.message_ = msg.Render();
    return result;
  }

  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    const auto msg = PL_TEXT("Cannot delete images while import is running.");
    backend_.SetTaskState(msg, 0, false);
    result.message_ = msg.Render();
    return result;
  }
  if (ie.export_inflight()) {
    const auto msg = PL_TEXT("Cannot delete images while export is running.");
    backend_.SetTaskState(msg, 0, false);
    result.message_ = msg.Render();
    return result;
  }

  if (targets.empty()) {
    const auto msg = PL_TEXT("No valid images selected for deletion.");
    backend_.SetTaskState(msg, 0, false);
    result.message_ = msg.Render();
    return result;
  }

  std::unordered_set<sl_element_id_t> target_ids;
  target_ids.reserve(targets.size() * 2 + 1);
  std::vector<std::filesystem::path> delete_paths;
  delete_paths.reserve(targets.size());
  std::vector<DeleteTarget> resolved_targets = targets;
  for (auto& target : resolved_targets) {
    if (target.image_id_ == 0 || target.file_path_.empty()) {
      if (const auto* item = backend_.FindAlbumItem(target.element_id_); item) {
        if (target.image_id_ == 0) {
          target.image_id_ = item->image_id;
        }
        if (target.file_path_.empty()) {
          target.file_path_ = item->file_path_;
        }
      }
    }

    target_ids.insert(target.element_id_);
    if (!target.file_path_.empty()) {
      delete_paths.push_back(target.file_path_);
    }
  }

  if (backend_.editor_.editor_active() &&
      target_ids.contains(backend_.editor_.editor_element_id())) {
    backend_.editor_.FinalizeEditorSession(true);
  }

  auto proj       = ph.project();
  auto browse     = proj->GetAlbumBrowseService();
  auto image_pool = proj->GetImagePoolService();
  auto export_svc = ph.export_service();
  auto pipeline_svc = ph.pipeline_service();
  auto history_svc = ph.history_service();

  if (!browse) {
    const auto msg = PL_TEXT("Image service is unavailable.");
    backend_.SetTaskState(msg, 0, false);
    result.message_ = msg.Render();
    return result;
  }

  const auto delete_result = browse->DeleteFiles(delete_paths);
  std::vector<sl_element_id_t> deleted_ids;
  deleted_ids.reserve(delete_result.deleted_files_.size());
  for (const auto& file : delete_result.deleted_files_) {
    deleted_ids.push_back(file.element_id_);
  }

  std::vector<sl_element_id_t> failed_ids;
  failed_ids.reserve(resolved_targets.size());
  for (const auto& target : resolved_targets) {
    if (target.file_path_.empty()) {
      failed_ids.push_back(target.element_id_);
    }
  }
  for (const auto& path : delete_result.failed_paths_) {
    const auto it = std::find_if(resolved_targets.begin(), resolved_targets.end(),
                                 [&path](const DeleteTarget& target) {
      return target.file_path_.lexically_normal() == path.lexically_normal();
    });
    if (it != resolved_targets.end()) {
      failed_ids.push_back(it->element_id_);
    }
  }

  bool image_pool_dirty = false;
  for (const auto& target : resolved_targets) {
    if (std::find(deleted_ids.begin(), deleted_ids.end(), target.element_id_) ==
        deleted_ids.end()) {
      continue;
    }

    try {
      backend_.thumb_.RemoveThumbnailState(target.element_id_, target.image_id_);
    } catch (...) {
    }

    if (export_svc) {
      try {
        export_svc->RemoveExportTask(target.element_id_);
      } catch (...) {
      }
    }
    if (pipeline_svc) {
      try {
        pipeline_svc->DeletePipeline(target.element_id_);
      } catch (...) {
      }
    }
    if (history_svc) {
      try {
        history_svc->DeleteHistory(target.element_id_);
      } catch (...) {
      }
    }
    if (image_pool && target.image_id_ != 0) {
      try {
        image_pool->Remove(target.image_id_);
        image_pool_dirty = true;
      } catch (...) {
      }
    }
  }

  bool save_ok = true;
  if (image_pool_dirty && image_pool) {
    try {
      image_pool->SyncWithStorage();
    } catch (...) {
      save_ok = false;
    }
  }

  if (!deleted_ids.empty()) {
    try {
      if (!ph.meta_path().empty()) {
        proj->SaveProject(ph.meta_path());
      }
      QString ignored_error;
      if (!ph.PackageCurrentProjectFiles(&ignored_error)) {
        save_ok = false;
      }
    } catch (...) {
      save_ok = false;
    }
  }

  if (!deleted_ids.empty()) {
    backend_.ReloadCurrentFolder();
  }

  const int deleted_count = static_cast<int>(deleted_ids.size());
  const int failed_count  = static_cast<int>(failed_ids.size());

  auto msg = i18n::LocalizedText{};
  if (deleted_count == 0) {
    msg = PL_TEXT("No images were deleted.");
  } else if (failed_count == 0) {
    msg = PL_TEXT("Deleted %1 image(s).", deleted_count);
  } else {
    msg = PL_TEXT("Deleted %1 image(s); %2 failed.", deleted_count, failed_count);
  }
  if (!save_ok) {
    msg = PL_TEXT("%1 Project state save failed.", msg.Render());
  }

  backend_.SetServiceMessageForCurrentProject(msg);
  backend_.SetTaskState(msg, deleted_count > 0 ? 100 : 0, false);
  if (deleted_count > 0) {
    backend_.ScheduleIdleTaskStateReset(1500);
  }

  result.success_             = deleted_count > 0;
  result.deleted_count_       = deleted_count;
  result.failed_count_        = failed_count;
  result.deleted_element_ids_ = std::move(deleted_ids);
  result.failed_element_ids_  = std::move(failed_ids);
  result.message_             = msg.Render();
  return result;
}

auto ImageController::GetImageDetails(uint elementId, uint imageId) -> QVariantMap {
  QVariantMap result{{"success", false},
                     {"message", QString{}},
                     {"title", QString{}},
                     {"subtitle", QString{}},
                     {"rows", QVariantList{}}};

  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    const auto msg = PL_TEXT("Project is loading. Please wait.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }
  if (!ph.project()) {
    const auto msg = PL_TEXT("No project is loaded.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }

  image_id_t resolved_image_id = static_cast<image_id_t>(imageId);
  const auto resolved_element_id = static_cast<sl_element_id_t>(elementId);
  const auto* item =
      resolved_element_id != 0 ? backend_.FindAlbumItem(resolved_element_id) : nullptr;
  if (resolved_image_id == 0 && item) {
    resolved_image_id = item->image_id;
  }
  if (resolved_image_id == 0) {
    const auto msg = PL_TEXT("No valid image was selected.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }

  auto image_pool = ph.project()->GetImagePoolService();
  if (!image_pool) {
    const auto msg = PL_TEXT("Image service is unavailable.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }

  try {
    return image_pool->Read<QVariantMap>(
        resolved_image_id,
        [item](const std::shared_ptr<Image>& image) { return BuildDetailsResult(item, image); });
  } catch (const std::exception&) {
    const auto msg = PL_TEXT("Failed to load image details.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  } catch (...) {
    const auto msg = PL_TEXT("Failed to load image details.");
    backend_.SetTaskState(msg, 0, false);
    result["message"] = msg.Render();
    return result;
  }
}

}  // namespace alcedo::ui

#undef PL_TEXT
