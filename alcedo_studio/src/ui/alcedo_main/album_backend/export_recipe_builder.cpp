//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/album_backend/export_recipe_builder.hpp"

#include <QLocale>
#include <QMetaType>
#include <cmath>
#include <limits>
#include <utility>

#include "io/image/export_recipe.hpp"

namespace alcedo::ui {
namespace {

constexpr double kNotANumber = std::numeric_limits<double>::quiet_NaN();

auto             IsIntegerInRange(double value, int minimum, int maximum) -> bool {
  return std::isfinite(value) && value == std::floor(value) && value >= minimum && value <= maximum;
}

auto RangeError(const QString& field, int minimum, int maximum) -> ExportRecipeError {
  return {field, QStringLiteral("%1 must be in [%2, %3]").arg(field).arg(minimum).arg(maximum)};
}

auto EnumError(const QString& field, const QStringList& allowed) -> ExportRecipeError {
  return {field, QStringLiteral("%1 must be one of: %2").arg(field, allowed.join(", "))};
}

auto IsText(const QVariant& value) -> bool { return value.typeId() == QMetaType::QString; }

/// A pixel size or dpi value: a number, or panel text read like JavaScript Number(text.trim()).
auto ReadNumber(const QVariant& value) -> double {
  if (!IsText(value)) {
    bool         ok     = false;
    const double result = value.toDouble(&ok);
    return ok ? result : kNotANumber;
  }
  const QString text   = value.toString().trimmed();
  bool          ok     = false;
  const double  result = text.toDouble(&ok);
  return ok ? result : kNotANumber;
}

/// A print size: a number, or panel text in the default locale (Number.fromLocaleString).
auto ReadLocaleNumber(const QVariant& value) -> double {
  if (!IsText(value)) {
    return ReadNumber(value);
  }
  bool         ok     = false;
  const double result = QLocale().toDouble(value.toString().trimmed(), &ok);
  return ok ? result : kNotANumber;
}

/// The resize mode name of ImportExportHandler::BuildExportQueue.
auto HandlerResizeMode(const QString& resize_mode) -> QString {
  if (resize_mode == QStringLiteral("long_edge")) {
    return QStringLiteral("longEdge");
  }
  if (resize_mode == QStringLiteral("print")) {
    return QStringLiteral("physical");
  }
  return resize_mode;
}

auto ErrorMap(const std::optional<ExportRecipeError>& error, const QString& ok_key) -> QVariantMap {
  if (!error.has_value()) {
    return QVariantMap{{ok_key, true}, {"field", QString()}, {"message", QString()}};
  }
  return QVariantMap{{ok_key, false}, {"field", error->field}, {"message", error->message}};
}

}  // namespace

ExportRecipeBuilder::ExportRecipeBuilder(ImportExportHandler* import_export, QObject* parent)
    : QObject(parent), import_export_(import_export) {}

auto ExportRecipeBuilder::Defaults() -> ExportRecipeOptions { return ExportRecipeOptions{}; }

auto ExportRecipeBuilder::Formats() -> QStringList {
  return {QStringLiteral("JPEG"), QStringLiteral("PNG"), QStringLiteral("TIFF"),
          QStringLiteral("EXR")};
}

auto ExportRecipeBuilder::AllowedBitDepths(const QString& format) -> QList<int> {
  if (format == QStringLiteral("JPEG")) return {8};
  if (format == QStringLiteral("PNG")) return {8, 16};
  if (format == QStringLiteral("TIFF")) return {8, 16, 32};
  if (format == QStringLiteral("EXR")) return {16, 32};
  return {};
}

auto ExportRecipeBuilder::PreferredBitDepth(const QString& format) -> int {
  return format == QStringLiteral("JPEG") ? 8 : 16;
}

auto ExportRecipeBuilder::TiffCompressions() -> QStringList {
  return {QStringLiteral("NONE"), QStringLiteral("LZW"), QStringLiteral("ZIP")};
}

auto ExportRecipeBuilder::ResizeModes() -> QStringList {
  return {QStringLiteral("original"), QStringLiteral("long_edge"), QStringLiteral("bounds"),
          QStringLiteral("print")};
}

auto ExportRecipeBuilder::PrintUnits() -> QStringList {
  return {QStringLiteral("mm"), QStringLiteral("cm"), QStringLiteral("in")};
}

auto ExportRecipeBuilder::CheckFileNamePattern(const QString& pattern) -> std::optional<QString> {
  const auto parsed = ParseExportFileNamePattern(pattern.toStdWString());
  if (parsed.success_) {
    return std::nullopt;
  }
  return QString::fromStdString(parsed.message_);
}

auto ExportRecipeBuilder::Validate(const ExportRecipeOptions& options)
    -> std::optional<ExportRecipeError> {
  if (!Formats().contains(options.format)) {
    return EnumError(QStringLiteral("format"), Formats());
  }
  const QList<int> bit_depths = AllowedBitDepths(options.format);
  if (!bit_depths.contains(options.bit_depth)) {
    QStringList allowed;
    for (const int depth : bit_depths) {
      allowed.push_back(QString::number(depth));
    }
    return ExportRecipeError{QStringLiteral("bit_depth"),
                             QStringLiteral("bit_depth for %1 must be one of: %2")
                                 .arg(options.format, allowed.join(", "))};
  }
  if (options.quality < kMinQuality || options.quality > kMaxQuality) {
    return RangeError(QStringLiteral("quality"), kMinQuality, kMaxQuality);
  }
  if (options.ultra_hdr_quality < kMinQuality || options.ultra_hdr_quality > kMaxQuality) {
    return RangeError(QStringLiteral("ultra_hdr_quality"), kMinQuality, kMaxQuality);
  }
  if (options.png_level < kMinPngLevel || options.png_level > kMaxPngLevel) {
    return RangeError(QStringLiteral("png_level"), kMinPngLevel, kMaxPngLevel);
  }
  if (!TiffCompressions().contains(options.tiff_compression)) {
    return EnumError(QStringLiteral("tiff_compression"), TiffCompressions());
  }
  if (!ResizeModes().contains(options.resize_mode)) {
    return EnumError(QStringLiteral("resize_mode"), ResizeModes());
  }
  if (options.resize_mode == QStringLiteral("long_edge") &&
      !IsIntegerInRange(options.long_edge, kMinLongEdge, kMaxLongEdge)) {
    return RangeError(QStringLiteral("long_edge"), kMinLongEdge, kMaxLongEdge);
  }
  if (options.resize_mode == QStringLiteral("bounds")) {
    if (!IsIntegerInRange(options.width, kMinPixelBound, kMaxPixelBound)) {
      return RangeError(QStringLiteral("width"), kMinPixelBound, kMaxPixelBound);
    }
    if (!IsIntegerInRange(options.height, kMinPixelBound, kMaxPixelBound)) {
      return RangeError(QStringLiteral("height"), kMinPixelBound, kMaxPixelBound);
    }
  }
  if (options.resize_mode == QStringLiteral("print")) {
    for (const auto& [field, value] :
         {std::pair{QStringLiteral("print_width"), options.print_width},
          std::pair{QStringLiteral("print_height"), options.print_height}}) {
      if (!std::isfinite(value) || value <= 0.0 || value > kMaxPrintSize) {
        return ExportRecipeError{field, QStringLiteral("%1 must be more than 0 and at most %2")
                                            .arg(field)
                                            .arg(static_cast<int>(kMaxPrintSize))};
      }
    }
    if (!IsIntegerInRange(options.dpi, kMinDpi, kMaxDpi)) {
      return RangeError(QStringLiteral("dpi"), kMinDpi, kMaxDpi);
    }
    if (!PrintUnits().contains(options.print_unit)) {
      return EnumError(QStringLiteral("print_unit"), PrintUnits());
    }
  }
  if (const auto pattern_error = CheckFileNamePattern(options.file_name_pattern)) {
    return ExportRecipeError{QStringLiteral("file_name_pattern"), *pattern_error};
  }
  return std::nullopt;
}

auto ExportRecipeBuilder::OutputFolder(const ExportRecipeOptions& options) -> QString {
  const QString subfolder = options.subfolder.trimmed();
  if (!options.subfolder_enabled || subfolder.isEmpty()) {
    return options.out_dir;
  }
  QString base = options.out_dir;
  while (base.endsWith(QLatin1Char('/')) || base.endsWith(QLatin1Char('\\'))) {
    base.chop(1);
  }
  return base + QLatin1Char('/') + subfolder;
}

auto ExportRecipeBuilder::FromVariantMap(const QVariantMap& values) -> ExportRecipeOptions {
  ExportRecipeOptions options;
  const auto          text = [&values](const char* key, QString* target) {
    if (values.contains(key)) *target = values.value(key).toString();
  };
  const auto integer = [&values](const char* key, int* target) {
    if (values.contains(key)) *target = values.value(key).toInt();
  };
  const auto flag = [&values](const char* key, bool* target) {
    if (values.contains(key)) *target = values.value(key).toBool();
  };
  const auto number = [&values](const char* key, double* target) {
    if (values.contains(key)) *target = ReadNumber(values.value(key));
  };
  const auto locale_number = [&values](const char* key, double* target) {
    if (values.contains(key)) *target = ReadLocaleNumber(values.value(key));
  };
  text("format", &options.format);
  integer("bit_depth", &options.bit_depth);
  integer("quality", &options.quality);
  integer("png_level", &options.png_level);
  text("tiff_compression", &options.tiff_compression);
  flag("include_metadata", &options.include_metadata);
  flag("embed_icc_profile", &options.embed_icc_profile);
  integer("ultra_hdr_quality", &options.ultra_hdr_quality);
  flag("ultra_hdr_dither", &options.ultra_hdr_dither);
  text("out_dir", &options.out_dir);
  flag("subfolder_enabled", &options.subfolder_enabled);
  text("subfolder", &options.subfolder);
  text("resize_mode", &options.resize_mode);
  number("long_edge", &options.long_edge);
  number("width", &options.width);
  number("height", &options.height);
  locale_number("print_width", &options.print_width);
  locale_number("print_height", &options.print_height);
  text("print_unit", &options.print_unit);
  number("dpi", &options.dpi);
  text("file_name_pattern", &options.file_name_pattern);
  return options;
}

auto ExportRecipeBuilder::ToVariantMap(const ExportRecipeOptions& options) -> QVariantMap {
  return QVariantMap{{"format", options.format},
                     {"bit_depth", options.bit_depth},
                     {"quality", options.quality},
                     {"png_level", options.png_level},
                     {"tiff_compression", options.tiff_compression},
                     {"include_metadata", options.include_metadata},
                     {"embed_icc_profile", options.embed_icc_profile},
                     {"ultra_hdr_quality", options.ultra_hdr_quality},
                     {"ultra_hdr_dither", options.ultra_hdr_dither},
                     {"out_dir", options.out_dir},
                     {"subfolder_enabled", options.subfolder_enabled},
                     {"subfolder", options.subfolder},
                     {"resize_mode", options.resize_mode},
                     {"long_edge", options.long_edge},
                     {"width", options.width},
                     {"height", options.height},
                     {"print_width", options.print_width},
                     {"print_height", options.print_height},
                     {"print_unit", options.print_unit},
                     {"dpi", options.dpi},
                     {"file_name_pattern", options.file_name_pattern}};
}

auto ExportRecipeBuilder::Start(const ExportRecipeOptions& options, const QVariantList& targets)
    -> std::optional<ExportRecipeError> {
  if (auto error = Validate(options)) {
    return error;
  }
  if (import_export_ == nullptr) {
    return ExportRecipeError{QStringLiteral("export"),
                             QStringLiteral("The export handler is not available.")};
  }
  // The argument form of the former export panel: the SDR long edge is set only in the long
  // edge mode, and every mode except the original size enables the SDR resize.
  const bool resize_enabled = options.resize_mode != QStringLiteral("original");
  const int  long_edge =
      options.resize_mode == QStringLiteral("long_edge") ? static_cast<int>(options.long_edge) : 0;
  QVariantMap recipe{
      {"includeMetadata", options.include_metadata},
      {"embedIccProfile", options.embed_icc_profile},
      {"fileNamePattern", options.file_name_pattern},
      {"resizeMode", HandlerResizeMode(options.resize_mode)},
      {"widthPixels", std::isfinite(options.width) ? static_cast<int>(options.width) : 0},
      {"heightPixels", std::isfinite(options.height) ? static_cast<int>(options.height) : 0},
      {"physicalWidth", options.print_width},
      {"physicalHeight", options.print_height},
      {"physicalUnit", options.print_unit},
      {"dpi", std::isfinite(options.dpi) ? static_cast<int>(options.dpi) : 0}};
  import_export_->StartExportWithRecipeOptionsForTargets(
      OutputFolder(options), resize_enabled, long_edge, kUltraHdrMaxLongEdge, options.format,
      options.quality, options.bit_depth, options.png_level, options.tiff_compression,
      options.ultra_hdr_quality, options.ultra_hdr_dither, recipe, targets);
  if (import_export_->export_inflight()) {
    return std::nullopt;
  }
  QString message = import_export_->ExportStatus();
  if (const QString summary = import_export_->ExportErrorSummary(); !summary.isEmpty()) {
    message += QLatin1Char(' ') + summary;
  }
  return ExportRecipeError{QStringLiteral("export"), message};
}

auto ExportRecipeBuilder::DefaultOptions() const -> QVariantMap { return ToVariantMap(Defaults()); }

auto ExportRecipeBuilder::AllowedBitDepthsFor(const QString& format) const -> QVariantList {
  QVariantList depths;
  for (const int depth : AllowedBitDepths(format)) {
    depths.push_back(depth);
  }
  return depths;
}

auto ExportRecipeBuilder::PreferredBitDepthFor(const QString& format) const -> int {
  return PreferredBitDepth(format);
}

auto ExportRecipeBuilder::IsFileNamePatternValid(const QString& pattern) const -> bool {
  return !CheckFileNamePattern(pattern).has_value();
}

auto ExportRecipeBuilder::ValidateOptions(const QVariantMap& options) const -> QVariantMap {
  return ErrorMap(Validate(FromVariantMap(options)), QStringLiteral("valid"));
}

auto ExportRecipeBuilder::OutputFolderFor(const QVariantMap& options) const -> QString {
  return OutputFolder(FromVariantMap(options));
}

auto ExportRecipeBuilder::StartExport(const QVariantMap& options, const QVariantList& targets)
    -> QVariantMap {
  return ErrorMap(Start(FromVariantMap(options), targets), QStringLiteral("started"));
}

}  // namespace alcedo::ui
