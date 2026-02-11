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
#include <QRandomGenerator>
#include <QStandardPaths>
#include <QTimer>
#include <QUrl>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cwctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include <duckdb.h>
#include <json.hpp>
#include <opencv2/opencv.hpp>

#include "EditorDialog.h"
#include "app/render_service.hpp"
#include "image/image.hpp"
#include "io/image/image_loader.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "type/supported_file_type.hpp"
#include "utils/string/convert.hpp"

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

auto FolderPathToDisplay(const std::filesystem::path& path) -> QString {
  if (path.empty()) {
    return "/";
  }
#if defined(_WIN32)
  const QString text = QString::fromStdWString(path.generic_wstring());
#else
  const QString text = QString::fromStdString(path.generic_string());
#endif
  if (text == "/") {
    return text;
  }
  return text.startsWith('/') ? text : ("/" + text);
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

constexpr std::array<const char*, 6> kThumbnailAccentPalette = {
    "#5AA2FF",
    "#4CC9A6",
    "#F7B267",
    "#E08BFF",
    "#7AD1FF",
    "#9BD65B",
};

auto AccentForIndex(size_t index) -> QString {
  return QString::fromLatin1(kThumbnailAccentPalette[index % kThumbnailAccentPalette.size()]);
}

auto ExportTargetKey(sl_element_id_t elementId, image_id_t imageId) -> uint64_t {
  return (static_cast<uint64_t>(elementId) << 32U) | static_cast<uint64_t>(imageId);
}

constexpr std::wstring_view kPackedProjectExtension = L".puerhproj";
constexpr std::array<char, 8> kPackedProjectMagic{
    {'P', 'U', 'E', 'R', 'H', 'P', 'K', '1'}};
constexpr uint32_t kPackedProjectVersion = 1;
constexpr uint64_t kMaxPackedComponentBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

auto QStringToFsPath(const QString& text) -> std::filesystem::path {
#if defined(_WIN32)
  return std::filesystem::path(text.toStdWString());
#else
  return std::filesystem::path(text.toStdString());
#endif
}

void CleanupWorkspaceDirectory(const std::filesystem::path& dir) {
  if (dir.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

auto EnsureDirectoryExists(const std::filesystem::path& dir) -> bool {
  if (dir.empty()) {
    return true;
  }
  std::error_code ec;
  if (!std::filesystem::exists(dir, ec)) {
    std::filesystem::create_directories(dir, ec);
  }
  if (ec) {
    return false;
  }
  return std::filesystem::is_directory(dir, ec) && !ec;
}

auto IsMetadataJsonPath(const std::filesystem::path& path) -> bool {
  std::wstring ext = path.extension().wstring();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
  return ext == L".json";
}

auto IsPackedProjectPath(const std::filesystem::path& path) -> bool {
  std::wstring ext = path.extension().wstring();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
  return ext == kPackedProjectExtension;
}

auto ReadExact(std::istream& stream, char* data, std::streamsize size) -> bool {
  stream.read(data, size);
  return stream.good() || stream.gcount() == size;
}

auto WriteU32Le(std::ostream& stream, uint32_t value) -> bool {
  std::array<unsigned char, 4> bytes{};
  bytes[0] = static_cast<unsigned char>(value & 0xFFU);
  bytes[1] = static_cast<unsigned char>((value >> 8U) & 0xFFU);
  bytes[2] = static_cast<unsigned char>((value >> 16U) & 0xFFU);
  bytes[3] = static_cast<unsigned char>((value >> 24U) & 0xFFU);
  stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(stream);
}

auto WriteU64Le(std::ostream& stream, uint64_t value) -> bool {
  std::array<unsigned char, 8> bytes{};
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<unsigned char>((value >> (i * 8ULL)) & 0xFFULL);
  }
  stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(stream);
}

auto ReadU32Le(std::istream& stream, uint32_t* value) -> bool {
  std::array<unsigned char, 4> bytes{};
  if (!ReadExact(stream, reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()))) {
    return false;
  }
  *value = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
           (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
  return true;
}

auto ReadU64Le(std::istream& stream, uint64_t* value) -> bool {
  std::array<unsigned char, 8> bytes{};
  if (!ReadExact(stream, reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()))) {
    return false;
  }
  *value = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    *value |= (static_cast<uint64_t>(bytes[i]) << (i * 8ULL));
  }
  return true;
}

auto ReadFileBytes(const std::filesystem::path& path, std::string* out) -> bool {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) {
    return false;
  }
  in.seekg(0, std::ios::beg);
  out->assign(static_cast<size_t>(size), '\0');
  if (size == 0) {
    return true;
  }
  return ReadExact(in, out->data(), size);
}

auto WriteFileBytes(const std::filesystem::path& path, const std::string& data) -> bool {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    return false;
  }
  if (!data.empty()) {
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
  }
  return static_cast<bool>(out);
}

auto IsPackedProjectFile(const std::filesystem::path& path) -> bool {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return false;
  }
  std::array<char, kPackedProjectMagic.size()> magic{};
  if (!ReadExact(in, magic.data(), static_cast<std::streamsize>(magic.size()))) {
    return false;
  }
  return std::equal(magic.begin(), magic.end(), kPackedProjectMagic.begin());
}

auto ValidateProjectName(const QString& rawName, QString* errorOut) -> std::optional<QString> {
  const QString trimmed = rawName.trimmed();
  if (trimmed.isEmpty()) {
    if (errorOut) {
      *errorOut = "Project name cannot be empty.";
    }
    return std::nullopt;
  }

  if (trimmed == "." || trimmed == "..") {
    if (errorOut) {
      *errorOut = "Project name is invalid.";
    }
    return std::nullopt;
  }

  static const QString kInvalidChars = "<>:\"/\\|?*";
  for (const QChar ch : trimmed) {
    if (ch.unicode() < 32U || kInvalidChars.contains(ch)) {
      if (errorOut) {
        *errorOut = "Project name contains invalid characters.";
      }
      return std::nullopt;
    }
  }

  if (trimmed.endsWith(' ') || trimmed.endsWith('.')) {
    if (errorOut) {
      *errorOut = "Project name cannot end with a space or period.";
    }
    return std::nullopt;
  }

  return trimmed;
}

auto BuildUniquePackedProjectPath(const std::filesystem::path& folder, const QString& projectName,
                                  QString* errorOut) -> std::optional<std::filesystem::path> {
  std::error_code ec;
  if (!std::filesystem::exists(folder, ec)) {
    std::filesystem::create_directories(folder, ec);
  }
  if (ec || !std::filesystem::is_directory(folder, ec) || ec) {
    if (errorOut) {
      *errorOut = "Selected folder is invalid or not writable.";
    }
    return std::nullopt;
  }

  const auto valid_name = ValidateProjectName(projectName, errorOut);
  if (!valid_name.has_value()) {
    return std::nullopt;
  }

  QString base_name_text = valid_name.value();
  if (base_name_text.endsWith(".puerhproj", Qt::CaseInsensitive)) {
    base_name_text.chop(QString(".puerhproj").size());
    base_name_text = base_name_text.trimmed();
  }
  if (base_name_text.isEmpty()) {
    if (errorOut) {
      *errorOut = "Project name cannot be only an extension.";
    }
    return std::nullopt;
  }

  const std::wstring base_name = base_name_text.toStdWString();
  for (int index = 0; index < 1024; ++index) {
    std::wstring suffix;
    if (index > 0) {
      suffix = L"_" + std::to_wstring(index);
    }

    const auto packed_path = folder / (base_name + suffix + std::wstring(kPackedProjectExtension));
    ec.clear();
    const bool exists = std::filesystem::exists(packed_path, ec);
    if (ec) {
      continue;
    }
    if (!exists) {
      return packed_path;
    }
  }

  if (errorOut) {
    *errorOut = "A packed project with that name already exists.";
  }
  return std::nullopt;
}

auto BuildBundlePathFromMetaPath(const std::filesystem::path& metaPath) -> std::filesystem::path {
  if (metaPath.empty()) {
    return {};
  }
  return metaPath.parent_path() /
         (metaPath.stem().wstring() + std::wstring(kPackedProjectExtension));
}

auto CreateProjectWorkspace(const QString& projectName,
                            std::filesystem::path* workspaceOut,
                            QString*               errorOut) -> bool {
  const QString temp_dir_text =
      QStandardPaths::writableLocation(QStandardPaths::TempLocation).isEmpty()
          ? QDir::tempPath()
          : QStandardPaths::writableLocation(QStandardPaths::TempLocation);
  const auto temp_dir = QStringToFsPath(temp_dir_text);
  const auto root     = temp_dir / L"puerh_lab_album_editor_qml";
  if (!EnsureDirectoryExists(root)) {
    if (errorOut) {
      *errorOut = "Unable to create project temp directory.";
    }
    return false;
  }

  QString normalized_name = projectName.trimmed();
  if (normalized_name.isEmpty()) {
    normalized_name = "project";
  }
  for (QChar& ch : normalized_name) {
    if (!(ch.isLetterOrNumber() || ch == '_' || ch == '-')) {
      ch = '_';
    }
  }

  const auto stamp = static_cast<unsigned long long>(QDateTime::currentMSecsSinceEpoch());
  for (int attempt = 0; attempt < 128; ++attempt) {
    const auto nonce =
        static_cast<unsigned long long>(QRandomGenerator::global()->generate64());
    const std::wstring dir_name =
        QString("runtime_%1_%2_%3")
            .arg(normalized_name)
            .arg(QString::number(stamp))
            .arg(QString::number(nonce + static_cast<unsigned long long>(attempt)))
            .toStdWString();

    const auto candidate = root / dir_name;
    std::error_code ec;
    std::filesystem::create_directories(candidate, ec);
    if (!ec) {
      *workspaceOut = candidate;
      return true;
    }
  }

  if (errorOut) {
    *errorOut = "Unable to allocate a unique project temp directory.";
  }
  return false;
}

auto BuildRuntimeProjectPair(const std::filesystem::path& workspace, const QString& projectName)
    -> std::pair<std::filesystem::path, std::filesystem::path> {
  QString base = projectName.trimmed();
  if (base.isEmpty()) {
    base = "album_editor_project";
  }
  if (base.endsWith(".puerhproj", Qt::CaseInsensitive)) {
    base.chop(QString(".puerhproj").size());
    base = base.trimmed();
  }
  auto valid_name = ValidateProjectName(base, nullptr);
  if (valid_name.has_value()) {
    base = valid_name.value();
  } else {
    base = "album_editor_project";
  }
  const std::wstring stem = base.toStdWString();
  return {workspace / (stem + L".db"), workspace / (stem + L".json")};
}

auto EscapeSqlStringLiteral(const std::string& text) -> std::string {
  std::string escaped;
  escaped.reserve(text.size() + 8);
  for (const char ch : text) {
    if (ch == '\'') {
      escaped.push_back('\'');
    }
    escaped.push_back(ch);
  }
  return escaped;
}

auto EscapeSqlIdentifier(const std::string& text) -> std::string {
  std::string escaped;
  escaped.reserve(text.size() + 8);
  for (const char ch : text) {
    if (ch == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(ch);
  }
  return escaped;
}

auto RunDuckDbQuery(duckdb_connection conn, const std::string& sql, const char* stage,
                    QString* errorOut) -> bool {
  duckdb_result result;
  if (duckdb_query(conn, sql.c_str(), &result) != DuckDBSuccess) {
    const char* err_msg = duckdb_result_error(&result);
    if (errorOut) {
      *errorOut = QString("DuckDB %1 failed: %2")
                      .arg(QString::fromUtf8(stage))
                      .arg(QString::fromUtf8(err_msg ? err_msg : ""));
    }
    duckdb_destroy_result(&result);
    return false;
  }
  duckdb_destroy_result(&result);
  return true;
}

auto QueryCurrentCatalog(duckdb_connection conn, std::string* catalogOut,
                         QString* errorOut) -> bool {
  duckdb_result result;
  if (duckdb_query(conn, "SELECT current_catalog();", &result) != DuckDBSuccess) {
    const char* err_msg = duckdb_result_error(&result);
    if (errorOut) {
      *errorOut = QString("DuckDB query current catalog failed: %1")
                      .arg(QString::fromUtf8(err_msg ? err_msg : ""));
    }
    duckdb_destroy_result(&result);
    return false;
  }

  const idx_t row_count = duckdb_row_count(&result);
  const idx_t col_count = duckdb_column_count(&result);
  if (row_count == 0 || col_count == 0) {
    if (errorOut) {
      *errorOut = "DuckDB query current catalog returned no rows.";
    }
    duckdb_destroy_result(&result);
    return false;
  }

  const char* value = duckdb_value_varchar(&result, 0, 0);
  if (!value || value[0] == '\0') {
    if (errorOut) {
      *errorOut = "DuckDB query current catalog returned empty value.";
    }
    if (value) {
      duckdb_free(const_cast<char*>(value));
    }
    duckdb_destroy_result(&result);
    return false;
  }
  *catalogOut = value;
  duckdb_free(const_cast<char*>(value));
  duckdb_destroy_result(&result);
  return true;
}

auto BuildTempDbSnapshotPath(std::filesystem::path* snapshotPathOut, QString* errorOut) -> bool {
  const QString temp_dir_text =
      QStandardPaths::writableLocation(QStandardPaths::TempLocation).isEmpty()
          ? QDir::tempPath()
          : QStandardPaths::writableLocation(QStandardPaths::TempLocation);
  const auto temp_dir = QStringToFsPath(temp_dir_text);
  const auto root     = temp_dir / L"puerh_lab_album_editor_qml";
  if (!EnsureDirectoryExists(root)) {
    if (errorOut) {
      *errorOut = "Unable to prepare temp folder for DB snapshot.";
    }
    return false;
  }

  const auto stamp = static_cast<unsigned long long>(QDateTime::currentMSecsSinceEpoch());
  for (int attempt = 0; attempt < 128; ++attempt) {
    const auto nonce =
        static_cast<unsigned long long>(QRandomGenerator::global()->generate64());
    const auto candidate =
        root / QString("pack_snapshot_%1_%2.db")
                   .arg(QString::number(stamp))
                   .arg(QString::number(nonce + static_cast<unsigned long long>(attempt)))
                   .toStdWString();
    std::error_code ec;
    if (!std::filesystem::exists(candidate, ec) || ec) {
      *snapshotPathOut = candidate;
      return true;
    }
  }

  if (errorOut) {
    *errorOut = "Unable to allocate temp DB snapshot path.";
  }
  return false;
}

auto CreateLiveDbSnapshot(const std::shared_ptr<ProjectService>& project,
                          const std::filesystem::path&          snapshotPath,
                          QString*                              errorOut) -> bool {
  if (!project) {
    if (errorOut) {
      *errorOut = "No active project for DB snapshot.";
    }
    return false;
  }

  try {
    auto storage = project->GetStorageService();
    if (!storage) {
      if (errorOut) {
        *errorOut = "Storage service is unavailable for DB snapshot.";
      }
      return false;
    }

    auto& db_ctrl = storage->GetDBController();
    auto  guard   = db_ctrl.GetConnectionGuard();

    std::error_code ec;
    std::filesystem::remove(snapshotPath, ec);

    // Use POSIX separators inside SQL strings on Windows to avoid backslash escaping surprises.
    const std::string path_utf8 = conv::ToBytes(snapshotPath.generic_wstring());
    const std::string escaped   = EscapeSqlStringLiteral(path_utf8);
    std::string       source_catalog;
    if (!QueryCurrentCatalog(guard.conn_, &source_catalog, errorOut)) {
      std::filesystem::remove(snapshotPath, ec);
      return false;
    }
    const std::string source_ident = "\"" + EscapeSqlIdentifier(source_catalog) + "\"";

    if (!RunDuckDbQuery(guard.conn_, "CHECKPOINT;", "checkpoint", errorOut) ||
        !RunDuckDbQuery(guard.conn_, "ATTACH '" + escaped + "' AS pack_snapshot;",
                        "attach snapshot", errorOut) ||
        !RunDuckDbQuery(guard.conn_,
                        "COPY FROM DATABASE " + source_ident + " TO pack_snapshot;",
                        "copy database", errorOut) ||
        !RunDuckDbQuery(guard.conn_, "CHECKPOINT pack_snapshot;", "checkpoint snapshot",
                        errorOut) ||
        !RunDuckDbQuery(guard.conn_, "DETACH pack_snapshot;", "detach snapshot", errorOut)) {
      std::filesystem::remove(snapshotPath, ec);
      return false;
    }

    ec.clear();
    if (!std::filesystem::is_regular_file(snapshotPath, ec) || ec) {
      if (errorOut) {
        *errorOut = QString("DuckDB snapshot file was not created: %1")
                        .arg(PathToQString(snapshotPath));
      }
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    if (errorOut) {
      *errorOut = QString("DuckDB snapshot failed: %1").arg(QString::fromUtf8(e.what()));
    }
    return false;
  } catch (...) {
    if (errorOut) {
      *errorOut = "DuckDB snapshot failed with an unknown error.";
    }
    return false;
  }
}

auto WritePackedProject(const std::filesystem::path& packedPath,
                        const std::filesystem::path& metaPath,
                        const std::filesystem::path& dbPath, QString* errorOut) -> bool {
  std::string meta_bytes;
  if (!ReadFileBytes(metaPath, &meta_bytes)) {
    if (errorOut) {
      *errorOut = QString("Failed to read project metadata: %1").arg(PathToQString(metaPath));
    }
    return false;
  }

  std::string db_bytes;
  if (!ReadFileBytes(dbPath, &db_bytes)) {
    if (errorOut) {
      *errorOut = QString("Failed to read project database: %1").arg(PathToQString(dbPath));
    }
    return false;
  }

  if (!EnsureDirectoryExists(packedPath.parent_path())) {
    if (errorOut) {
      *errorOut = "Packed project destination folder is not writable.";
    }
    return false;
  }

  const auto temp_path = packedPath.wstring() + L".tmp";
  std::ofstream out(std::filesystem::path(temp_path), std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    if (errorOut) {
      *errorOut = QString("Failed to open packed project file for writing: %1")
                      .arg(PathToQString(packedPath));
    }
    return false;
  }

  out.write(kPackedProjectMagic.data(),
            static_cast<std::streamsize>(kPackedProjectMagic.size()));
  if (!WriteU32Le(out, kPackedProjectVersion) ||
      !WriteU64Le(out, static_cast<uint64_t>(meta_bytes.size())) ||
      !WriteU64Le(out, static_cast<uint64_t>(db_bytes.size()))) {
    if (errorOut) {
      *errorOut = "Failed to write packed project header.";
    }
    return false;
  }

  if (!meta_bytes.empty()) {
    out.write(meta_bytes.data(), static_cast<std::streamsize>(meta_bytes.size()));
  }
  if (!db_bytes.empty()) {
    out.write(db_bytes.data(), static_cast<std::streamsize>(db_bytes.size()));
  }
  out.flush();
  if (!out.good()) {
    if (errorOut) {
      *errorOut = "Failed to write packed project payload.";
    }
    return false;
  }
  out.close();

  std::error_code ec;
  if (std::filesystem::exists(packedPath, ec) && !ec) {
    std::filesystem::remove(packedPath, ec);
    if (ec) {
      if (errorOut) {
        *errorOut =
            QString("Failed to replace existing packed project file: %1").arg(PathToQString(packedPath));
      }
      std::filesystem::remove(std::filesystem::path(temp_path), ec);
      return false;
    }
  }
  ec.clear();
  std::filesystem::rename(std::filesystem::path(temp_path), packedPath, ec);
  if (ec) {
    if (errorOut) {
      *errorOut =
          QString("Failed to finalize packed project file: %1").arg(PathToQString(packedPath));
    }
    std::filesystem::remove(std::filesystem::path(temp_path), ec);
    return false;
  }

  return true;
}

auto ReadPackedProject(const std::filesystem::path& packedPath, std::string* metaBytes,
                       std::string* dbBytes, QString* errorOut) -> bool {
  std::ifstream in(packedPath, std::ios::binary);
  if (!in.is_open()) {
    if (errorOut) {
      *errorOut =
          QString("Failed to open packed project: %1").arg(PathToQString(packedPath));
    }
    return false;
  }

  std::array<char, kPackedProjectMagic.size()> magic{};
  if (!ReadExact(in, magic.data(), static_cast<std::streamsize>(magic.size())) ||
      !std::equal(magic.begin(), magic.end(), kPackedProjectMagic.begin())) {
    if (errorOut) {
      *errorOut = "Packed project signature is invalid.";
    }
    return false;
  }

  uint32_t version = 0;
  if (!ReadU32Le(in, &version) || version != kPackedProjectVersion) {
    if (errorOut) {
      *errorOut = "Packed project version is not supported.";
    }
    return false;
  }

  uint64_t meta_size = 0;
  uint64_t db_size   = 0;
  if (!ReadU64Le(in, &meta_size) || !ReadU64Le(in, &db_size)) {
    if (errorOut) {
      *errorOut = "Packed project header is corrupted.";
    }
    return false;
  }
  if (meta_size > kMaxPackedComponentBytes || db_size > kMaxPackedComponentBytes) {
    if (errorOut) {
      *errorOut = "Packed project is too large.";
    }
    return false;
  }
  if (meta_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      db_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    if (errorOut) {
      *errorOut = "Packed project is too large for this build.";
    }
    return false;
  }

  metaBytes->assign(static_cast<size_t>(meta_size), '\0');
  if (meta_size > 0 &&
      !ReadExact(in, metaBytes->data(), static_cast<std::streamsize>(meta_size))) {
    if (errorOut) {
      *errorOut = "Failed to read packed metadata payload.";
    }
    return false;
  }

  dbBytes->assign(static_cast<size_t>(db_size), '\0');
  if (db_size > 0 && !ReadExact(in, dbBytes->data(), static_cast<std::streamsize>(db_size))) {
    if (errorOut) {
      *errorOut = "Failed to read packed database payload.";
    }
    return false;
  }

  return true;
}

auto UnpackProjectToWorkspace(const std::filesystem::path& packedPath,
                              const std::filesystem::path& workspaceDir,
                              const QString& projectName,
                              std::filesystem::path* dbPathOut,
                              std::filesystem::path* metaPathOut,
                              QString*               errorOut) -> bool {
  std::string meta_bytes;
  std::string db_bytes;
  if (!ReadPackedProject(packedPath, &meta_bytes, &db_bytes, errorOut)) {
    return false;
  }

  const auto runtime_pair = BuildRuntimeProjectPair(workspaceDir, projectName);
  const auto db_path      = runtime_pair.first;
  const auto meta_path    = runtime_pair.second;

  nlohmann::json metadata;
  try {
    metadata = nlohmann::json::parse(meta_bytes);
  } catch (...) {
    if (errorOut) {
      *errorOut = "Packed project metadata JSON is invalid.";
    }
    return false;
  }

  metadata["db_path"]   = conv::ToBytes(db_path.wstring());
  metadata["meta_path"] = conv::ToBytes(meta_path.wstring());

  std::error_code ec;
  std::filesystem::create_directories(db_path.parent_path(), ec);
  if (ec) {
    if (errorOut) {
      *errorOut = "Failed to prepare temp files for packed project.";
    }
    return false;
  }

  if (!WriteFileBytes(db_path, db_bytes)) {
    if (errorOut) {
      *errorOut = QString("Failed to materialize project database: %1")
                      .arg(PathToQString(db_path));
    }
    return false;
  }

  const std::string meta_text = metadata.dump(4);
  if (!WriteFileBytes(meta_path, meta_text)) {
    if (errorOut) {
      *errorOut = QString("Failed to materialize project metadata: %1")
                      .arg(PathToQString(meta_path));
    }
    return false;
  }

  *dbPathOut   = db_path;
  *metaPathOut = meta_path;
  return true;
}

}  // namespace

AlbumBackend::AlbumBackend(QObject* parent) : QObject(parent), rule_model_(this) {
  const QString pictures = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
  default_export_folder_ = pictures.isEmpty() ? QDir::currentPath() : pictures;

  initializeEditorLuts();
  setServiceState(
      false,
      "Select a project: load a .puerhproj package or metadata JSON, or create a new packed project.");
  task_status_ = "Open or create a project to begin.";
}

AlbumBackend::~AlbumBackend() {
  try {
    releaseVisibleThumbnailPins();
    finalizeEditorSession(true);
    if (current_import_job_) {
      current_import_job_->canceled_.store(true);
    }
    if (pipeline_service_) {
      pipeline_service_->Sync();
    }
    if (persistCurrentProjectState()) {
      QString ignored_error;
      (void)packageCurrentProjectFiles(&ignored_error);
    }
    CleanupWorkspaceDirectory(project_workspace_dir_);
  } catch (...) {
  }
}

auto AlbumBackend::fieldOptions() const -> QVariantList {
  return rule_model_.fieldOptions();
}

auto AlbumBackend::filterInfo() const -> QString {
  return formatFilterInfo(shownCount(), totalCount());
}

int AlbumBackend::totalCount() const {
  int count = 0;
  for (const auto& image : all_images_) {
    if (isImageInCurrentFolder(image)) {
      ++count;
    }
  }
  return count;
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
    const auto idsOpt   = filter_service_->ApplyFilterOn(filterId, current_folder_id_);
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

bool AlbumBackend::loadProject(const QString& metaFileUrlOrPath) {
  if (project_loading_) {
    setServiceMessageForCurrentProject("A project load is already in progress.");
    return false;
  }

  const auto project_path_opt = InputToPath(metaFileUrlOrPath);
  if (!project_path_opt.has_value()) {
    setServiceMessageForCurrentProject("Select a valid project file.");
    return false;
  }

  const auto project_path = project_path_opt.value();
  std::error_code ec;
  if (!std::filesystem::is_regular_file(project_path, ec) || ec) {
    setServiceMessageForCurrentProject("Project file was not found.");
    return false;
  }

  if (IsPackedProjectPath(project_path) || IsPackedProjectFile(project_path)) {
    const QString project_name = QFileInfo(PathToQString(project_path)).completeBaseName();
    std::filesystem::path workspace_dir;
    QString               workspace_error;
    if (!CreateProjectWorkspace(project_name, &workspace_dir, &workspace_error)) {
      setServiceMessageForCurrentProject(
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
      setServiceMessageForCurrentProject(
          unpack_error.isEmpty() ? "Failed to unpack project package." : unpack_error);
      return false;
    }

    return initializeServices(unpacked_db_path, unpacked_meta_path, ProjectOpenMode::kLoadExisting,
                              project_path, workspace_dir);
  }

  if (!IsMetadataJsonPath(project_path)) {
    setServiceMessageForCurrentProject(
        "Unsupported project format. Choose a .json or .puerhproj file.");
    return false;
  }

  const auto db_hint_path =
      project_path.parent_path() / (project_path.stem().wstring() + L".db");
  return initializeServices(db_hint_path, project_path, ProjectOpenMode::kLoadExisting,
                            BuildBundlePathFromMetaPath(project_path), {});
}

bool AlbumBackend::createProjectInFolder(const QString& folderUrlOrPath) {
  return createProjectInFolderNamed(folderUrlOrPath, "album_editor_project");
}

bool AlbumBackend::createProjectInFolderNamed(const QString& folderUrlOrPath,
                                              const QString& projectName) {
  if (project_loading_) {
    setServiceMessageForCurrentProject("A project load is already in progress.");
    return false;
  }

  const auto folder_path_opt = InputToPath(folderUrlOrPath);
  if (!folder_path_opt.has_value()) {
    setServiceMessageForCurrentProject("Select a valid folder for the new project.");
    return false;
  }

  QString build_error;
  const auto packed_path_opt =
      BuildUniquePackedProjectPath(folder_path_opt.value(), projectName, &build_error);
  if (!packed_path_opt.has_value()) {
    setServiceMessageForCurrentProject(
        build_error.isEmpty() ? "Failed to prepare project package path in selected folder."
                              : build_error);
    return false;
  }

  std::filesystem::path workspace_dir;
  QString               workspace_error;
  if (!CreateProjectWorkspace(projectName, &workspace_dir, &workspace_error)) {
    setServiceMessageForCurrentProject(workspace_error.isEmpty()
                                           ? "Failed to prepare project temp workspace."
                                           : workspace_error);
    return false;
  }

  const auto runtime_pair = BuildRuntimeProjectPair(workspace_dir, projectName);
  const bool started =
      initializeServices(runtime_pair.first, runtime_pair.second, ProjectOpenMode::kCreateNew,
                         packed_path_opt.value(), workspace_dir);
  if (!started) {
    CleanupWorkspaceDirectory(workspace_dir);
  }
  return started;
}

bool AlbumBackend::saveProject() {
  if (project_loading_) {
    setServiceMessageForCurrentProject("Please wait until project loading finishes.");
    return false;
  }

  if (!project_ || meta_path_.empty()) {
    setServiceState(false, "No project is loaded yet.");
    setTaskState("No project to save.", 0, false);
    return false;
  }

  if (editor_active_) {
    finalizeEditorSession(true);
  }

  if (!persistCurrentProjectState()) {
    setServiceMessageForCurrentProject("Project save failed.");
    setTaskState("Project save failed.", 0, false);
    return false;
  }

  QString package_error;
  if (!packageCurrentProjectFiles(&package_error)) {
    setServiceMessageForCurrentProject(package_error.isEmpty() ? "Project saved, but packing failed."
                                                               : package_error);
    setTaskState("Project packing failed.", 0, false);
    return false;
  }

  setServiceMessageForCurrentProject(project_package_path_.empty()
                                         ? QString("Project saved to %1")
                                               .arg(PathToQString(meta_path_))
                                         : QString("Project saved and packed to %1")
                                               .arg(PathToQString(project_package_path_)));
  setTaskState(project_package_path_.empty() ? "Project saved." : "Project saved and packed.", 100,
               false);
  scheduleIdleTaskStateReset(1200);
  return true;
}

auto AlbumBackend::compareOptionsForField(int fieldValue) const -> QVariantList {
  return FilterRuleModel::compareOptionsForField(static_cast<FilterField>(fieldValue));
}

auto AlbumBackend::placeholderForField(int fieldValue) const -> QString {
  return FilterRuleModel::placeholderForField(static_cast<FilterField>(fieldValue));
}

void AlbumBackend::selectFolder(uint folderId) {
  if (project_loading_ || !project_) {
    return;
  }

  applyFolderSelection(static_cast<sl_element_id_t>(folderId), true);
  reapplyCurrentFilters();
}

void AlbumBackend::createFolder(const QString& folderName) {
  if (project_loading_) {
    setTaskState("Project is loading. Please wait.", 0, false);
    return;
  }
  if (!project_) {
    setTaskState("No project is loaded.", 0, false);
    return;
  }
  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    setTaskState("Cannot create folder while import is running.", 0, false);
    return;
  }
  if (export_inflight_) {
    setTaskState("Cannot create folder while export is running.", 0, false);
    return;
  }

  const QString trimmed = folderName.trimmed();
  if (trimmed.isEmpty()) {
    setTaskState("Folder name cannot be empty.", 0, false);
    return;
  }
  if (trimmed.contains('/') || trimmed.contains('\\')) {
    setTaskState("Folder name cannot contain '/' or '\\'.", 0, false);
    return;
  }

  const auto parent_path = currentFolderFsPath();
  try {
    const auto create_result = project_->GetSleeveService()->Write<std::shared_ptr<SleeveElement>>(
        [parent_path, trimmed](FileSystem& fs) {
          return fs.Create(parent_path, trimmed.toStdWString(), ElementType::FOLDER);
        });
    const auto created = create_result.first;
    if (!create_result.second.success_ || !created || created->type_ != ElementType::FOLDER) {
      throw std::runtime_error("Failed to create folder.");
    }

    ExistingFolderEntry folder_entry;
    folder_entry.folder_id_   = created->element_id_;
    folder_entry.parent_id_   = current_folder_id_;
    folder_entry.folder_name_ = created->element_name_;
    folder_entry.folder_path_ = parent_path / created->element_name_;
    folder_entry.depth_       = 0;
    for (const auto& existing : folder_entries_) {
      if (existing.folder_id_ == current_folder_id_) {
        folder_entry.depth_ = existing.depth_ + 1;
        break;
      }
    }

    folder_entries_.push_back(folder_entry);
    folder_parent_by_id_[folder_entry.folder_id_] = folder_entry.parent_id_;
    folder_path_by_id_[folder_entry.folder_id_]   = folder_entry.folder_path_;
    rebuildFolderView();

    if (!meta_path_.empty()) {
      project_->SaveProject(meta_path_);
    }

    setServiceMessageForCurrentProject(
        QString("Created folder %1").arg(WStringToQString(folder_entry.folder_name_)));
    setTaskState(service_message_, 100, false);
    scheduleIdleTaskStateReset(1200);
  } catch (const std::exception& e) {
    const QString err = QString("Failed to create folder: %1").arg(QString::fromUtf8(e.what()));
    setServiceMessageForCurrentProject(err);
    setTaskState(err, 0, false);
  }
}

void AlbumBackend::deleteFolder(uint folderId) {
  if (project_loading_) {
    setTaskState("Project is loading. Please wait.", 0, false);
    return;
  }
  if (!project_) {
    setTaskState("No project is loaded.", 0, false);
    return;
  }
  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    setTaskState("Cannot delete folder while import is running.", 0, false);
    return;
  }
  if (export_inflight_) {
    setTaskState("Cannot delete folder while export is running.", 0, false);
    return;
  }

  const auto folder_id = static_cast<sl_element_id_t>(folderId);
  if (folder_id == 0) {
    setTaskState("Root folder cannot be deleted.", 0, false);
    return;
  }

  const auto path_it = folder_path_by_id_.find(folder_id);
  if (path_it == folder_path_by_id_.end()) {
    setTaskState("Folder no longer exists.", 0, false);
    return;
  }
  const auto parent_it_before = folder_parent_by_id_.find(folder_id);
  const sl_element_id_t fallback_folder =
      parent_it_before != folder_parent_by_id_.end() ? parent_it_before->second
                                                     : static_cast<sl_element_id_t>(0);

  std::unordered_set<sl_element_id_t> deleted_folder_ids;
  deleted_folder_ids.insert(folder_id);
  bool expanded = true;
  while (expanded) {
    expanded = false;
    for (const auto& [candidate_id, parent_id] : folder_parent_by_id_) {
      if (!deleted_folder_ids.contains(parent_id) || deleted_folder_ids.contains(candidate_id)) {
        continue;
      }
      deleted_folder_ids.insert(candidate_id);
      expanded = true;
    }
  }

  try {
    const auto remove_result = project_->GetSleeveService()->Write<void>(
        [target_path = path_it->second](FileSystem& fs) { fs.Delete(target_path); });
    if (!remove_result.success_) {
      throw std::runtime_error(remove_result.message_);
    }

    if (!meta_path_.empty()) {
      project_->SaveProject(meta_path_);
    }
  } catch (const std::exception& e) {
    const QString err = QString("Failed to delete folder: %1").arg(QString::fromUtf8(e.what()));
    setServiceMessageForCurrentProject(err);
    setTaskState(err, 0, false);
    return;
  }

  if (thumbnail_service_) {
    for (const auto& image : all_images_) {
      if (!deleted_folder_ids.contains(image.parent_folder_id)) {
        continue;
      }
      try {
        thumbnail_service_->InvalidateThumbnail(image.element_id);
        thumbnail_service_->ReleaseThumbnail(image.element_id);
      } catch (...) {
      }
    }
  }

  const auto snapshot = collectProjectSnapshot(project_);

  releaseVisibleThumbnailPins();
  all_images_.clear();
  index_by_element_id_.clear();
  visible_thumbnails_.clear();
  emit thumbnailsChanged();

  folder_entries_      = snapshot.folder_entries_;
  folder_parent_by_id_ = snapshot.folder_parent_by_id_;
  folder_path_by_id_   = snapshot.folder_path_by_id_;
  rebuildFolderView();

  for (const auto& entry : snapshot.album_entries_) {
    addOrUpdateAlbumItem(entry.element_id_, entry.image_id_, entry.file_name_,
                         entry.parent_folder_id_);
  }

  if (folder_path_by_id_.contains(current_folder_id_)) {
    applyFolderSelection(current_folder_id_, true);
  } else {
    applyFolderSelection(fallback_folder, true);
  }

  active_filter_ids_.reset();
  reapplyCurrentFilters();

  setServiceMessageForCurrentProject("Folder deleted.");
  setTaskState("Folder deleted.", 100, false);
  scheduleIdleTaskStateReset(1200);
}

void AlbumBackend::startImport(const QStringList& fileUrlsOrPaths) {
  if (project_loading_) {
    setTaskState("Project is loading. Please wait.", 0, false);
    return;
  }
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

  import_target_folder_id_   = current_folder_id_;
  import_target_folder_path_ = currentFolderFsPath();

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
    current_import_job_ =
        import_service_->ImportToFolder(paths, import_target_folder_path_, options, job);
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
  if (project_loading_) {
    setExportFailureState("Project is loading. Please wait.");
    return;
  }

  if (!export_service_ || !project_) {
    setExportFailureState("Export service is unavailable.");
    return;
  }
  if (export_inflight_) {
    setExportFailureState("Export already running.");
    return;
  }

  resetExportProgressState("Preparing export queue...");

  const auto outDirOpt = InputToPath(outputDirUrlOrPath);
  if (!outDirOpt.has_value()) {
    setExportFailureState("No export folder selected.");
    return;
  }

  std::error_code ec;
  if (!std::filesystem::exists(outDirOpt.value(), ec)) {
    std::filesystem::create_directories(outDirOpt.value(), ec);
  }
  if (ec || !std::filesystem::is_directory(outDirOpt.value(), ec) || ec) {
    setExportFailureState("Export folder is invalid.");
    return;
  }

  const auto targets = collectExportTargets(targetEntries);

  if (targets.empty()) {
    setExportFailureState("No images to export.");
    return;
  }

  const ImageFormatType format        = FormatFromName(formatName);
  const int             clamped_max   = std::clamp(maxLengthSide, 256, 16384);
  const int             clamped_q     = std::clamp(quality, 1, 100);
  const auto            bit_depth     = BitDepthFromInt(bitDepth);
  const int             clamped_png   = std::clamp(pngCompressionLevel, 0, 9);
  const auto            tiff_compress = TiffCompressFromName(tiffCompression);

  export_service_->ClearAllExportTasks();
  const auto queue_result =
      buildExportQueue(targets, outDirOpt.value(), format, resizeEnabled, clamped_max, clamped_q,
                       bit_depth, clamped_png, tiff_compress);

  if (queue_result.queued_count_ == 0) {
    export_status_ = "No export tasks were queued.";
    if (!queue_result.first_error_.isEmpty()) {
      export_error_summary_ = queue_result.first_error_;
    }
    emit exportStateChanged();
    setTaskState("No valid export tasks could be created.", 0, false);
    return;
  }

  export_inflight_ = true;
  export_total_    = queue_result.queued_count_;
  export_skipped_  = queue_result.skipped_count_;
  if (queue_result.skipped_count_ > 0) {
    export_status_ = QString("Exporting %1 image(s). Skipped %2 invalid item(s).")
                         .arg(queue_result.queued_count_)
                         .arg(queue_result.skipped_count_);
  } else {
    export_status_ = QString("Exporting %1 image(s)...").arg(queue_result.queued_count_);
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
      [self, skipped = queue_result.skipped_count_](std::shared_ptr<std::vector<ExportResult>> results) {
        if (!self) {
          return;
        }

        QMetaObject::invokeMethod(
            self,
            [self, results, skipped]() {
              if (!self) {
                return;
              }
              self->finishExport(results, skipped);
            },
            Qt::QueuedConnection);
      });
}

void AlbumBackend::resetExportState() {
  if (export_inflight_) {
    return;
  }
  resetExportProgressState("Ready to export.");
}

void AlbumBackend::openEditor(uint elementId, uint imageId) {
  if (project_loading_) {
    editor_status_ = "Project is loading. Please wait.";
    emit editorStateChanged();
    return;
  }
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
      if (isThumbnailPinned(nextElementId)) {
        requestThumbnail(nextElementId, nextImageId);
      } else {
        updateThumbnailDataUrl(nextElementId, QString());
      }
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

bool AlbumBackend::initializeServices(const std::filesystem::path& dbPath,
                                      const std::filesystem::path& metaPath,
                                      ProjectOpenMode              openMode,
                                      const std::filesystem::path& packagePath,
                                      const std::filesystem::path& workspaceDir) {
  if (project_loading_) {
    setServiceMessageForCurrentProject("A project load is already in progress.");
    return false;
  }

  if (current_import_job_ && !current_import_job_->IsCancelationAcked()) {
    setServiceMessageForCurrentProject("Cannot switch project while an import is running.");
    return false;
  }
  if (export_inflight_) {
    setServiceMessageForCurrentProject("Cannot switch project while export is running.");
    return false;
  }

  if (editor_active_) {
    finalizeEditorSession(true);
  }

  setServiceMessageForCurrentProject((openMode == ProjectOpenMode::kCreateNew)
                                         ? "Creating project..."
                                         : "Loading project...");
  setProjectLoadingState(true, service_message_);
  setTaskState("Opening project...", 0, false);

  const auto request_id = ++project_load_request_id_;

  auto old_project  = project_;
  auto old_pipeline = pipeline_service_;
  auto old_meta     = meta_path_;
  auto old_package  = project_package_path_;
  auto old_workspace = project_workspace_dir_;

  QPointer<AlbumBackend> self(this);
  std::thread([self, request_id, old_project = std::move(old_project),
               old_pipeline = std::move(old_pipeline), old_meta = std::move(old_meta),
               old_package = std::move(old_package), old_workspace = std::move(old_workspace),
               dbPath, metaPath, packagePath, workspaceDir, openMode]() mutable {
    struct LoadResult {
      bool                                 success_ = false;
      QString                              error_{};
      std::shared_ptr<ProjectService>      project_{};
      std::shared_ptr<PipelineMgmtService> pipeline_{};
      std::shared_ptr<EditHistoryMgmtService> history_{};
      std::shared_ptr<ThumbnailService>    thumbnail_{};
      std::unique_ptr<SleeveFilterService> filter_{};
      std::unique_ptr<ImportServiceImpl>   import_{};
      std::shared_ptr<ExportService>       export_{};
      std::filesystem::path                db_path_{};
      std::filesystem::path                meta_path_{};
      std::filesystem::path                package_path_{};
      std::filesystem::path                workspace_dir_{};
      std::filesystem::path                workspace_to_cleanup_{};
      std::vector<ExistingAlbumEntry>      album_entries_{};
      std::vector<ExistingFolderEntry>     folder_entries_{};
      std::unordered_map<sl_element_id_t, sl_element_id_t> folder_parent_by_id_{};
      std::unordered_map<sl_element_id_t, std::filesystem::path> folder_path_by_id_{};
    };

    auto result = std::make_shared<LoadResult>();

    try {
      if (old_pipeline) {
        old_pipeline->Sync();
      }
      if (old_project && !old_meta.empty()) {
        old_project->GetSleeveService()->Sync();
        old_project->GetImagePoolService()->SyncWithStorage();
        old_project->SaveProject(old_meta);
        if (!old_package.empty()) {
          QString package_error;
          std::filesystem::path snapshot_path;
          if (!BuildTempDbSnapshotPath(&snapshot_path, &package_error) ||
              !CreateLiveDbSnapshot(old_project, snapshot_path, &package_error) ||
              !WritePackedProject(old_package, old_meta, snapshot_path, &package_error)) {
            std::error_code ec;
            if (!snapshot_path.empty()) {
              std::filesystem::remove(snapshot_path, ec);
            }
            const QByteArray err = package_error.toUtf8();
            throw std::runtime_error(err.isEmpty() ? "Failed to pack previous project."
                                                   : err.constData());
          }
          std::error_code ec;
          std::filesystem::remove(snapshot_path, ec);
        }
      }
      result->workspace_to_cleanup_ = old_workspace;

      result->project_ = std::make_shared<ProjectService>(dbPath, metaPath, openMode);
      result->pipeline_ =
          std::make_shared<PipelineMgmtService>(result->project_->GetStorageService());
      result->history_ =
          std::make_shared<EditHistoryMgmtService>(result->project_->GetStorageService());
      result->thumbnail_ = std::make_shared<ThumbnailService>(
          result->project_->GetSleeveService(), result->project_->GetImagePoolService(),
          result->pipeline_);
      result->filter_ =
          std::make_unique<SleeveFilterService>(result->project_->GetStorageService());
      result->import_ = std::make_unique<ImportServiceImpl>(result->project_->GetSleeveService(),
                                                            result->project_->GetImagePoolService());
      result->export_ = std::make_shared<ExportService>(result->project_->GetSleeveService(),
                                                        result->project_->GetImagePoolService(),
                                                        result->pipeline_);

      if (openMode == ProjectOpenMode::kCreateNew) {
        result->project_->GetSleeveService()->Sync();
        result->project_->GetImagePoolService()->SyncWithStorage();
        result->project_->SaveProject(metaPath);
      }

      result->db_path_   = result->project_->GetDBPath();
      result->meta_path_ = result->project_->GetMetaPath();
      if (result->meta_path_.empty()) {
        result->meta_path_ = metaPath;
      }
      result->package_path_ = packagePath;
      result->workspace_dir_ = workspaceDir;

      if (self) {
        auto snapshot              = self->collectProjectSnapshot(result->project_);
        result->album_entries_     = std::move(snapshot.album_entries_);
        result->folder_entries_    = std::move(snapshot.folder_entries_);
        result->folder_parent_by_id_ = std::move(snapshot.folder_parent_by_id_);
        result->folder_path_by_id_ = std::move(snapshot.folder_path_by_id_);
      }
      result->success_ = true;
    } catch (const std::exception& e) {
      result->success_ = false;
      result->error_   = QString::fromUtf8(e.what());
    } catch (...) {
      result->success_ = false;
      result->error_   = "Unknown project load error.";
    }

    if (!result->success_ && !workspaceDir.empty()) {
      CleanupWorkspaceDirectory(workspaceDir);
    }

    if (!self) {
      return;
    }

    QMetaObject::invokeMethod(
        self,
        [self, request_id, result]() mutable {
          if (!self || request_id != self->project_load_request_id_) {
            return;
          }

          if (!result->success_) {
            self->setProjectLoadingState(false, QString());
            self->setServiceMessageForCurrentProject(
                self->project_
                    ? QString("Requested project failed to open: %1").arg(result->error_)
                    : QString("Project open failed: %1").arg(result->error_));
            self->setTaskState("Project open failed.", 0, false);
            return;
          }

          self->project_           = std::move(result->project_);
          self->pipeline_service_  = std::move(result->pipeline_);
          self->history_service_   = std::move(result->history_);
          self->thumbnail_service_ = std::move(result->thumbnail_);
          self->filter_service_    = std::move(result->filter_);
          self->import_service_    = std::move(result->import_);
          self->export_service_    = std::move(result->export_);
          self->db_path_           = std::move(result->db_path_);
          self->meta_path_         = std::move(result->meta_path_);
          self->project_package_path_ = std::move(result->package_path_);
          self->project_workspace_dir_ = std::move(result->workspace_dir_);

          self->clearProjectData();
          self->resetExportState();
          self->pending_project_entries_      = std::move(result->album_entries_);
          self->pending_folder_entries_       = std::move(result->folder_entries_);
          self->pending_folder_parent_by_id_  = std::move(result->folder_parent_by_id_);
          self->pending_folder_path_by_id_    = std::move(result->folder_path_by_id_);
          self->pending_project_entry_index_  = 0;
          self->applyLoadedProjectEntriesBatch();

          if (!result->workspace_to_cleanup_.empty() &&
              result->workspace_to_cleanup_ != self->project_workspace_dir_) {
            CleanupWorkspaceDirectory(result->workspace_to_cleanup_);
          }
        },
        Qt::QueuedConnection);
  }).detach();

  return true;
}

bool AlbumBackend::persistCurrentProjectState() {
  try {
    if (pipeline_service_) {
      pipeline_service_->Sync();
    }
    if (project_) {
      project_->GetSleeveService()->Sync();
      project_->GetImagePoolService()->SyncWithStorage();
      if (!meta_path_.empty()) {
        project_->SaveProject(meta_path_);
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool AlbumBackend::packageCurrentProjectFiles(QString* errorOut) const {
  if (!project_ || db_path_.empty() || meta_path_.empty() || project_package_path_.empty()) {
    return true;
  }

  std::filesystem::path snapshot_path;
  if (!BuildTempDbSnapshotPath(&snapshot_path, errorOut)) {
    return false;
  }

  const bool snapshot_ok = CreateLiveDbSnapshot(project_, snapshot_path, errorOut);
  if (!snapshot_ok) {
    std::error_code ec;
    std::filesystem::remove(snapshot_path, ec);
    return false;
  }

  const bool packed_ok =
      WritePackedProject(project_package_path_, meta_path_, snapshot_path, errorOut);
  std::error_code ec;
  std::filesystem::remove(snapshot_path, ec);
  return packed_ok;
}

auto AlbumBackend::collectProjectSnapshot(const std::shared_ptr<ProjectService>& project) const
    -> ProjectSnapshot {
  ProjectSnapshot snapshot;
  if (!project) {
    return snapshot;
  }

  try {
    auto& element_ctrl = project->GetStorageService()->GetElementController();

    struct FolderVisit {
      sl_element_id_t      folder_id_ = 0;
      sl_element_id_t      parent_id_ = 0;
      std::filesystem::path folder_path_{};
      int                  depth_     = 0;
    };

    std::vector<FolderVisit>        stack{{0, 0, std::filesystem::path(L"/"), 0}};
    std::unordered_set<sl_element_id_t> visited;
    visited.reserve(4096);

    while (!stack.empty()) {
      const auto visit = stack.back();
      stack.pop_back();

      if (!visited.insert(visit.folder_id_).second) {
        continue;
      }

      std::shared_ptr<SleeveElement> folder_element;
      try {
        folder_element = element_ctrl.GetElementById(visit.folder_id_);
      } catch (...) {
        continue;
      }

      if (!folder_element || folder_element->sync_flag_ == SyncFlag::DELETED ||
          folder_element->type_ != ElementType::FOLDER) {
        continue;
      }

      ExistingFolderEntry folder_entry;
      folder_entry.folder_id_   = visit.folder_id_;
      folder_entry.parent_id_   = visit.parent_id_;
      folder_entry.folder_name_ = visit.folder_id_ == 0 ? L"" : folder_element->element_name_;
      folder_entry.folder_path_ = visit.folder_path_;
      folder_entry.depth_       = visit.depth_;
      snapshot.folder_entries_.push_back(folder_entry);
      snapshot.folder_parent_by_id_[folder_entry.folder_id_] = folder_entry.parent_id_;
      snapshot.folder_path_by_id_[folder_entry.folder_id_]   = folder_entry.folder_path_;

      std::vector<sl_element_id_t> children;
      try {
        children = element_ctrl.GetFolderContent(visit.folder_id_);
      } catch (...) {
        children.clear();
      }

      std::vector<std::shared_ptr<SleeveElement>> child_elements;
      child_elements.reserve(children.size());
      for (const auto child_id : children) {
        if (child_id == visit.folder_id_) {
          continue;
        }
        try {
          auto child = element_ctrl.GetElementById(child_id);
          if (!child || child->sync_flag_ == SyncFlag::DELETED) {
            continue;
          }
          child_elements.push_back(std::move(child));
        } catch (...) {
        }
      }

      std::sort(child_elements.begin(), child_elements.end(),
                [](const std::shared_ptr<SleeveElement>& lhs,
                   const std::shared_ptr<SleeveElement>& rhs) {
                  if (lhs->type_ != rhs->type_) {
                    return lhs->type_ == ElementType::FOLDER;
                  }
                  return lhs->element_name_ < rhs->element_name_;
                });

      for (auto it = child_elements.rbegin(); it != child_elements.rend(); ++it) {
        const auto& child = *it;
        if (child->type_ == ElementType::FOLDER) {
          stack.push_back(
              {child->element_id_, visit.folder_id_, visit.folder_path_ / child->element_name_,
               visit.depth_ + 1});
          continue;
        }

        const auto file = std::dynamic_pointer_cast<SleeveFile>(child);
        if (!file || file->image_id_ == 0) {
          continue;
        }
        snapshot.album_entries_.push_back(
            {file->element_id_, visit.folder_id_, file->image_id_, file->element_name_});
      }
    }
  } catch (...) {
  }

  std::sort(snapshot.album_entries_.begin(), snapshot.album_entries_.end(),
            [](const ExistingAlbumEntry& lhs, const ExistingAlbumEntry& rhs) {
              if (lhs.file_name_ != rhs.file_name_) {
                return lhs.file_name_ < rhs.file_name_;
              }
              return lhs.element_id_ < rhs.element_id_;
            });

  std::sort(snapshot.folder_entries_.begin(), snapshot.folder_entries_.end(),
            [](const ExistingFolderEntry& lhs, const ExistingFolderEntry& rhs) {
              if (lhs.folder_id_ == 0 || rhs.folder_id_ == 0) {
                return lhs.folder_id_ == 0;
              }
              if (lhs.folder_path_ != rhs.folder_path_) {
                return lhs.folder_path_.generic_wstring() < rhs.folder_path_.generic_wstring();
              }
              return lhs.folder_id_ < rhs.folder_id_;
            });

  return snapshot;
}

void AlbumBackend::applyLoadedProjectEntriesBatch() {
  if (!project_loading_) {
    return;
  }

  const size_t total = pending_project_entries_.size();
  if (total == 0 || pending_project_entry_index_ >= total) {
    pending_project_entries_.clear();
    pending_project_entry_index_ = 0;
    folder_entries_     = std::move(pending_folder_entries_);
    folder_parent_by_id_ = std::move(pending_folder_parent_by_id_);
    folder_path_by_id_   = std::move(pending_folder_path_by_id_);
    rebuildFolderView();
    applyFolderSelection(0, true);
    rebuildThumbnailView(std::nullopt);
    setTaskState("No background tasks", 0, false);

    setServiceState(true, project_package_path_.empty()
                              ? QString("Loaded project. DB: %1  Meta: %2")
                                    .arg(PathToQString(db_path_))
                                    .arg(PathToQString(meta_path_))
                              : QString("Loaded packed project: %1 (DB temp: %2)")
                                    .arg(PathToQString(project_package_path_))
                                    .arg(PathToQString(db_path_)));
    emit projectChanged();
    setProjectLoadingState(false, QString());
    return;
  }

  constexpr size_t kBatchSize = 24;
  const size_t     end_index  = std::min(total, pending_project_entry_index_ + kBatchSize);

  for (; pending_project_entry_index_ < end_index; ++pending_project_entry_index_) {
    const auto& entry = pending_project_entries_[pending_project_entry_index_];
    addOrUpdateAlbumItem(entry.element_id_, entry.image_id_, entry.file_name_,
                         entry.parent_folder_id_);
  }

  const int pct =
      total == 0 ? 0 : static_cast<int>((pending_project_entry_index_ * 100ULL) / total);
  setTaskState(
      QString("Loading album... %1/%2").arg(static_cast<int>(pending_project_entry_index_)).arg(
          static_cast<int>(total)),
      pct, false);
  setProjectLoadingState(
      true, QString("Loading album... %1/%2")
                .arg(static_cast<int>(pending_project_entry_index_))
                .arg(static_cast<int>(total)));

  QTimer::singleShot(0, this, [this]() { applyLoadedProjectEntriesBatch(); });
}

void AlbumBackend::setProjectLoadingState(bool loading, const QString& message) {
  const QString next_message = loading ? message : QString();
  if (project_loading_ == loading && project_loading_message_ == next_message) {
    return;
  }
  project_loading_         = loading;
  project_loading_message_ = next_message;
  emit projectLoadStateChanged();
}

void AlbumBackend::clearProjectData() {
  releaseVisibleThumbnailPins();

  all_images_.clear();
  index_by_element_id_.clear();
  visible_thumbnails_.clear();
  folder_entries_.clear();
  folder_parent_by_id_.clear();
  folder_path_by_id_.clear();
  folders_.clear();
  current_folder_id_        = 0;
  current_folder_path_text_ = "/";
  active_filter_ids_.reset();
  pending_project_entries_.clear();
  pending_folder_entries_.clear();
  pending_folder_parent_by_id_.clear();
  pending_folder_path_by_id_.clear();
  pending_project_entry_index_ = 0;
  import_target_folder_id_     = 0;
  import_target_folder_path_.clear();

  rule_model_.clearAndReset();
  last_join_op_ = FilterOp::AND;

  if (!sql_preview_.isEmpty()) {
    sql_preview_.clear();
    emit sqlPreviewChanged();
  }
  if (!validation_error_.isEmpty()) {
    validation_error_.clear();
    emit validationErrorChanged();
  }

  emit thumbnailsChanged();
  emit foldersChanged();
  emit folderSelectionChanged();
  emit countsChanged();
}

void AlbumBackend::setThumbnailVisible(uint elementId, uint imageId, bool visible) {
  const auto id       = static_cast<sl_element_id_t>(elementId);
  const auto image_id = static_cast<image_id_t>(imageId);
  if (id == 0 || image_id == 0) {
    return;
  }

  if (visible) {
    if (!thumbnail_service_) {
      return;
    }
    auto& ref = thumbnail_pin_ref_counts_[id];
    ref++;
    if (ref == 1) {
      requestThumbnail(id, image_id);
    }
    return;
  }

  const auto it = thumbnail_pin_ref_counts_.find(id);
  if (it == thumbnail_pin_ref_counts_.end()) {
    return;
  }

  if (it->second > 1) {
    it->second--;
    return;
  }

  thumbnail_pin_ref_counts_.erase(it);
  updateThumbnailDataUrl(id, QString());
  if (thumbnail_service_) {
    try {
      thumbnail_service_->ReleaseThumbnail(id);
    } catch (...) {
    }
  }
}

void AlbumBackend::rebuildFolderView() {
  std::sort(folder_entries_.begin(), folder_entries_.end(),
            [](const ExistingFolderEntry& lhs, const ExistingFolderEntry& rhs) {
              if (lhs.folder_id_ == 0 || rhs.folder_id_ == 0) {
                return lhs.folder_id_ == 0;
              }
              if (lhs.folder_path_ != rhs.folder_path_) {
                return lhs.folder_path_.generic_wstring() < rhs.folder_path_.generic_wstring();
              }
              return lhs.folder_id_ < rhs.folder_id_;
            });

  QVariantList next;
  next.reserve(static_cast<qsizetype>(folder_entries_.size()));

  for (const auto& folder : folder_entries_) {
    const QString name =
        folder.folder_id_ == 0 ? "Root" : WStringToQString(folder.folder_name_);
    next.push_back(QVariantMap{
        {"folderId", static_cast<uint>(folder.folder_id_)},
        {"name", name},
        {"depth", folder.depth_},
        {"path", FolderPathToDisplay(folder.folder_path_)},
        {"deletable", folder.folder_id_ != 0},
    });
  }

  folders_ = std::move(next);
  emit foldersChanged();
}

void AlbumBackend::applyFolderSelection(sl_element_id_t folderId, bool emitSignal) {
  sl_element_id_t next_folder_id = folderId;
  if (!folder_path_by_id_.contains(next_folder_id)) {
    next_folder_id = 0;
  }
  if (!folder_path_by_id_.contains(next_folder_id) && !folder_entries_.empty()) {
    next_folder_id = folder_entries_.front().folder_id_;
  }

  const bool    id_changed   = current_folder_id_ != next_folder_id;
  current_folder_id_         = next_folder_id;
  const auto    path_it      = folder_path_by_id_.find(current_folder_id_);
  const QString next_path_ui =
      path_it != folder_path_by_id_.end() ? FolderPathToDisplay(path_it->second) : QString("/");
  const bool path_changed = current_folder_path_text_ != next_path_ui;
  current_folder_path_text_ = next_path_ui;

  if (emitSignal || id_changed || path_changed) {
    emit folderSelectionChanged();
  }
}

auto AlbumBackend::currentFolderFsPath() const -> std::filesystem::path {
  const auto it = folder_path_by_id_.find(current_folder_id_);
  if (it == folder_path_by_id_.end()) {
    return std::filesystem::path(L"/");
  }
  return it->second;
}

void AlbumBackend::releaseVisibleThumbnailPins() {
  if (thumbnail_pin_ref_counts_.empty()) {
    return;
  }

  for (const auto& [id, _] : thumbnail_pin_ref_counts_) {
    const auto index_it = index_by_element_id_.find(id);
    if (index_it != index_by_element_id_.end()) {
      all_images_[index_it->second].thumb_data_url.clear();
    }
    if (thumbnail_service_) {
      try {
        thumbnail_service_->ReleaseThumbnail(id);
      } catch (...) {
      }
    }
  }
  thumbnail_pin_ref_counts_.clear();
}

void AlbumBackend::rebuildThumbnailView(
    const std::optional<std::unordered_set<sl_element_id_t>>& allowedElementIds) {
  releaseVisibleThumbnailPins();

  QVariantList next;
  next.reserve(static_cast<qsizetype>(all_images_.size()));

  int index = 0;
  for (const AlbumItem& image : all_images_) {
    if (!isImageInCurrentFolder(image)) {
      continue;
    }
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
    addOrUpdateAlbumItem(created.element_id_, created.image_id_, created.file_name_,
                         import_target_folder_id_);
  }
}

void AlbumBackend::addOrUpdateAlbumItem(sl_element_id_t elementId, image_id_t imageId,
                                        const file_name_t& fallbackName,
                                        sl_element_id_t parentFolderId) {
  AlbumItem* item = nullptr;

  if (const auto it = index_by_element_id_.find(elementId); it != index_by_element_id_.end()) {
    item = &all_images_[it->second];
  } else {
    AlbumItem next;
    next.element_id   = elementId;
    next.image_id     = imageId;
    next.file_name    = WStringToQString(fallbackName);
    next.extension    = ExtensionFromFileName(next.file_name);
    next.accent = AccentForIndex(all_images_.size());

    all_images_.push_back(std::move(next));
    index_by_element_id_[elementId] = all_images_.size() - 1;
    item = &all_images_.back();
  }

  if (!item) {
    return;
  }

  item->element_id = elementId;
  item->image_id   = imageId;
  item->parent_folder_id = parentFolderId;

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
        if (!guard || !guard->thumbnail_buffer_) {
          if (self && !self->isThumbnailPinned(elementId) && service) {
            try {
              service->ReleaseThumbnail(elementId);
            } catch (...) {
            }
          }
          return;
        }
        if (!self) {
          try {
            if (service) {
              service->ReleaseThumbnail(elementId);
            }
          } catch (...) {
          }
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
                [self, service, elementId, dataUrl]() {
                  if (!self) {
                    return;
                  }
                  const bool pinned = self->isThumbnailPinned(elementId);
                  if (pinned) {
                    self->updateThumbnailDataUrl(elementId, dataUrl);
                  } else {
                    self->updateThumbnailDataUrl(elementId, QString());
                  }
                  if (!pinned && service) {
                    try {
                      service->ReleaseThumbnail(elementId);
                    } catch (...) {
                    }
                  }
                },
                Qt::QueuedConnection);
          }
        }).detach();
      },
      true, dispatcher);
}

void AlbumBackend::updateThumbnailDataUrl(sl_element_id_t elementId, const QString& dataUrl) {
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

bool AlbumBackend::isThumbnailPinned(sl_element_id_t elementId) const {
  const auto it = thumbnail_pin_ref_counts_.find(elementId);
  return it != thumbnail_pin_ref_counts_.end() && it->second > 0;
}

void AlbumBackend::finishImport(const ImportResult& result) {
  const auto importJob = current_import_job_;
  current_import_job_.reset();

  if (!importJob || !importJob->import_log_) {
    setTaskState("Import finished but no log snapshot is available.", 0, false);
    return;
  }

  const auto snapshot = importJob->import_log_->Snapshot();

  bool state_saved = true;
  try {
    if (import_service_) {
      import_service_->SyncImports(snapshot, import_target_folder_path_);
    }
    if (project_) {
      project_->GetSleeveService()->Sync();
      project_->GetImagePoolService()->SyncWithStorage();
      project_->SaveProject(meta_path_);
    }
  } catch (...) {
    state_saved = false;
  }

  QString package_error;
  bool    package_saved = true;
  if (state_saved) {
    package_saved = packageCurrentProjectFiles(&package_error);
  }

  addImportedEntries(snapshot);
  reapplyCurrentFilters();

  import_target_folder_id_   = current_folder_id_;
  import_target_folder_path_ = currentFolderFsPath();

  QString task_text =
      QString("Import complete: %1 imported, %2 failed").arg(result.imported_).arg(result.failed_);
  if (!state_saved) {
    task_text += " (project sync/save failed)";
    setServiceMessageForCurrentProject("Import finished, but saving project state failed.");
  } else if (!package_saved) {
    task_text += " (project packing failed)";
    setServiceMessageForCurrentProject(
        package_error.isEmpty() ? "Import finished, but project packing failed."
                                : package_error);
  }
  setTaskState(task_text, 100, false);
  scheduleIdleTaskStateReset(1800);
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
  scheduleIdleTaskStateReset(1800);
}

void AlbumBackend::reapplyCurrentFilters() {
  applyFilters(static_cast<int>(last_join_op_));
  if (!validation_error_.isEmpty()) {
    rebuildThumbnailView(active_filter_ids_);
  }
}

void AlbumBackend::setServiceState(bool ready, const QString& message) {
  if (service_ready_ == ready && service_message_ == message) {
    return;
  }
  service_ready_   = ready;
  service_message_ = message;
  emit serviceStateChanged();
}

void AlbumBackend::setServiceMessageForCurrentProject(const QString& message) {
  setServiceState(project_ != nullptr, message);
}

void AlbumBackend::scheduleIdleTaskStateReset(int delayMs) {
  QTimer::singleShot(std::max(delayMs, 0), this, [this]() {
    if (!export_inflight_ && !task_cancel_visible_) {
      setTaskState("No background tasks", 0, false);
    }
  });
}

void AlbumBackend::setExportFailureState(const QString& message) {
  export_status_ = message;
  emit exportStateChanged();
  setTaskState(message, 0, false);
}

void AlbumBackend::resetExportProgressState(const QString& status) {
  export_status_        = status;
  export_error_summary_.clear();
  export_total_         = 0;
  export_completed_     = 0;
  export_succeeded_     = 0;
  export_failed_        = 0;
  export_skipped_       = 0;
  emit exportStateChanged();
}

auto AlbumBackend::collectExportTargets(const QVariantList& targetEntries) const
    -> std::vector<ExportTarget> {
  const QVariantList& source = targetEntries.empty() ? visible_thumbnails_ : targetEntries;
  std::vector<ExportTarget> targets;
  targets.reserve(static_cast<size_t>(source.size()));

  std::unordered_set<uint64_t> dedupe;
  dedupe.reserve(static_cast<size_t>(source.size()) * 2 + 1);

  for (const QVariant& entry : source) {
    const auto map       = entry.toMap();
    const auto elementId = static_cast<sl_element_id_t>(map.value("elementId").toUInt());
    const auto imageId   = static_cast<image_id_t>(map.value("imageId").toUInt());
    if (elementId == 0 || imageId == 0) {
      continue;
    }

    if (!dedupe.insert(ExportTargetKey(elementId, imageId)).second) {
      continue;
    }
    targets.emplace_back(elementId, imageId);
  }
  return targets;
}

auto AlbumBackend::buildExportQueue(const std::vector<ExportTarget>& targets,
                                    const std::filesystem::path&   outputDir,
                                    ImageFormatType                format,
                                    bool                           resizeEnabled,
                                    int                            maxLengthSide,
                                    int                            quality,
                                    ExportFormatOptions::BIT_DEPTH bitDepth,
                                    int                            pngCompressionLevel,
                                    ExportFormatOptions::TIFF_COMPRESS tiffCompression)
    -> ExportQueueBuildResult {
  ExportQueueBuildResult summary;
  if (!project_ || !export_service_) {
    summary.first_error_ = "Export service is unavailable.";
    return summary;
  }

  for (const auto& [elementId, imageId] : targets) {
    try {
      const auto srcPath = project_->GetImagePoolService()->Read<std::filesystem::path>(
          imageId,
          [](const std::shared_ptr<Image>& image) { return image ? image->image_path_ : image_path_t{}; });
      if (srcPath.empty()) {
        ++summary.skipped_count_;
        if (summary.first_error_.isEmpty()) {
          summary.first_error_ = "Image source path is empty.";
        }
        continue;
      }

      ExportTask task;
      task.sleeve_id_                  = elementId;
      task.image_id_                   = imageId;
      task.options_.format_            = format;
      task.options_.resize_enabled_    = resizeEnabled;
      task.options_.max_length_side_   = resizeEnabled ? maxLengthSide : 0;
      task.options_.quality_           = quality;
      task.options_.bit_depth_         = bitDepth;
      task.options_.compression_level_ = pngCompressionLevel;
      task.options_.tiff_compress_     = tiffCompression;
      task.options_.export_path_ =
          ExportPathForOptions(srcPath, outputDir, elementId, imageId, format);

      export_service_->EnqueueExportTask(task);
      ++summary.queued_count_;
    } catch (const std::exception& e) {
      ++summary.skipped_count_;
      if (summary.first_error_.isEmpty()) {
        summary.first_error_ = QString::fromUtf8(e.what());
      }
    } catch (...) {
      ++summary.skipped_count_;
      if (summary.first_error_.isEmpty()) {
        summary.first_error_ = "Unknown error while preparing export task.";
      }
    }
  }

  return summary;
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

bool AlbumBackend::isImageInCurrentFolder(const AlbumItem& image) const {
  return image.parent_folder_id == current_folder_id_;
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
      {"accent", image.accent.isEmpty() ? AccentForIndex(static_cast<size_t>(index)) : image.accent},
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

  // if (!basic.GetOperator(OperatorType::EXPOSURE).has_value()) {
  //   return false;
  // }

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
    if (isThumbnailPinned(finishedElement)) {
      requestThumbnail(finishedElement, finishedImage);
    } else {
      updateThumbnailDataUrl(finishedElement, QString());
    }
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
