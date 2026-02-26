#include "ui/puerhlab_main/album_backend/album_backend.hpp"

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

#include "ui/puerhlab_main/editor_dialog/editor_dialog.hpp"
#include "app/render_service.hpp"
#include "image/image.hpp"
#include "io/image/image_loader.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "type/supported_file_type.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab::ui {
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

auto RootFsPath() -> std::filesystem::path {
  return std::filesystem::path(L"/");
}

auto RootPathText() -> QString {
  return "\\";
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
    return RootPathText();
  }
#if defined(_WIN32)
  const QString text = QString::fromStdWString(path.generic_wstring());
#else
  const QString text = QString::fromStdString(path.generic_string());
#endif
  if (text == "/" || text == "\\") {
    return RootPathText();
  }
  QString normalized = text;
  normalized.replace('/', '\\');
  if (!normalized.startsWith('\\')) {
    normalized.prepend('\\');
  }
  return normalized;
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
  const auto root     = temp_dir / L"puerh_lab_main";
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
  const auto root     = temp_dir / L"puerh_lab_main";
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

  InitializeEditorLuts();
  SetServiceState(
      false,
      "Select a project: load a .puerhproj package or metadata JSON, or create a new packed project.");
  task_status_ = "Open or create a project to begin.";
}

AlbumBackend::~AlbumBackend() {
  try {
    ReleaseVisibleThumbnailPins();
    FinalizeEditorSession(true);
    if (current_import_job_) {
      current_import_job_->canceled_.store(true);
    }
    if (pipeline_service_) {
      pipeline_service_->Sync();
    }
    if (PersistCurrentProjectState()) {
      QString ignored_error;
      (void)PackageCurrentProjectFiles(&ignored_error);
    }
    CleanupWorkspaceDirectory(project_workspace_dir_);
  } catch (...) {
  }
}

#include "album_backend_filter.cpp"
#include "album_backend_import_export.cpp"
#include "album_backend_project.cpp"
#include "album_backend_library.cpp"
#include "album_backend_editor.cpp"

}  // namespace puerhlab::ui
