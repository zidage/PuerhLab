#include "ui/puerhlab_main/album_backend/packed_project.hpp"

#include <QDateTime>
#include <QDir>
#include <QRandomGenerator>
#include <QStandardPaths>

#include <algorithm>
#include <cwctype>
#include <fstream>
#include <limits>

#include <duckdb.h>
#include <json.hpp>

#include "ui/puerhlab_main/album_backend/path_utils.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab::ui::packed_proj {

namespace {

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
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(stream);
}

auto WriteU64Le(std::ostream& stream, uint64_t value) -> bool {
  std::array<unsigned char, 8> bytes{};
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<unsigned char>((value >> (i * 8ULL)) & 0xFFULL);
  }
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
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

}  // namespace

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

auto BuildUniquePackedProjectPath(const std::filesystem::path& folder,
                                  const QString& projectName,
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

  const auto valid_name = album_util::ValidateProjectName(projectName, errorOut);
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
                            QString* errorOut) -> bool {
  const QString temp_dir_text =
      QStandardPaths::writableLocation(QStandardPaths::TempLocation).isEmpty()
          ? QDir::tempPath()
          : QStandardPaths::writableLocation(QStandardPaths::TempLocation);
  const auto temp_dir = album_util::QStringToFsPath(temp_dir_text);
  const auto root     = temp_dir / L"puerh_lab_main";
  if (!album_util::EnsureDirectoryExists(root)) {
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

auto BuildRuntimeProjectPair(const std::filesystem::path& workspace,
                             const QString& projectName)
    -> std::pair<std::filesystem::path, std::filesystem::path> {
  QString base = projectName.trimmed();
  if (base.isEmpty()) {
    base = "album_editor_project";
  }
  if (base.endsWith(".puerhproj", Qt::CaseInsensitive)) {
    base.chop(QString(".puerhproj").size());
    base = base.trimmed();
  }
  auto valid_name = album_util::ValidateProjectName(base, nullptr);
  if (valid_name.has_value()) {
    base = valid_name.value();
  } else {
    base = "album_editor_project";
  }
  const std::wstring stem = base.toStdWString();
  return {workspace / (stem + L".db"), workspace / (stem + L".json")};
}

auto RunDuckDbQuery(duckdb_connection conn, const std::string& sql,
                    const char* stage, QString* errorOut) -> bool {
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

auto BuildTempDbSnapshotPath(std::filesystem::path* snapshotPathOut,
                             QString* errorOut) -> bool {
  const QString temp_dir_text =
      QStandardPaths::writableLocation(QStandardPaths::TempLocation).isEmpty()
          ? QDir::tempPath()
          : QStandardPaths::writableLocation(QStandardPaths::TempLocation);
  const auto temp_dir = album_util::QStringToFsPath(temp_dir_text);
  const auto root     = temp_dir / L"puerh_lab_main";
  if (!album_util::EnsureDirectoryExists(root)) {
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
                          const std::filesystem::path& snapshotPath,
                          QString* errorOut) -> bool {
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

    const std::string path_utf8 = conv::ToBytes(snapshotPath.generic_wstring());
    const std::string escaped   = album_util::EscapeSqlStringLiteral(path_utf8);
    std::string       source_catalog;
    if (!QueryCurrentCatalog(guard.conn_, &source_catalog, errorOut)) {
      std::filesystem::remove(snapshotPath, ec);
      return false;
    }
    const std::string source_ident =
        "\"" + album_util::EscapeSqlIdentifier(source_catalog) + "\"";

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
                        .arg(album_util::PathToQString(snapshotPath));
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
                        const std::filesystem::path& dbPath,
                        QString* errorOut) -> bool {
  std::string meta_bytes;
  if (!ReadFileBytes(metaPath, &meta_bytes)) {
    if (errorOut) {
      *errorOut = QString("Failed to read project metadata: %1")
                      .arg(album_util::PathToQString(metaPath));
    }
    return false;
  }

  std::string db_bytes;
  if (!ReadFileBytes(dbPath, &db_bytes)) {
    if (errorOut) {
      *errorOut = QString("Failed to read project database: %1")
                      .arg(album_util::PathToQString(dbPath));
    }
    return false;
  }

  if (!album_util::EnsureDirectoryExists(packedPath.parent_path())) {
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
                      .arg(album_util::PathToQString(packedPath));
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
        *errorOut = QString("Failed to replace existing packed project file: %1")
                        .arg(album_util::PathToQString(packedPath));
      }
      std::filesystem::remove(std::filesystem::path(temp_path), ec);
      return false;
    }
  }
  ec.clear();
  std::filesystem::rename(std::filesystem::path(temp_path), packedPath, ec);
  if (ec) {
    if (errorOut) {
      *errorOut = QString("Failed to finalize packed project file: %1")
                      .arg(album_util::PathToQString(packedPath));
    }
    std::filesystem::remove(std::filesystem::path(temp_path), ec);
    return false;
  }

  return true;
}

auto ReadPackedProject(const std::filesystem::path& packedPath,
                       std::string* metaBytes, std::string* dbBytes,
                       QString* errorOut) -> bool {
  std::ifstream in(packedPath, std::ios::binary);
  if (!in.is_open()) {
    if (errorOut) {
      *errorOut = QString("Failed to open packed project: %1")
                      .arg(album_util::PathToQString(packedPath));
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
                              QString* errorOut) -> bool {
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
                      .arg(album_util::PathToQString(db_path));
    }
    return false;
  }

  const std::string meta_text = metadata.dump(4);
  if (!WriteFileBytes(meta_path, meta_text)) {
    if (errorOut) {
      *errorOut = QString("Failed to materialize project metadata: %1")
                      .arg(album_util::PathToQString(meta_path));
    }
    return false;
  }

  *dbPathOut   = db_path;
  *metaPathOut = meta_path;
  return true;
}

}  // namespace puerhlab::ui::packed_proj
