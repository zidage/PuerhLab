#pragma once

#include <QString>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "app/project_service.hpp"

namespace puerhlab::ui::packed_proj {

constexpr std::wstring_view kPackedProjectExtension = L".puerhproj";
constexpr std::array<char, 8> kPackedProjectMagic{
    {'P', 'U', 'E', 'R', 'H', 'P', 'K', '1'}};
constexpr uint32_t kPackedProjectVersion = 1;
constexpr uint64_t kMaxPackedComponentBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

auto IsMetadataJsonPath(const std::filesystem::path& path) -> bool;
auto IsPackedProjectPath(const std::filesystem::path& path) -> bool;
auto IsPackedProjectFile(const std::filesystem::path& path) -> bool;

auto ReadFileBytes(const std::filesystem::path& path, std::string* out) -> bool;
auto WriteFileBytes(const std::filesystem::path& path, const std::string& data) -> bool;

auto BuildUniquePackedProjectPath(const std::filesystem::path& folder,
                                  const QString& projectName,
                                  QString* errorOut) -> std::optional<std::filesystem::path>;
auto BuildBundlePathFromMetaPath(const std::filesystem::path& metaPath) -> std::filesystem::path;

auto CreateProjectWorkspace(const QString& projectName,
                            std::filesystem::path* workspaceOut,
                            QString* errorOut) -> bool;
auto BuildRuntimeProjectPair(const std::filesystem::path& workspace,
                             const QString& projectName)
    -> std::pair<std::filesystem::path, std::filesystem::path>;

auto RunDuckDbQuery(duckdb_connection conn, const std::string& sql,
                    const char* stage, QString* errorOut) -> bool;
auto QueryCurrentCatalog(duckdb_connection conn, std::string* catalogOut,
                         QString* errorOut) -> bool;

auto BuildTempDbSnapshotPath(std::filesystem::path* snapshotPathOut,
                             QString* errorOut) -> bool;
auto CreateLiveDbSnapshot(const std::shared_ptr<ProjectService>& project,
                          const std::filesystem::path& snapshotPath,
                          QString* errorOut) -> bool;

auto WritePackedProject(const std::filesystem::path& packedPath,
                        const std::filesystem::path& metaPath,
                        const std::filesystem::path& dbPath,
                        QString* errorOut) -> bool;
auto ReadPackedProject(const std::filesystem::path& packedPath,
                       std::string* metaBytes, std::string* dbBytes,
                       QString* errorOut) -> bool;

auto UnpackProjectToWorkspace(const std::filesystem::path& packedPath,
                              const std::filesystem::path& workspaceDir,
                              const QString& projectName,
                              std::filesystem::path* dbPathOut,
                              std::filesystem::path* metaPathOut,
                              QString* errorOut) -> bool;

}  // namespace puerhlab::ui::packed_proj
