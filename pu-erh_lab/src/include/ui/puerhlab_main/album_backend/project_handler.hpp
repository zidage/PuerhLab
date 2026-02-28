#pragma once

#include <QVariantList>

#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ui/puerhlab_main/album_backend/album_types.hpp"
#include "app/export_service.hpp"
#include "app/history_mgmt_service.hpp"
#include "app/import_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "app/sleeve_filter_service.hpp"
#include "app/thumbnail_service.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Owns all back-end services and manages lifetime of a single project.
class ProjectHandler {
 public:
  explicit ProjectHandler(AlbumBackend& backend);

  bool InitializeServices(const std::filesystem::path& dbPath,
                          const std::filesystem::path& metaPath,
                          ProjectOpenMode              openMode,
                          const std::filesystem::path& packagePath = {},
                          const std::filesystem::path& workspaceDir = {});
  bool PersistCurrentProjectState();
  bool PackageCurrentProjectFiles(QString* errorOut = nullptr) const;
  auto CollectProjectSnapshot() const -> ProjectSnapshot;
  void ApplyLoadedProjectEntriesBatch();
  void SetProjectLoadingState(bool loading, const QString& message);
  void ClearProjectData();

  [[nodiscard]] auto project() const -> const std::shared_ptr<ProjectService>& { return project_; }
  [[nodiscard]] auto pipeline_service() const -> const std::shared_ptr<PipelineMgmtService>& {
    return pipeline_service_;
  }
  [[nodiscard]] auto history_service() const -> const std::shared_ptr<EditHistoryMgmtService>& {
    return history_service_;
  }
  [[nodiscard]] auto thumbnail_service() const -> const std::shared_ptr<ThumbnailService>& {
    return thumbnail_service_;
  }
  [[nodiscard]] auto filter_service() const -> SleeveFilterService* { return filter_service_.get(); }
  [[nodiscard]] auto import_service() const -> ImportServiceImpl* { return import_service_.get(); }
  [[nodiscard]] auto export_service() const -> const std::shared_ptr<ExportService>& {
    return export_service_;
  }

  [[nodiscard]] auto db_path() const -> const std::filesystem::path& { return db_path_; }
  [[nodiscard]] auto meta_path() const -> const std::filesystem::path& { return meta_path_; }
  [[nodiscard]] auto package_path() const -> const std::filesystem::path& { return project_package_path_; }
  [[nodiscard]] auto workspace_dir() const -> const std::filesystem::path& { return project_workspace_dir_; }
  [[nodiscard]] bool project_loading() const { return project_loading_; }
  [[nodiscard]] auto project_loading_message() const -> const QString& { return project_loading_message_; }

 private:
  AlbumBackend& backend_;

  std::shared_ptr<ProjectService>        project_{};
  std::shared_ptr<PipelineMgmtService>   pipeline_service_{};
  std::shared_ptr<EditHistoryMgmtService> history_service_{};
  std::shared_ptr<ThumbnailService>      thumbnail_service_{};
  std::unique_ptr<SleeveFilterService>   filter_service_{};
  std::unique_ptr<ImportServiceImpl>     import_service_{};
  std::shared_ptr<ExportService>         export_service_{};

  std::filesystem::path db_path_{};
  std::filesystem::path meta_path_{};
  std::filesystem::path project_package_path_{};
  std::filesystem::path project_workspace_dir_{};

  bool     project_loading_         = false;
  QString  project_loading_message_{};
  uint64_t project_load_request_id_ = 0;

  std::vector<ExistingAlbumEntry>                          pending_project_entries_{};
  std::vector<ExistingFolderEntry>                         pending_folder_entries_{};
  std::unordered_map<sl_element_id_t, sl_element_id_t>     pending_folder_parent_by_id_{};
  std::unordered_map<sl_element_id_t, std::filesystem::path> pending_folder_path_by_id_{};
  size_t pending_project_entry_index_ = 0;
};

}  // namespace puerhlab::ui
