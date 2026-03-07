//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <filesystem>
#include <memory>

#include "app/sleeve_service.hpp"
#include "image_pool_service.hpp"
#include "sleeve/storage_service.hpp"
#include "type/type.hpp"

namespace puerhlab {
enum class ProjectOpenMode {
  kLoadOrCreate = 0,
  kLoadExisting,
  kCreateNew,
};

class ProjectService {
 public:
  ProjectService(const std::filesystem::path& db_path, const std::filesystem::path& meta_path,
                 ProjectOpenMode open_mode = ProjectOpenMode::kLoadOrCreate);
  ~ProjectService();

  void SaveProject(const std::filesystem::path& meta_path);
  void LoadProject(const std::filesystem::path& meta_path);

  auto GetStorageService() const -> std::shared_ptr<StorageService> { return storage_service_; }

  auto GetSleeveService() const -> std::shared_ptr<SleeveServiceImpl> { return sleeve_service_; }
  auto GetImagePoolService() const -> std::shared_ptr<ImagePoolService> {
    return pool_service_;
  }

  auto GetDBPath() const -> const std::filesystem::path& { return db_path_; }
  auto GetMetaPath() const -> const std::filesystem::path& { return meta_path_; }

 private:
  void                                  RecreateSleeveService(sl_element_id_t start_id);

  std::filesystem::path                 db_path_;
  std::filesystem::path                 meta_path_;
  std::shared_ptr<StorageService>       storage_service_;
  std::shared_ptr<SleeveServiceImpl>    sleeve_service_;
  // TODO: Add ImagePoolService and store its start_id into the metadata
  std::shared_ptr<ImagePoolService> pool_service_;
};
};  // namespace puerhlab
