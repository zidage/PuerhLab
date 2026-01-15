//  Copyright 2026 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

#include <filesystem>
#include <memory>

#include "app/sleeve_service.hpp"
#include "sleeve/storage_service.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ProjectService {
 public:
  ProjectService(const std::filesystem::path& db_path, const std::filesystem::path& meta_path,
                 sl_element_id_t start_id = 0);

  void SaveProject(const std::filesystem::path& meta_path);
  void LoadProject(const std::filesystem::path& meta_path);

  auto GetStorageService() const -> std::shared_ptr<StorageService> {
    return storage_service_;
  }

  auto GetSleeveService() const -> std::shared_ptr<SleeveServiceImpl> {
    return sleeve_service_;
  }

  auto GetDBPath() const -> const std::filesystem::path& { return db_path_; }
  auto GetMetaPath() const -> const std::filesystem::path& { return meta_path_; }

 private:
  void RecreateSleeveService(sl_element_id_t start_id);

  std::filesystem::path           db_path_;
  std::filesystem::path           meta_path_;
  std::shared_ptr<StorageService> storage_service_;
  std::shared_ptr<SleeveServiceImpl> sleeve_service_;
};
};  // namespace puerhlab
