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

#include "app/project_service.hpp"

#include <fstream>
#include <json.hpp>
#include <stdexcept>

#include "utils/string/convert.hpp"

namespace puerhlab {
ProjectService::ProjectService(const std::filesystem::path& db_path,
                               const std::filesystem::path& meta_path)
    : db_path_(db_path), meta_path_(meta_path) {
  try {
    LoadProject(meta_path);
  } catch (...) {
    // Load failed, create new project
    storage_service_ = std::make_shared<StorageService>(db_path_);
    RecreateSleeveService(0);
    pool_service_ = std::make_shared<ImagePoolService>(storage_service_, 0);
  }
}

ProjectService::~ProjectService() {
  pool_service_.reset();
  sleeve_service_.reset();
  storage_service_.reset();
}

void ProjectService::SaveProject(const std::filesystem::path& meta_path) {
  if (!sleeve_service_) {
    throw std::runtime_error("SleeveService is not initialized");
  }

  meta_path_ = meta_path;

  nlohmann::json metadata;
  metadata["db_path"]             = conv::ToBytes(db_path_.wstring());
  metadata["meta_path"]           = conv::ToBytes(meta_path_.wstring());
  metadata["start_id"]            = sleeve_service_->GetCurrentID();
  metadata["image_pool_start_id"] = pool_service_->GetCurrentID();

  std::ofstream file(meta_path_);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open meta file for writing");
  }
  file << metadata.dump(4);
  file.close();
}

void ProjectService::LoadProject(const std::filesystem::path& meta_path) {
  std::ifstream file(meta_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open meta file for reading");
  }

  nlohmann::json metadata;
  file >> metadata;

  db_path_                 = std::filesystem::path(conv::FromBytes(metadata["db_path"]));
  meta_path_               = std::filesystem::path(conv::FromBytes(metadata["meta_path"]));
  sl_element_id_t start_id = static_cast<sl_element_id_t>(metadata["start_id"]);
  sl_element_id_t image_pool_start_id =
      static_cast<sl_element_id_t>(metadata["image_pool_start_id"]);
  storage_service_ = std::make_shared<StorageService>(db_path_);
  RecreateSleeveService(start_id);
  pool_service_ = std::make_shared<ImagePoolService>(storage_service_, image_pool_start_id);
}

void ProjectService::RecreateSleeveService(sl_element_id_t start_id) {
  if (!storage_service_) {
    throw std::runtime_error("StorageService is not initialized");
  }
  sleeve_service_ = std::make_shared<SleeveServiceImpl>(storage_service_, db_path_, start_id);
}
};  // namespace puerhlab
