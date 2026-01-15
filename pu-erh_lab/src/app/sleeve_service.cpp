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

#include "app/sleeve_service.hpp"

#include <mutex>

#include "sleeve/storage_service.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
SleeveServiceImpl::SleeveServiceImpl(const std::filesystem::path& meta_path) {
  meta_path_ = meta_path;
  // Recover a already existing sleeve
  LoadSleeve(meta_path_);
}

SleeveServiceImpl::SleeveServiceImpl(const std::filesystem::path& db_path,
                                     const std::filesystem::path& meta_path,
                                     sl_element_id_t              start_id) {
  // Create a fresh sleeve
  db_path_         = db_path;
  meta_path_       = meta_path;
  storage_service_ = std::make_unique<StorageService>(db_path_);
  fs_              = std::make_unique<FileSystem>(db_path_, *storage_service_, start_id);
}

template <typename TResult>
auto SleeveServiceImpl::Read(std::function<TResult(const FileSystem&)> operation) -> TResult {
  std::lock_guard<std::mutex> lock(fs_lock_);
  return operation(*fs_);
}

template <typename TResult>
auto SleeveServiceImpl::Write(std::function<TResult(FileSystem&)> operation)
    -> std::pair<TResult, SyncResult> {
  std::lock_guard<std::mutex> lock(fs_lock_);

  // Perform the write operation
  TResult                     result      = operation(*fs_);

  // Automatic sync after write operation
  SyncResult                  sync_result = Sync();
  return {result, sync_result};
}

void SleeveServiceImpl::SaveSleeve(const std::filesystem::path& meta_path) {
  std::lock_guard<std::mutex> lock(fs_lock_);
  nlohmann::json              metadata;
  meta_path_            = meta_path;
  metadata["db_path"]   = conv::ToBytes(db_path_.wstring());
  metadata["meta_path"] = conv::ToBytes(meta_path_.wstring());
  metadata["start_id"]  = fs_->GetCurrentID();

  std::ofstream file(meta_path);
  if (file.is_open()) {
    file << metadata.dump(4);
    file.close();
  }
}

void SleeveServiceImpl::LoadSleeve(const std::filesystem::path& meta_path) {
  std::lock_guard<std::mutex> lock(fs_lock_);
  std::ifstream               file(meta_path);
  if (file.is_open()) {
    nlohmann::json metadata;
    file >> metadata;
    db_path_                 = std::filesystem::path(conv::FromBytes(metadata["db_path"]));
    meta_path_               = std::filesystem::path(conv::FromBytes(metadata["meta_path"]));
    sl_element_id_t start_id = static_cast<sl_element_id_t>(metadata["start_id"]);
    storage_service_         = std::make_unique<StorageService>(db_path_);
    fs_                      = std::make_unique<FileSystem>(db_path_, *storage_service_, start_id);

    // Re-init the root to trigger loading from DB
    fs_->InitRoot();
  }
}

auto SleeveServiceImpl::Sync() -> SyncResult {
  SyncResult result{true, 0, ""};
  try {
    auto& element_ctrl      = storage_service_->GetElementController();
    auto  modified_elements = fs_->GetModifiedElements();
    auto  unsynced_elements = fs_->GetUnsyncedElements();

    // Sync unsynced elements first
    for (auto& element : unsynced_elements) {
      element_ctrl.AddElement(element);
      result.elements_synced_++;
    }
    // Then sync modified elements
    for (auto& element : modified_elements) {
      element_ctrl.UpdateElement(element);
      result.elements_synced_++;
    }
  } catch (std::exception& e) {
    result.success_ = false;
    result.message_ = e.what();
  }
  return result;
}
};  // namespace puerhlab