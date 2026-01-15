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

#include <opencv2/core/hal/interface.h>

#include <memory>

#include "sleeve/sleeve_filesystem.hpp"
#include "sleeve/storage_service.hpp"
#include "type/type.hpp"

namespace puerhlab {
// An interaction level below SleeveManager (Future Sleeve API for UI) and above ImagePoolManager
// and FileSystem
enum class FSOperationType {
  READ,  // Read-only operation, no sync required
  WRITE  // Write operation, requires sync after operation
};

struct SyncResult {
  bool        success_         = true;
  uint32_t    elements_synced_ = 0;
  std::string message_         = "";
};

class SleeveService {
 public:
  virtual ~SleeveService() = default;

  template <typename TResult>
  auto Read(std::function<TResult(FileSystem&)> operation) -> TResult;

  template <typename TResult>
  auto Write(std::function<TResult(FileSystem&)> operation) -> std::pair<TResult, SyncResult>;

  virtual auto Sync() -> SyncResult                               = 0;

  virtual void SaveSleeve(const std::filesystem::path& meta_path) = 0;
  virtual void LoadSleeve(const std::filesystem::path& meta_path) = 0;
};

class SleeveServiceImpl final : public SleeveService {
 private:
  std::unique_ptr<StorageService> storage_service_;
  std::unique_ptr<FileSystem>     fs_;
  std::filesystem::path           db_path_;
  std::filesystem::path           meta_path_;

  mutable std::mutex fs_lock_;  // Use a fat lock for now, but I have no plan to design a
                                // fine-grained concurrency model for FS

 public:
  SleeveServiceImpl() = delete;
  SleeveServiceImpl(const std::filesystem::path& meta_path);
  SleeveServiceImpl(const std::filesystem::path& db_path, const std::filesystem::path& meta_path,
                    sl_element_id_t start_id = 0);

  template <typename TResult>
  auto Read(std::function<TResult(FileSystem&)> operation) -> TResult {
    std::lock_guard<std::mutex> lock(fs_lock_);
    return operation(*fs_);
  }

  template <typename TResult>
  auto Write(std::function<TResult(FileSystem&)> operation) -> std::pair<TResult, SyncResult> {
    std::lock_guard<std::mutex> lock(fs_lock_);

    // Perform the write operation
    TResult                     result      = operation(*fs_);

    // Automatic sync after write operation
    SyncResult                  sync_result = Sync();
    return {result, sync_result};
  }

  auto Sync() -> SyncResult override;

  void SaveSleeve(const std::filesystem::path& meta_path) override;
  void LoadSleeve(const std::filesystem::path& meta_path) override;
};
};  // namespace puerhlab