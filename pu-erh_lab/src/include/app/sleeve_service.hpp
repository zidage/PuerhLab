//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <opencv2/core/hal/interface.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>

#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "sleeve/storage_service.hpp"
#include "type/type.hpp"

namespace puerhlab {
// An interaction level below SleeveManager (Future Sleeve API for UI) and above ImagePoolManager
// and FileSystem

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
  auto Write(std::function<TResult(FileSystem&)> operation)
      -> std::conditional_t<std::is_void_v<TResult>, SyncResult,
                            std::pair<TResult, SyncResult>>;

  virtual auto Sync() -> SyncResult = 0;
};

class SleeveServiceImpl final : public SleeveService {
 private:
  std::shared_ptr<StorageService> storage_service_;
  std::unique_ptr<FileSystem>     fs_;
  std::filesystem::path           db_path_;

  mutable std::mutex fs_lock_;  // Use a fat lock for now, but I have no plan to design a
                                // fine-grained concurrency model for FS

 public:
  SleeveServiceImpl() = delete;
  SleeveServiceImpl(std::shared_ptr<StorageService> storage_service,
                    const std::filesystem::path&   db_path,
                    sl_element_id_t                start_id = 0);

  template <typename TResult>
  auto Read(std::function<TResult(FileSystem&)> operation) -> TResult {
    std::lock_guard<std::mutex> lock(fs_lock_);
    if constexpr (std::is_void_v<TResult>) {
      operation(*fs_);
      return;
    } else {
      return operation(*fs_);
    }
  }

  template <typename TResult>
  auto Write(std::function<TResult(FileSystem&)> operation)
      -> std::conditional_t<std::is_void_v<TResult>, SyncResult,
                            std::pair<TResult, SyncResult>> {
    std::lock_guard<std::mutex> lock(fs_lock_);

    if constexpr (std::is_void_v<TResult>) {
      operation(*fs_);
      return Sync();  // void: only return SyncResult
    } else {
      TResult    result      = operation(*fs_);
      SyncResult sync_result = Sync();
      return {result, sync_result};
    }
  }

  template <typename TResult>
  auto Write_NoSync(std::function<TResult(FileSystem&)> operation) -> TResult {
    std::lock_guard<std::mutex> lock(fs_lock_);

    if constexpr (std::is_void_v<TResult>) {
      operation(*fs_);
      return;
    } else {
      TResult result = operation(*fs_);
      return result;
    }
  }


  auto Sync() -> SyncResult override;

  auto GetCurrentID() const -> sl_element_id_t;
  auto ResolveElement(const std::filesystem::path& path) -> std::shared_ptr<SleeveElement>;
  auto ResolveFolder(const std::filesystem::path& path) -> std::shared_ptr<SleeveFolder>;
  auto ResolveFile(const std::filesystem::path& path) -> std::shared_ptr<SleeveFile>;
  auto ListFolderEntries(const std::filesystem::path& folder_path)
      -> std::vector<std::shared_ptr<SleeveElement>>;
  auto CreateFolder(const std::filesystem::path& parent_path, const file_name_t& name)
      -> std::pair<std::shared_ptr<SleeveFolder>, SyncResult>;
  auto DeletePath(const std::filesystem::path& target_path) -> SyncResult;

  auto GetStorageService() -> std::shared_ptr<StorageService> {
    return storage_service_;
  }
};
};  // namespace puerhlab
