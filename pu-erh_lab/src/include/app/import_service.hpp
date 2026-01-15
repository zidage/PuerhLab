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

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>

#include "concurrency/thread_pool.hpp"
#include "image/image.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"
#include "utils/id/id_generator.hpp"
#include "utils/import/import_log.hpp"

namespace puerhlab {
enum class ImportSortMode : uint8_t {
  NONE      = 0,
  FILE_NAME = 1,  // DEFAULT
  FULL_PATH = 2
};

struct ImportOptions {
  ImportSortMode sort_mode_                        = ImportSortMode::FILE_NAME;

  bool           persist_placeholders_immediately_ = false;

  // If set, record the sequence for deterministic import order.
  bool           write_import_sequence_            = true;
};

struct ImportProgress {
  uint32_t              total_                = 0;

  // Phase A (fast): created SleeveFile nodes + bindings (image_id -> SleeveFile)
  std::atomic<uint32_t> placeholders_created_ = 0;

  // Phase B (slow): metadata extracted + ImagePool updated + DB updated (as applicable)
  std::atomic<uint32_t> metadata_done_        = 0;

  std::atomic<uint32_t> failed_               = 0;
};

enum class ImportErrorCode : uint8_t {
  UNKNOWN = 0,
  FILE_NOT_FOUND,
  UNSUPPORTED_FORMAT,
  READ_FAILED,
  METADATA_EXTRACTION_FAILED,
  SLEEVE_CREATE_FAILED,
  DB_WRITE_FAILED,
  CANCELED
};

struct ImportError {
  ImportErrorCode code_ = ImportErrorCode::UNKNOWN;
  image_path_t    path_{};
  std::string     message_{};
};

struct ImportResult {
  uint32_t requested_ = 0;
  uint32_t imported_  = 0;
  uint32_t failed_    = 0;
};

class ImportJob {
 public:
  using ProgressCallback = std::function<void(const ImportProgress&)>;
  using FinishedCallback = std::function<void(const ImportResult&)>;

  // Cancellation toke observed by implementation
  std::atomic<bool>          canceled_{false};
  std::atomic<bool>          cancelation_acked_{false};

  ProgressCallback           on_progress_{};
  FinishedCallback           on_finished_{};

  std::shared_ptr<ImportLog> import_log_ = nullptr;

  ~ImportJob()                           = default;

  auto IsCancelled() const -> bool { return canceled_.load(); }
  auto IsCancelationAcked() const -> bool { return cancelation_acked_.load(); }
};

class ImportService {
 public:
  virtual ~ImportService() = default;

  virtual auto ImportToFolder(const std::vector<image_path_t>& paths, const image_path_t& dest,
                              const ImportOptions&       options = {},
                              std::shared_ptr<ImportJob> job     = nullptr)
      -> std::shared_ptr<ImportJob>                           = 0;

  virtual void CleanupFailedImports(const ImportLogSnapshot& log_snapshot,
                                    const image_path_t&      dest) = 0;
};

class ImportServiceImpl final : public ImportService {
 public:
  ImportServiceImpl() = default;
  ImportServiceImpl(std::shared_ptr<FileSystem>       fs,
                    std::shared_ptr<ImagePoolManager> image_pool_manager)
      : fs_(fs), image_pool_manager_(image_pool_manager) {}

  ~ImportServiceImpl()                                  = default;

  std::shared_ptr<FileSystem>       fs_                 = nullptr;
  std::shared_ptr<ImagePoolManager> image_pool_manager_ = nullptr;

  ThreadPool                        thread_pool_{8};

  auto ImportToFolder(const std::vector<image_path_t>& paths, const image_path_t& dest,
                      const ImportOptions& options = {}, std::shared_ptr<ImportJob> job = nullptr)
      -> std::shared_ptr<ImportJob> override;

  void CleanupFailedImports(const ImportLogSnapshot& log_snapshot,
                            const image_path_t&      dest) override;
};
};  // namespace puerhlab