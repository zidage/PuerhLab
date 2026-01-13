//  Copyright 2025 Yurun Zi
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

// TODO: Refactor to UI layer as "LibrarySession"

#pragma once

#include <opencv2/core/hal/interface.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include "io/image/image_loader.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "sleeve_base.hpp"
#include "sleeve_view.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage_service.hpp"
#include "type/type.hpp"

namespace puerhlab {
class SleeveManager {
 private:
  std::shared_ptr<FileSystem>       fs_;
  std::shared_ptr<SleeveView>       view_;
  std::shared_ptr<ImagePoolManager> image_pool_;

  StorageService                    storage_service_;

 public:
  explicit SleeveManager(std::filesystem::path db_path);

  auto GetFilesystem() -> std::shared_ptr<FileSystem>;
  auto GetView() -> std::shared_ptr<SleeveView>;
  auto GetPool() -> std::shared_ptr<ImagePoolManager>;
  auto GetImgCount() -> uint32_t;
  auto GetStorageService() -> StorageService& { return storage_service_; }

  auto LoadToPath(std::vector<image_path_t> img_os_path, sl_path_t dest) -> uint32_t;

  auto RestoreSleeveFromDB(sleeve_id_t sleeve_id) -> std::shared_ptr<FileSystem>;
};
};  // namespace puerhlab