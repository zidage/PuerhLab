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

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include "path_resolver.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage_service.hpp"
#include "type/type.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
class FileSystem {
  using NodeMapping = std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>;

 private:
  // A mapping between node id and pointers to actual nodes.
  NodeMapping                          _storage;
  std::shared_ptr<SleeveFolder>        _root;

  // ID Generation
  IncrID::IDGenerator<sl_element_id_t> _id_gen;
  /** @name Database interaction */
  ///@{
  std::filesystem::path                _db_path;
  std::filesystem::path                _meta_path;
  StorageService&                      _storage_service;
  NodeStorageHandler                   _storage_handler;
  PathResolver                         _resolver;
  ///@}

 public:
  // FileSystem(std::filesystem::path db_path, sl_element_id_t start_id);
  FileSystem(std::filesystem::path db_path, StorageService& storage_service,
             sl_element_id_t start_id);

  auto InitRoot() -> bool;

  auto Create(std::filesystem::path dest, std::wstring filename, ElementType type)
      -> std::shared_ptr<SleeveElement>;
  void Delete(std::filesystem::path target);
  auto Get(std::filesystem::path target, bool write) -> std::shared_ptr<SleeveElement>;
  auto Get(sl_element_id_t id) -> std::shared_ptr<SleeveElement>;

  auto ApplyFilterToFolder(const std::filesystem::path&       folder_path,
                           const std::shared_ptr<FilterCombo> filter)
      -> std::vector<std::shared_ptr<SleeveElement>>;

  void Copy(std::filesystem::path from, std::filesystem::path dest);

  void SyncToDB();
  void WriteSleeveMeta(const std::filesystem::path& meta_path);
  void ReadSleeveMeta(const std::filesystem::path& meta_path);

  auto Tree(const std::filesystem::path& path) -> std::wstring;
};
};  // namespace puerhlab