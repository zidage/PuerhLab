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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dentry_cache_manager.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/controller/sleeve/element_controller.hpp"
#include "storage_service.hpp"
#include "type/type.hpp"
#include "utils/cache/lru_cache.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
class PathResolver {
 private:
  std::shared_ptr<SleeveFolder>        root_;
  LRUCache<sl_path_t, sl_element_id_t> directory_cache_;
  NodeStorageHandler&                  storage_handler_;
  IncrID::IDGenerator<uint32_t>&       id_gen_;

  auto                                 CoWHandler(const std::shared_ptr<SleeveElement> to_copy,
                                                  const std::shared_ptr<SleeveFolder>  parent_folder)
      -> std::shared_ptr<SleeveElement>;

 public:
  PathResolver();
  PathResolver(NodeStorageHandler& lazy_handler, IncrID::IDGenerator<uint32_t>& id_gen);
  void        SetRoot(std::shared_ptr<SleeveFolder> root);
  static auto Normalize(const std::filesystem::path raw_path) -> std::wstring;

  auto IsSubpath(const std::filesystem::path& base, const std::filesystem::path& target) -> bool;
  auto Contains(const std::filesystem::path& path, ElementType type) -> bool;
  auto Contains(const std::filesystem::path& path) -> bool;
  auto Resolve(const std::filesystem::path& path) -> std::shared_ptr<SleeveElement>;
  auto ResolveForWrite(const std::filesystem::path& path) -> std::shared_ptr<SleeveElement>;

  auto Tree(const std::filesystem::path& path) -> std::wstring;
};
};  // namespace puerhlab