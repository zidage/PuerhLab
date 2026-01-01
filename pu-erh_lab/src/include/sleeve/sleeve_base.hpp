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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>

#include "dentry_cache_manager.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "storage_service.hpp"
#include "type/type.hpp"

namespace puerhlab {

// TODO: Implement access guard
class ElementAccessGuard {
 public:
  std::shared_ptr<SleeveElement> access_element_;

  ElementAccessGuard(std::shared_ptr<SleeveElement> element);
  ~ElementAccessGuard();
};

class SleeveBase {
 private:
  size_t                                                              size_;
  sl_element_id_t                                                     next_element_id_;
  filter_id_t                                                         next_filter_id_;

  std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>> storage_;
  std::unordered_map<filter_id_t, std::shared_ptr<FilterCombo>>       filter_storage_;

  DCacheManager                                                       dentry_cache_;
  uint32_t                                                            dcache_capacity_;

  std::wstring                                                        delimiter_ = L"/";
  std::wregex                                                         re_;

  auto GetWriteGuard(const std::shared_ptr<SleeveFolder> parent_folder,
                     const file_name_t& file_name) -> std::optional<ElementAccessGuard>;
  auto WriteCopy(std::shared_ptr<SleeveElement> src_element,
                 std::shared_ptr<SleeveFolder>  dest_folder) -> std::shared_ptr<SleeveElement>;

 public:
  sleeve_id_t                   sleeve_id_;
  std::shared_ptr<SleeveFolder> root_;

  explicit SleeveBase(sleeve_id_t id);

  void InitializeRoot();

  auto GetStorage() -> std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>&;
  auto GetFilterStorage() -> std::unordered_map<filter_id_t, std::shared_ptr<FilterCombo>>&;

  auto AccessElementById(const sl_element_id_t& id) const
      -> std::optional<std::shared_ptr<SleeveElement>>;
  auto AccessElementByPath(const sl_path_t& path) -> std::optional<std::shared_ptr<SleeveElement>>;

  auto CreateElementToPath(const sl_path_t& path, const file_name_t& file_name,
                           const ElementType& type)
      -> std::optional<std::shared_ptr<SleeveElement>>;

  auto RemoveElementInPath(const sl_path_t& target)
      -> std::optional<std::shared_ptr<SleeveElement>>;
  auto RemoveElementInPath(const sl_path_t& path, const file_name_t& file_name)
      -> std::optional<std::shared_ptr<SleeveElement>>;

  auto CopyElement(const sl_path_t& src, const sl_path_t& dest)
      -> std::optional<std::shared_ptr<SleeveElement>>;

  auto MoveElement(const sl_path_t& src, const sl_path_t& dest)
      -> std::optional<std::shared_ptr<SleeveElement>>;

  auto GetReadGuard(const sl_path_t& target) -> std::optional<ElementAccessGuard>;

  auto GetWriteGuard(const sl_path_t& target) -> std::optional<ElementAccessGuard>;
  auto GetWriteGuard(const sl_path_t& parent_folder_path, const file_name_t& file_name)
      -> std::optional<ElementAccessGuard>;

  void GarbageCollect();

  auto Tree(const sl_path_t& path) -> std::wstring;
  auto TreeBFS(const sl_path_t& path) -> std::wstring;

  auto IsSubFolder(const std::shared_ptr<SleeveFolder> folder_a, const sl_path_t& path_b) const
      -> bool;
};
};  // namespace puerhlab