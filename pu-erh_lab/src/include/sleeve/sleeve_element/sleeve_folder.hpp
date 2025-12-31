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
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "sleeve_element.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief A type of element that contains files or folders of its kind
 *
 */
class SleeveFolder : public SleeveElement {
 protected:
  std::unordered_map<file_name_t, sl_element_id_t>              _contents;
  std::unordered_map<filter_id_t, std::vector<sl_element_id_t>> _indicies_cache;
  filter_id_t                                                   _default_filter = 0;

  uint32_t                                                      _file_count;
  uint32_t                                                      _folder_count;

 public:
  explicit SleeveFolder(sl_element_id_t id, file_name_t element_name);
  ~SleeveFolder();

  auto Copy(sl_element_id_t new_id) const -> std::shared_ptr<SleeveElement>;

  void AddElementToMap(const std::shared_ptr<SleeveElement> element);
  void ReplaceChild(const sl_element_id_t from, const sl_element_id_t to);
  void UpdateElementMap(const file_name_t& name, const sl_element_id_t old_id,
                        const sl_element_id_t new_id);
  auto GetElementIdByName(const file_name_t& name) const -> std::optional<sl_element_id_t>;
  auto ListElements() const -> const std::vector<sl_element_id_t>&;

  auto HasFilterIndex(const filter_id_t filter_id) const -> bool {
    return _indicies_cache.contains(filter_id);
  }

  auto ListElementsByFilter(const filter_id_t filter_id) const
      -> const std::vector<sl_element_id_t>&;
  auto Contains(const file_name_t& name) const -> bool;
  void RemoveNameFromMap(const file_name_t& name);

  void CreateIndex(const std::vector<std::shared_ptr<SleeveElement>>& filtered_elements,
                   const filter_id_t                                  filter_id);

  void IncrementFolderCount();
  void IncrementFileCount();
  void DecrementFolderCount();
  void DecrementFileCount();
  auto Clear() -> bool;
  auto ResetFilters() -> bool;
  auto ContentSize() -> size_t;
};
};  // namespace puerhlab