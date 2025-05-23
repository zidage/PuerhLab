/*
 * @file        pu-erh_lab/src/include/sleeve/sleeve_element/sleeve_folder.hpp
 * @brief       A subtype of sleeve element representing a folder in a typical file system
 * @author      Yurun Zi
 * @date        2025-03-26
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
  std::unordered_map<file_name_t, sl_element_id_t>                            _contents;
  std::unordered_map<filter_id_t, std::shared_ptr<std::set<sl_element_id_t>>> _indicies_cache;

  filter_id_t                                                                 _default_filter;

  uint32_t                                                                    _file_count;
  uint32_t                                                                    _folder_count;

 public:
  explicit SleeveFolder(sl_element_id_t id, file_name_t element_name);
  ~SleeveFolder();

  auto Copy(sl_element_id_t new_id) const -> std::shared_ptr<SleeveElement>;

  void AddElementToMap(const std::shared_ptr<SleeveElement> element);
  void UpdateElementMap(const file_name_t &name, const sl_element_id_t old_id, const sl_element_id_t new_id);
  void CreateIndex(const std::shared_ptr<FilterCombo> filter);
  auto GetElementIdByName(const file_name_t &name) const -> std::optional<sl_element_id_t>;
  auto ListElements() const -> std::shared_ptr<std::vector<sl_element_id_t>>;
  auto ListFilters() const -> std::vector<filter_id_t>;
  auto Contains(const file_name_t &name) const -> bool;
  void RemoveNameFromMap(const file_name_t &name);

  void IncrementFolderCount();
  void IncrementFileCount();
  void DecrementFolderCount();
  void DecrementFileCount();
  auto Clear() -> bool;
  auto ResetFilters() -> bool;
};
};  // namespace puerhlab