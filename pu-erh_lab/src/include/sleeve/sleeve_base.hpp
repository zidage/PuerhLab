/*
 * @file        pu-erh_lab/src/include/sleeve/sleeve_base.hpp
 * @brief       A data structure used with DuckDB to store indexed-images
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

#include <opencv2/core/hal/interface.h>

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
#include "type/type.hpp"

namespace puerhlab {

// TODO: Implement access guard
class ElementAccessGuard {
 public:
  std::shared_ptr<SleeveElement> _access_element;

  ElementAccessGuard(std::shared_ptr<SleeveElement> element);
  ~ElementAccessGuard();
};

class SleeveBase {
 private:
  size_t                                                              _size;
  sl_element_id_t                                                     _next_element_id;
  filter_id_t                                                         _next_filter_id;

  std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>> _storage;
  std::unordered_map<uint32_t, std::shared_ptr<FilterCombo>>          _filter_storage;

  DCacheManager                                                       _dentry_cache;
  uint32_t                                                            _dcache_capacity;

  std::wstring                                                        delimiter = L"/";
  std::wregex                                                         re;

  auto GetWriteGuard(const std::shared_ptr<SleeveFolder> parent_folder, const file_name_t &file_name)
      -> std::optional<ElementAccessGuard>;
  auto WriteCopy(std::shared_ptr<SleeveElement> src_element, std::shared_ptr<SleeveFolder> dest_folder)
      -> std::shared_ptr<SleeveElement>;

 public:
  sleeve_id_t                   _sleeve_id;
  std::shared_ptr<SleeveFolder> _root;

  explicit SleeveBase(sleeve_id_t id);

  void InitializeRoot();

  auto GetStorage() -> std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>> &;

  auto AccessElementById(const sl_element_id_t &id) const -> std::optional<std::shared_ptr<SleeveElement>>;
  auto AccessElementByPath(const sl_path_t &path) -> std::optional<std::shared_ptr<SleeveElement>>;

  auto CreateElementToPath(const sl_path_t &path, const file_name_t &file_name, const ElementType &type)
      -> std::optional<std::shared_ptr<SleeveElement>>;

  auto RemoveElementInPath(const sl_path_t &target) -> std::optional<std::shared_ptr<SleeveElement>>;
  auto RemoveElementInPath(const sl_path_t &path, const file_name_t &file_name)
      -> std::optional<std::shared_ptr<SleeveElement>>;

  auto CopyElement(const sl_path_t &src, const sl_path_t &dest) -> std::optional<std::shared_ptr<SleeveElement>>;

  auto MoveElement(const sl_path_t &src, const sl_path_t &dest) -> std::optional<std::shared_ptr<SleeveElement>>;

  auto GetReadGuard(const sl_path_t &target) -> std::optional<ElementAccessGuard>;

  auto GetWriteGuard(const sl_path_t &target) -> std::optional<ElementAccessGuard>;
  auto GetWriteGuard(const sl_path_t &parent_folder_path, const file_name_t &file_name)
      -> std::optional<ElementAccessGuard>;

  void GarbageCollect();

  auto Tree(const sl_path_t &path) -> std::wstring;
  auto TreeBFS(const sl_path_t &path) -> std::wstring;

  auto IsSubFolder(const std::shared_ptr<SleeveFolder> folder_a, const sl_path_t &path_b) const -> bool;
};
};  // namespace puerhlab