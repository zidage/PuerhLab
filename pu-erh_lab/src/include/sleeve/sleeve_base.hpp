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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"

namespace puerhlab {
class SleeveBase {
 private:
  sleeve_id_t                                                         _sleeve_id;
  std::unique_ptr<SleeveFolder>                                       _root;
  size_t                                                              _size;
  uint32_t                                                            _next_element_id;

  std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>> _storage;

 public:
  explicit SleeveBase(sleeve_id_t id);

  void InitializeRoot();

  auto AccessElementById(sl_element_id_t id) -> std::shared_ptr<SleeveElement>;

  auto AccessElementByPath(sl_path_t path) -> std::shared_ptr<SleeveFolder>;

  auto CreateFolderToPath(sl_path_t path, file_name_t file_name) -> bool;

  auto CreateFileToPath(sl_path_t path, file_name_t file_name) -> bool;

  auto RemoveElementInPath(sl_path_t full_path);
};
};  // namespace puerhlab