/*
 * @file        pu-erh_lab/src/include/sleeve/sleeve_element/sleeve_element.hpp
 * @brief       The base class for two types of element in the sleeve
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
#include <ctime>
#include <memory>
#include <optional>
#include <string>

#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "type/type.hpp"

namespace puerhlab {
enum class ElementType { FILE, FOLDER };

enum class SyncFlag { UNSYNC, MODIFIED, SYNCED };
/**
 * @brief Abstract objects residing in a sleeve, it can be files or folders
 *
 */
class SleeveElement {
 public:
  sl_element_id_t _element_id;
  ElementType     _type;

  file_name_t     _element_name;

  std::time_t     _added_time;
  std::time_t     _last_modified_time;

  uint32_t        _ref_count;
  bool            _pinned;

  SyncFlag        _sync_flag = SyncFlag::UNSYNC;

  explicit SleeveElement(sl_element_id_t id, file_name_t element_name);

  virtual ~SleeveElement();

  virtual auto Copy(sl_element_id_t new_id) const -> std::shared_ptr<SleeveElement>;
  virtual auto Clear() -> bool;
  void         SetAddTime();
  void         SetLastModifiedTime();
  void         IncrementRefCount();
  void         DecrementRefCount();
  auto         IsShared() -> bool;
  void         SetSyncFlag(SyncFlag flag);

  // virtual void AddElement(std::shared_ptr<SleeveElement>);
  // virtual void CreateFilter(FilterCombo&& filter);
  // virtual auto GetElementByName(file_name_t name) -> std::optional<sl_element_id_t>;
  // virtual auto ListElements() -> std::vector<sl_element_id_t>;
  // virtual auto RecursiveListElements() -> std::vector<sl_element_id_t>;
  // virtual auto RemoveElementByName(file_name_t name) -> sl_element_id_t;
};
};  // namespace puerhlab
