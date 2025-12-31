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
