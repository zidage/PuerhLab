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

#include "sleeve/sleeve_element/sleeve_folder.hpp"

#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_element_factory.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief Construct a new Sleeve Folder:: Sleeve Folder object
 *
 * @param id
 * @param element_name
 */
SleeveFolder::SleeveFolder(sl_element_id_t id, file_name_t element_name)
    : SleeveElement(id, element_name), file_count_(0), folder_count_(0) {
  indicies_cache_[default_filter_] = {};
  type_                            = ElementType::FOLDER;
}

SleeveFolder::~SleeveFolder() {}

auto SleeveFolder::Copy(sl_element_id_t new_id) const -> std::shared_ptr<SleeveElement> {
  auto copy       = std::make_shared<SleeveFolder>(new_id, element_name_);
  // Copy the name-id map and filter map
  copy->contents_ = {contents_};
  // Only copy indicies cache under default filter
  copy->indicies_cache_[copy->default_filter_] =
      std::vector<sl_element_id_t>(indicies_cache_.at(default_filter_));
  return copy;
}
/**
 * @brief Add an element reference to the folder
 *
 * @param element
 */
void SleeveFolder::AddElementToMap(const std::shared_ptr<SleeveElement> element) {
  contents_[element->element_name_] = element->element_id_;
  indicies_cache_[default_filter_].push_back(element->element_id_);
  // Once a pinned element is added to the current folder, current folder also becomes pinned
  pinned_ |= element->pinned_;
  element->IncrementRefCount();
  // Mark this folder as modified so the updated content list is persisted on next sync.
  if (sync_flag_ == SyncFlag::SYNCED) {
    sync_flag_ = SyncFlag::MODIFIED;
  }
}

/**
 * @brief Update a name-id mapping
 *
 * @param name
 * @param old_id
 * @param new_id
 */
void SleeveFolder::UpdateElementMap(const file_name_t& name, const sl_element_id_t old_id,
                                    const sl_element_id_t new_id) {
  contents_.erase(name);
  contents_[name]     = new_id;
  auto default_filter = indicies_cache_[default_filter_];

  for (auto& id : default_filter) {
    if (id == old_id) {
      id = new_id;
      break;
    }
  }
  // Mark this folder as modified so the updated content list is persisted on next sync.
  if (sync_flag_ == SyncFlag::SYNCED) {
    sync_flag_ = SyncFlag::MODIFIED;
  }
}

/**
 * @brief Get an element's id from the _contents table
 *
 * @param name
 * @return std::optional<sl_element_id_t>
 */
auto SleeveFolder::GetElementIdByName(const file_name_t& name) const
    -> std::optional<sl_element_id_t> {
  if (!Contains(name)) {
    return std::nullopt;
  }
  return contents_.at(name);
}

/**
 * @brief List all the elements within this folder
 *
 * @return std::shared_ptr<std::vector<sl_element_id_t>>
 */
auto SleeveFolder::ListElements() const -> const std::vector<sl_element_id_t>& {
  const auto& default_list = indicies_cache_.at(default_filter_);
  return default_list;
}

auto SleeveFolder::Clear() -> bool {
  // TODO: Add Implementation
  contents_.clear();
  indicies_cache_.clear();

  return true;
}

/**
 * @brief Check whether the folder contains the element of the given name
 *
 * @param name
 * @return true
 * @return false
 */
auto SleeveFolder::Contains(const file_name_t& name) const -> bool {
  return contents_.count(name) != 0;
}

/**
 * @brief Remove a name-id mapping from the _contents table
 *
 * @param name
 * @return sl_element_id_t
 */
void SleeveFolder::RemoveNameFromMap(const file_name_t& name) {
  auto  removed_id    = contents_.at(name);
  // Also remove from default filter index
  auto& default_index = indicies_cache_[default_filter_];
  default_index.erase(std::remove(default_index.begin(), default_index.end(), removed_id),
                      default_index.end());
  contents_.erase(name);
  // Mark this folder as modified so the updated content list is persisted on next sync.
  if (sync_flag_ == SyncFlag::SYNCED) {
    sync_flag_ = SyncFlag::MODIFIED;
  }
}

void SleeveFolder::CreateIndex(const std::vector<std::shared_ptr<SleeveElement>>& filtered_elements,
                               const filter_id_t                                  filter_id) {
  std::vector<sl_element_id_t> new_index;
  for (const auto& element : filtered_elements) {
    new_index.push_back(element->element_id_);
  }
  indicies_cache_[filter_id] = new_index;
}

auto SleeveFolder::ListElementsByFilter(const filter_id_t filter_id) const
    -> const std::vector<sl_element_id_t>& {
  if (!HasFilterIndex(filter_id)) {
    throw std::runtime_error("Filter index not found in folder.");
  }
  return indicies_cache_.at(filter_id);
}

void SleeveFolder::IncrementFileCount() { ++file_count_; }

void SleeveFolder::DecrementFileCount() { --file_count_; }

void SleeveFolder::IncrementFolderCount() { ++folder_count_; }

void SleeveFolder::DecrementFolderCount() { --folder_count_; }  
auto SleeveFolder::ContentSize() -> size_t { return contents_.size(); }
};  // namespace puerhlab