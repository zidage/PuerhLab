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

#include "sleeve/sleeve_base.hpp"

#include <cstddef>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <unordered_set>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_element_factory.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "type/type.hpp"

namespace puerhlab {

ElementAccessGuard::ElementAccessGuard(std::shared_ptr<SleeveElement> element) {
  access_element_          = element;
  access_element_->pinned_ = true;
}

ElementAccessGuard::~ElementAccessGuard() {
  access_element_->pinned_ = false;
  access_element_.reset();
}

/**
 * @brief Construct a new Sleeve Base:: Sleeve Base object
 *
 * @param id
 */
SleeveBase::SleeveBase(sleeve_id_t id) : re_(delimiter_), sleeve_id_(id) {
  // TODO: change the id assignment logic
  next_element_id_   = 0;
  size_              = 0;
  filter_storage_[0] = std::make_shared<FilterCombo>();
  next_filter_id_    = 0;
  dcache_capacity_   = DCacheManager::default_capacity_;
  InitializeRoot();
}

/**
 * @brief Initialize the root folder within the base
 *
 */
void SleeveBase::InitializeRoot() {
  // FIXME: Inconsistent reinitalization logic here
  if (root_ == nullptr) size_ += 1;
  root_ = std::make_shared<SleeveFolder>(next_element_id_++, L"root");
  root_->IncrementRefCount();
  storage_[root_->element_id_] = root_;
}

auto SleeveBase::GetStorage()
    -> std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& {
  return storage_;
}

auto SleeveBase::GetFilterStorage()
    -> std::unordered_map<filter_id_t, std::shared_ptr<FilterCombo>>& {
  return filter_storage_;
}

/**
 * @brief Access an element by its id from the map
 *
 * @param id
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::AccessElementById(const sl_element_id_t& id) const
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto target = storage_.find(id);
  if (target == storage_.end()) {
    // TODO: Log
    return std::nullopt;
  }

  return (*target).second;
}

/**
 * @brief Access an element by its path
 *
 * @param path
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::AccessElementByPath(const sl_path_t& path)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto result = dentry_cache_.AccessElement(path);
  if (result.has_value()) {
    return storage_.at(result.value());
  }

  std::wsregex_token_iterator first{path.begin(), path.end(), re_, -1}, last;
  std::deque<std::wstring>    path_elements{first, last};

  auto                        curr_path = path_elements.front();
  // For illegal paths, return false
  if (curr_path != L"root") {
    return std::nullopt;
  }
  path_elements.pop_front();
  // Assume all the path start with "root"
  auto curr_element = std::optional<std::shared_ptr<SleeveElement>>(root_);

  while (curr_element != std::nullopt && !path_elements.empty()) {
    if (curr_element.value()->type_ == ElementType::FILE) {
      // TODO: Log
      // the path of a file cannot be a prefix of a full path
      return std::nullopt;
    }

    curr_path = path_elements.front();
    path_elements.pop_front();
    auto next_element_id = std::dynamic_pointer_cast<SleeveFolder>(curr_element.value())
                               ->GetElementIdByName(curr_path);
    if (!next_element_id.has_value()) {
      // TODO: Log
      return std::nullopt;
    }
    curr_element = AccessElementById(next_element_id.value());
  }

  if (curr_element.has_value()) dentry_cache_.RecordAccess(path, curr_element.value()->element_id_);
  return curr_element;
}

/**
 * @brief
 *
 * @param path
 * @param file_name
 * @param type
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::CreateElementToPath(const sl_path_t& path, const file_name_t& file_name,
                                     const ElementType& type)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto parent_folder_opt = GetWriteGuard(path);
  if (!parent_folder_opt.has_value() ||
      parent_folder_opt.value().access_element_->type_ == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }
  auto parent_folder =
      std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value().access_element_);
  if (parent_folder->Contains(file_name)) {
    // TODO: Log
    // TODO: Add duplication handling here
    return std::nullopt;
  }
  std::shared_ptr<SleeveElement> newElement =
      SleeveElementFactory::CreateElement(type, next_element_id_++, file_name);
  // Update the map
  storage_[newElement->element_id_] = newElement;
  parent_folder->AddElementToMap(newElement);
  // Increment the reference count of the file
  if (newElement->type_ == ElementType::FILE) {
    parent_folder->IncrementFileCount();
  } else {
    parent_folder->IncrementFolderCount();
  }
  newElement->IncrementRefCount();
  parent_folder->SetLastModifiedTime();
  return newElement;
}

/**
 * @brief Remove an element by its full path
 *
 * @param target
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::RemoveElementInPath(const sl_path_t& target)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  if (target == L"root") {
    // TODO: Log
    return std::nullopt;
  }
  size_t pos = target.rfind(delimiter_);
  return RemoveElementInPath(target.substr(0, pos), target.substr(pos + 1));
}

/**
 * @brief Remove an element by its parent folder path
 *
 * @param full_path
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::RemoveElementInPath(const sl_path_t& path, const file_name_t& file_name)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto parent_folder_opt = GetWriteGuard(path);
  if (!parent_folder_opt.has_value() ||
      parent_folder_opt.value().access_element_->type_ == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }
  auto parent_folder =
      std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value().access_element_);
  auto del_id = parent_folder->GetElementIdByName(file_name);
  if (!del_id.has_value()) {
    return std::nullopt;
  }
  auto del_element = AccessElementById(del_id.value()).value();
  if (del_element->pinned_) {
    return std::nullopt;
  }

  if (del_element->type_ == ElementType::FOLDER) {
    auto  del_folder = std::dynamic_pointer_cast<SleeveFolder>(del_element);
    auto& elements   = del_folder->ListElements();
    for (auto& element_id : elements) {
      auto& e = storage_.at(element_id);
      e->DecrementRefCount();
    }

    parent_folder->DecrementFolderCount();
  } else {
    // TODO: Add remove code for file elements
    parent_folder->DecrementFileCount();
  }
  parent_folder->RemoveNameFromMap(del_element->element_name_);
  parent_folder->SetLastModifiedTime();
  if (del_element->ref_count_ == 1)
    // clear all the metadata and actual data in the target element
    del_element->Clear();
  del_element->DecrementRefCount();

  if (del_element->type_ == ElementType::FOLDER)
    dentry_cache_.Flush();
  else
    dentry_cache_.RemoveRecord(path + L"/" + file_name);

  return del_element;
}

/**
 * @brief Acquire the read guard of a element
 *
 * @param target
 * @return std::optional<ElementAccessGuard>
 */
auto SleeveBase::GetReadGuard(const sl_path_t& target) -> std::optional<ElementAccessGuard> {
  auto read_element = AccessElementByPath(target);
  if (!read_element.has_value()) {
    // TODO: Log
    return std::nullopt;
  }
  return read_element.value();
}

auto inline SleeveBase::WriteCopy(std::shared_ptr<SleeveElement> src_element,
                                  std::shared_ptr<SleeveFolder>  dest_folder)
    -> std::shared_ptr<SleeveElement> {
  src_element->DecrementRefCount();
  auto copy                   = src_element->Copy(next_element_id_++);
  storage_[copy->element_id_] = copy;
  // Parent folder now contains a "true" copy of the current folder
  dest_folder->UpdateElementMap(src_element->element_name_, src_element->element_id_,
                                copy->element_id_);
  copy->IncrementRefCount();
  return copy;
}

/**
 * @brief Acquire the write guard of an element by its full path
 *
 * @param target
 * @return std::optional<ElementAccessGuard>
 */
auto SleeveBase::GetWriteGuard(const sl_path_t& target) -> std::optional<ElementAccessGuard> {
  // Copy on write, make a copy to the current folder
  if (target == L"root") {
    // TODO: Log
    return std::static_pointer_cast<SleeveElement>(root_);
  }
  size_t pos = target.rfind(delimiter_);

  return GetWriteGuard(target.substr(0, pos), target.substr(pos + 1));
}

/**
 * @brief Acquire the write guard of an element by its name and parent folder's path
 *
 * @param path
 * @param file_name
 * @return std::optional<ElementAccessGuard>
 */
auto SleeveBase::GetWriteGuard(const sl_path_t& parent_folder_path, const file_name_t& file_name)
    -> std::optional<ElementAccessGuard> {
  std::wsregex_token_iterator first{parent_folder_path.begin(), parent_folder_path.end(), re_, -1},
      last;
  std::deque<std::wstring> path_elements{first, last};

  auto                     curr_path = path_elements.front();
  // For illegal paths, return false
  if (curr_path != L"root") {
    return std::nullopt;
  }
  path_elements.pop_front();
  // Assume all the path start with "root"
  auto curr_element_opt = std::optional<std::shared_ptr<SleeveElement>>(root_);
  std::shared_ptr<SleeveFolder> curr_parent_folder =
      std::dynamic_pointer_cast<SleeveFolder>(curr_element_opt.value());
  std::optional<sl_element_id_t> next_element_id = 0;

  sl_path_t                      acc_path        = curr_path;

  while (curr_element_opt != std::nullopt && !path_elements.empty()) {
    if (curr_element_opt.value()->type_ == ElementType::FILE) {
      // TODO: Log
      // the path of a file cannot be a prefix of a full path
      return std::nullopt;
    }
    auto curr_element = std::dynamic_pointer_cast<SleeveFolder>(curr_element_opt.value());
    // Current PARENT folder will only have reference count equal to 1
    // If current FOLDER has ref count greater than one, then we have to create a new copy
    // of it, and make the PARENT folder reference to the new copy.
    if (curr_element->ref_count_ > 1) {
      auto copy = WriteCopy(curr_element, curr_parent_folder);
      dentry_cache_.RecordAccess(acc_path, copy->element_id_);
      curr_path = path_elements.front();
      acc_path += L"/" + curr_path;
      path_elements.pop_front();
      curr_parent_folder = std::dynamic_pointer_cast<SleeveFolder>(copy);
      next_element_id    = curr_parent_folder->GetElementIdByName(curr_path);
      if (!next_element_id.has_value()) {
        // TODO: Log
        return std::nullopt;
      }
      curr_element_opt = AccessElementById(next_element_id.value());
      continue;
    }

    // Step to the next element along the path
    curr_path = path_elements.front();
    acc_path += L"/" + curr_path;
    path_elements.pop_front();
    curr_parent_folder = curr_element;
    next_element_id    = curr_parent_folder->GetElementIdByName(curr_path);
    if (!next_element_id.has_value()) {
      // TODO: Log
      return std::nullopt;
    }
    curr_element_opt = AccessElementById(next_element_id.value());
  }
  // Check the last element along the path
  if (curr_element_opt == std::nullopt || curr_element_opt.value()->type_ == ElementType::FILE) {
    // TODO: Log
    // the path of a file cannot be a prefix of a full path
    return std::nullopt;
  }
  auto parent_folder = std::dynamic_pointer_cast<SleeveFolder>(curr_element_opt.value());
  if (parent_folder->ref_count_ > 1) {
    auto copy     = WriteCopy(parent_folder, curr_parent_folder);
    parent_folder = std::dynamic_pointer_cast<SleeveFolder>(copy);
  }

  auto write_file_opt = parent_folder->GetElementIdByName(file_name);
  if (!write_file_opt.has_value()) {
    // TODO: Log
    return std::nullopt;
  }
  auto write_file = storage_.at(write_file_opt.value());
  if (write_file->ref_count_ > 1) {
    write_file = WriteCopy(write_file, parent_folder);
  }
  dentry_cache_.RecordAccess(parent_folder_path + delimiter_ + file_name, write_file->element_id_);
  return write_file;
}

/**
 * @brief Check if folder of dest_folder_path is a subfolder of src_folder. It is a helper function
 * without any sanity check for the type of the elements of path_a and dest_folder_path
 *
 * @param path_a
 * @param dest_folder_path
 * @return true
 * @return false
 */
auto SleeveBase::IsSubFolder(const std::shared_ptr<SleeveFolder> src_folder,
                             const sl_path_t&                    dest_folder_path) const -> bool {
  std::wsregex_token_iterator first{dest_folder_path.begin(), dest_folder_path.end(), re_, -1}, last;
  std::deque<std::wstring>    path_elements{first, last};

  auto                        curr_path = path_elements.front();
  path_elements.pop_front();
  // Assume all the path start with "root"
  auto curr_element = std::optional<std::shared_ptr<SleeveElement>>(root_);

  while (curr_element != std::nullopt && !path_elements.empty()) {
    curr_path = path_elements.front();
    path_elements.pop_front();
    auto curr_folder = std::dynamic_pointer_cast<SleeveFolder>(curr_element.value());
    if (curr_folder == src_folder) {
      // TODO: LOG
      return true;
    }
    auto next_element_id = curr_folder->GetElementIdByName(curr_path);

    curr_element         = AccessElementById(next_element_id.value());
  }
  if (curr_element != std::nullopt &&
      std::dynamic_pointer_cast<SleeveFolder>(curr_element.value()) == src_folder) {
    return true;
  }
  return false;
}

auto SleeveBase::CopyElement(const sl_path_t& src, const sl_path_t& dest)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  if (src == L"root") {
    return std::nullopt;
  }
  auto src_read_guard = GetReadGuard(src);
  if (!src_read_guard.has_value()) {
    // TODO: Log
    return std::nullopt;
  }
  auto src_file        = src_read_guard->access_element_;

  auto dest_folder_opt = GetReadGuard(dest);
  if (!dest_folder_opt.has_value() ||
      dest_folder_opt.value().access_element_->type_ == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }

  auto dest_folder =
      std::dynamic_pointer_cast<SleeveFolder>(dest_folder_opt.value().access_element_);

  if (dest_folder->Contains(src_file->element_name_)) {
    // TODO: Log
    return std::nullopt;
  }
  // if source is a folder, target folder should not be a subfolder of the source
  if (src_file->type_ == ElementType::FOLDER &&
      IsSubFolder(std::dynamic_pointer_cast<SleeveFolder>(src_file), dest)) {
    // TODO: Log
    return std::nullopt;
  }

  dest_folder_opt = GetWriteGuard(dest);
  if (dest_folder_opt == std::nullopt) {
    return std::nullopt;
  }
  dest_folder = std::dynamic_pointer_cast<SleeveFolder>(dest_folder_opt.value().access_element_);
  if (src_file->type_ == ElementType::FOLDER) {
    auto  src_folder = std::dynamic_pointer_cast<SleeveFolder>(src_file);
    // Increment the reference count for all of its contents
    auto& elements   = src_folder->ListElements();
    for (auto& e : elements) {
      storage_.at(e)->IncrementRefCount();
    }
    dest_folder->IncrementFolderCount();
  } else {
    dest_folder->IncrementFileCount();
  }

  // For files, only the reference is copied
  dest_folder->AddElementToMap(src_file);

  src_file->IncrementRefCount();
  return src_file;
}

/**
 * @brief Move element from one path to another
 *
 * @param src
 * @param dest
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::MoveElement(const sl_path_t& src, const sl_path_t& dest)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto result = CopyElement(src, dest);
  if (!result.has_value()) {
    return std::nullopt;
  }
  RemoveElementInPath(src);
  return result;
}

/**
 * @brief Print the whole file structure under a given folder
 * DEBUG PURPOSE ONLY
 *
 */
auto SleeveBase::Tree(const sl_path_t& path) -> std::wstring {
  struct TreeNode {
    sl_element_id_t id_;
    int             depth_;
    bool            is_last_;
  };
  std::wstring prefix          = L"";
  auto         dest_folder_opt = GetReadGuard(path);
  if (!dest_folder_opt.has_value() ||
      dest_folder_opt->access_element_->type_ == ElementType::FILE) {
    std::wcout << L"Cannot call Tree() on an file" << std::endl;
    return L"";
  }
  std::wstring result = L"";
  auto         visit_folder =
      std::dynamic_pointer_cast<SleeveFolder>(dest_folder_opt.value().access_element_);
  auto contains  = visit_folder->ListElements();
  auto dfs_stack = std::stack<TreeNode>();
  for (auto e : contains) {
    dfs_stack.push({e, 0, storage_.at(e)->type_ == ElementType::FILE});
  }
  result +=
      visit_folder->element_name_ + L" id:" + std::to_wstring(visit_folder->element_id_) + L"\n";

  std::shared_ptr<SleeveElement> parent_node = nullptr;

  while (!dfs_stack.empty()) {
    auto next_visit = dfs_stack.top();
    dfs_stack.pop();
    auto next_visit_element = storage_.at(next_visit.id_);

    if (next_visit_element->type_ == ElementType::FOLDER) {
      for (int i = 0; i < next_visit.depth_; i++) {
        result += L"    ";
      }
      result += L"├── " + next_visit_element->element_name_ + L" id:" +
                std::to_wstring(next_visit_element->element_id_) + L"\n";
      auto sub_folder = std::dynamic_pointer_cast<SleeveFolder>(next_visit_element);
      contains        = sub_folder->ListElements();
      for (auto e : contains) {
        dfs_stack.push({e, next_visit.depth_ + 1, storage_.at(e)->type_ == ElementType::FILE});
      }
    } else {
      for (int i = 0; i < next_visit.depth_; i++) {
        result += L"    ";
      }
      result += L"└── " + next_visit_element->element_name_ + L" id:" +
                std::to_wstring(next_visit_element->element_id_) + L"\n";
    }
  }
  return result;
}

/**
 * @brief Print the whole file structure under a given folder
 * DEBUG PURPOSE ONLY
 *
 */
auto SleeveBase::TreeBFS(const sl_path_t& path) -> std::wstring {
  struct TreeNode {
    sl_element_id_t id_;
    int             depth_;
    bool            is_last_;
  };
  auto dest_folder_opt = GetReadGuard(path);
  if (!dest_folder_opt.has_value() ||
      dest_folder_opt->access_element_->type_ == ElementType::FILE) {
    std::wcout << L"Cannot call Tree() on an file" << std::endl;
    return L"";
  }
  std::wstring result = L"";
  auto         visit_folder =
      std::dynamic_pointer_cast<SleeveFolder>(dest_folder_opt.value().access_element_);
  auto contains  = visit_folder->ListElements();
  auto bfs_queue = std::deque<TreeNode>();
  for (auto e : contains) {
    bfs_queue.push_back({e, 1, storage_.at(e)->type_ == ElementType::FILE});
  }
  result +=
      visit_folder->element_name_ + L" id:" + std::to_wstring(visit_folder->element_id_) + L"\n";

  std::shared_ptr<SleeveElement> parent_node = nullptr;
  int                            last_depth  = 0;
  while (!bfs_queue.empty()) {
    auto next_visit = bfs_queue.front();
    bfs_queue.pop_front();
    auto next_visit_element = storage_.at(next_visit.id_);
    if (next_visit.depth_ != last_depth) {
      result += L"\nLayer: " + std::to_wstring(next_visit.depth_) + L" ";
      last_depth = next_visit.depth_;
    }
    if (next_visit_element->type_ == ElementType::FOLDER) {
      result += L"FOLDER: " + next_visit_element->element_name_ + L" id:" +
                std::to_wstring(next_visit_element->element_id_) + L" ";
      auto sub_folder = std::dynamic_pointer_cast<SleeveFolder>(next_visit_element);
      contains        = sub_folder->ListElements();
      for (auto e : contains) {
        bfs_queue.push_back({e, next_visit.depth_ + 1, storage_.at(e)->type_ == ElementType::FILE});
      }
    } else {
      result += L"FILE: " + next_visit_element->element_name_ + L" id:" +
                std::to_wstring(next_visit_element->element_id_) + L" ";
    }
  }
  return result;
}
};  // namespace puerhlab