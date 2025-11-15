/*
 * @file        pu-erh_lab/src/sleeve/sleeve_base.hpp
 * @brief       A file-system-like interface used with DuckDB to store images
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
  _access_element          = element;
  _access_element->_pinned = true;
}

ElementAccessGuard::~ElementAccessGuard() {
  _access_element->_pinned = false;
  _access_element.reset();
}

/**
 * @brief Construct a new Sleeve Base:: Sleeve Base object
 *
 * @param id
 */
SleeveBase::SleeveBase(sleeve_id_t id) : re(delimiter), _sleeve_id(id) {
  // TODO: change the id assignment logic
  _next_element_id   = 0;
  _size              = 0;
  _filter_storage[0] = std::make_shared<FilterCombo>();
  _next_filter_id    = 0;
  _dcache_capacity   = DCacheManager::_default_capacity;
  InitializeRoot();
}

/**
 * @brief Initialize the root folder within the base
 *
 */
void SleeveBase::InitializeRoot() {
  // FIXME: Inconsistent reinitalization logic here
  if (_root == nullptr) _size += 1;
  _root = std::make_shared<SleeveFolder>(_next_element_id++, L"root");
  _root->IncrementRefCount();
  _storage[_root->_element_id] = _root;
}

auto SleeveBase::GetStorage()
    -> std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& {
  return _storage;
}

auto SleeveBase::GetFilterStorage()
    -> std::unordered_map<filter_id_t, std::shared_ptr<FilterCombo>>& {
  return _filter_storage;
}

/**
 * @brief Access an element by its id from the map
 *
 * @param id
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::AccessElementById(const sl_element_id_t& id) const
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto target = _storage.find(id);
  if (target == _storage.end()) {
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
  auto result = _dentry_cache.AccessElement(path);
  if (result.has_value()) {
    return _storage.at(result.value());
  }

  std::wsregex_token_iterator first{path.begin(), path.end(), re, -1}, last;
  std::deque<std::wstring>    path_elements{first, last};

  auto                        curr_path = path_elements.front();
  // For illegal paths, return false
  if (curr_path != L"root") {
    return std::nullopt;
  }
  path_elements.pop_front();
  // Assume all the path start with "root"
  auto curr_element = std::optional<std::shared_ptr<SleeveElement>>(_root);

  while (curr_element != std::nullopt && !path_elements.empty()) {
    if (curr_element.value()->_type == ElementType::FILE) {
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

  if (curr_element.has_value()) _dentry_cache.RecordAccess(path, curr_element.value()->_element_id);
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
      parent_folder_opt.value()._access_element->_type == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }
  auto parent_folder =
      std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value()._access_element);
  if (parent_folder->Contains(file_name)) {
    // TODO: Log
    // TODO: Add duplication handling here
    return std::nullopt;
  }
  std::shared_ptr<SleeveElement> newElement =
      SleeveElementFactory::CreateElement(type, _next_element_id++, file_name);
  // Update the map
  _storage[newElement->_element_id] = newElement;
  parent_folder->AddElementToMap(newElement);
  // Increment the reference count of the file
  if (newElement->_type == ElementType::FILE) {
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
  size_t pos = target.rfind(delimiter);
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
      parent_folder_opt.value()._access_element->_type == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }
  auto parent_folder =
      std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value()._access_element);
  auto del_id = parent_folder->GetElementIdByName(file_name);
  if (!del_id.has_value()) {
    return std::nullopt;
  }
  auto del_element = AccessElementById(del_id.value()).value();
  if (del_element->_pinned) {
    return std::nullopt;
  }

  if (del_element->_type == ElementType::FOLDER) {
    auto del_folder = std::dynamic_pointer_cast<SleeveFolder>(del_element);
    auto& elements   = del_folder->ListElements();
    for (auto& element_id : elements) {
      auto& e = _storage.at(element_id);
      e->DecrementRefCount();
    }

    parent_folder->DecrementFolderCount();
  } else {
    // TODO: Add remove code for file elements
    parent_folder->DecrementFileCount();
  }
  parent_folder->RemoveNameFromMap(del_element->_element_name);
  parent_folder->SetLastModifiedTime();
  if (del_element->_ref_count == 1)
    // clear all the metadata and actual data in the target element
    del_element->Clear();
  del_element->DecrementRefCount();

  if (del_element->_type == ElementType::FOLDER)
    _dentry_cache.Flush();
  else
    _dentry_cache.RemoveRecord(path + L"/" + file_name);

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
  auto copy                   = src_element->Copy(_next_element_id++);
  _storage[copy->_element_id] = copy;
  // Parent folder now contains a "true" copy of the current folder
  dest_folder->UpdateElementMap(src_element->_element_name, src_element->_element_id,
                                copy->_element_id);
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
    return std::static_pointer_cast<SleeveElement>(_root);
  }
  size_t pos = target.rfind(delimiter);

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
  std::wsregex_token_iterator first{parent_folder_path.begin(), parent_folder_path.end(), re, -1},
      last;
  std::deque<std::wstring> path_elements{first, last};

  auto                     curr_path = path_elements.front();
  // For illegal paths, return false
  if (curr_path != L"root") {
    return std::nullopt;
  }
  path_elements.pop_front();
  // Assume all the path start with "root"
  auto curr_element_opt = std::optional<std::shared_ptr<SleeveElement>>(_root);
  std::shared_ptr<SleeveFolder> curr_parent_folder =
      std::dynamic_pointer_cast<SleeveFolder>(curr_element_opt.value());
  std::optional<sl_element_id_t> next_element_id = 0;

  sl_path_t                      acc_path        = curr_path;

  while (curr_element_opt != std::nullopt && !path_elements.empty()) {
    if (curr_element_opt.value()->_type == ElementType::FILE) {
      // TODO: Log
      // the path of a file cannot be a prefix of a full path
      return std::nullopt;
    }
    auto curr_element = std::dynamic_pointer_cast<SleeveFolder>(curr_element_opt.value());
    // Current PARENT folder will only have reference count equal to 1
    // If current FOLDER has ref count greater than one, then we have to create a new copy
    // of it, and make the PARENT folder reference to the new copy.
    if (curr_element->_ref_count > 1) {
      auto copy = WriteCopy(curr_element, curr_parent_folder);
      _dentry_cache.RecordAccess(acc_path, copy->_element_id);
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
  if (curr_element_opt == std::nullopt || curr_element_opt.value()->_type == ElementType::FILE) {
    // TODO: Log
    // the path of a file cannot be a prefix of a full path
    return std::nullopt;
  }
  auto parent_folder = std::dynamic_pointer_cast<SleeveFolder>(curr_element_opt.value());
  if (parent_folder->_ref_count > 1) {
    auto copy     = WriteCopy(parent_folder, curr_parent_folder);
    parent_folder = std::dynamic_pointer_cast<SleeveFolder>(copy);
  }

  auto write_file_opt = parent_folder->GetElementIdByName(file_name);
  if (!write_file_opt.has_value()) {
    // TODO: Log
    return std::nullopt;
  }
  auto write_file = _storage.at(write_file_opt.value());
  if (write_file->_ref_count > 1) {
    write_file = WriteCopy(write_file, parent_folder);
  }
  _dentry_cache.RecordAccess(parent_folder_path + delimiter + file_name, write_file->_element_id);
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
  std::wsregex_token_iterator first{dest_folder_path.begin(), dest_folder_path.end(), re, -1}, last;
  std::deque<std::wstring>    path_elements{first, last};

  auto                        curr_path = path_elements.front();
  path_elements.pop_front();
  // Assume all the path start with "root"
  auto curr_element = std::optional<std::shared_ptr<SleeveElement>>(_root);

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
  auto src_file        = src_read_guard->_access_element;

  auto dest_folder_opt = GetReadGuard(dest);
  if (!dest_folder_opt.has_value() ||
      dest_folder_opt.value()._access_element->_type == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }

  auto dest_folder =
      std::dynamic_pointer_cast<SleeveFolder>(dest_folder_opt.value()._access_element);

  if (dest_folder->Contains(src_file->_element_name)) {
    // TODO: Log
    return std::nullopt;
  }
  // if source is a folder, target folder should not be a subfolder of the source
  if (src_file->_type == ElementType::FOLDER &&
      IsSubFolder(std::dynamic_pointer_cast<SleeveFolder>(src_file), dest)) {
    // TODO: Log
    return std::nullopt;
  }

  dest_folder_opt = GetWriteGuard(dest);
  if (dest_folder_opt == std::nullopt) {
    return std::nullopt;
  }
  dest_folder = std::dynamic_pointer_cast<SleeveFolder>(dest_folder_opt.value()._access_element);
  if (src_file->_type == ElementType::FOLDER) {
    auto src_folder = std::dynamic_pointer_cast<SleeveFolder>(src_file);
    // Increment the reference count for all of its contents
    auto& elements   = src_folder->ListElements();
    for (auto& e : elements) {
      _storage.at(e)->IncrementRefCount();
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
    sl_element_id_t id;
    int             depth;
    bool            is_last;
  };
  std::wstring prefix          = L"";
  auto         dest_folder_opt = GetReadGuard(path);
  if (!dest_folder_opt.has_value() ||
      dest_folder_opt->_access_element->_type == ElementType::FILE) {
    std::wcout << L"Cannot call Tree() on an file" << std::endl;
    return L"";
  }
  std::wstring result = L"";
  auto         visit_folder =
      std::dynamic_pointer_cast<SleeveFolder>(dest_folder_opt.value()._access_element);
  auto contains  = visit_folder->ListElements();
  auto dfs_stack = std::stack<TreeNode>();
  for (auto e : contains) {
    dfs_stack.push({e, 0, _storage.at(e)->_type == ElementType::FILE});
  }
  result +=
      visit_folder->_element_name + L" id:" + std::to_wstring(visit_folder->_element_id) + L"\n";

  std::shared_ptr<SleeveElement> parent_node = nullptr;

  while (!dfs_stack.empty()) {
    auto next_visit = dfs_stack.top();
    dfs_stack.pop();
    auto next_visit_element = _storage.at(next_visit.id);

    if (next_visit_element->_type == ElementType::FOLDER) {
      for (int i = 0; i < next_visit.depth; i++) {
        result += L"    ";
      }
      result += L"├── " + next_visit_element->_element_name + L" id:" +
                std::to_wstring(next_visit_element->_element_id) + L"\n";
      auto sub_folder = std::dynamic_pointer_cast<SleeveFolder>(next_visit_element);
      contains        = sub_folder->ListElements();
      for (auto e : contains) {
        dfs_stack.push({e, next_visit.depth + 1, _storage.at(e)->_type == ElementType::FILE});
      }
    } else {
      for (int i = 0; i < next_visit.depth; i++) {
        result += L"    ";
      }
      result += L"└── " + next_visit_element->_element_name + L" id:" +
                std::to_wstring(next_visit_element->_element_id) + L"\n";
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
    sl_element_id_t id;
    int             depth;
    bool            is_last;
  };
  auto dest_folder_opt = GetReadGuard(path);
  if (!dest_folder_opt.has_value() ||
      dest_folder_opt->_access_element->_type == ElementType::FILE) {
    std::wcout << L"Cannot call Tree() on an file" << std::endl;
    return L"";
  }
  std::wstring result = L"";
  auto         visit_folder =
      std::dynamic_pointer_cast<SleeveFolder>(dest_folder_opt.value()._access_element);
  auto contains  = visit_folder->ListElements();
  auto bfs_queue = std::deque<TreeNode>();
  for (auto e : contains) {
    bfs_queue.push_back({e, 1, _storage.at(e)->_type == ElementType::FILE});
  }
  result +=
      visit_folder->_element_name + L" id:" + std::to_wstring(visit_folder->_element_id) + L"\n";

  std::shared_ptr<SleeveElement> parent_node = nullptr;
  int                            last_depth  = 0;
  while (!bfs_queue.empty()) {
    auto next_visit = bfs_queue.front();
    bfs_queue.pop_front();
    auto next_visit_element = _storage.at(next_visit.id);
    if (next_visit.depth != last_depth) {
      result += L"\nLayer: " + std::to_wstring(next_visit.depth) + L" ";
      last_depth = next_visit.depth;
    }
    if (next_visit_element->_type == ElementType::FOLDER) {
      result += L"FOLDER: " + next_visit_element->_element_name + L" id:" +
                std::to_wstring(next_visit_element->_element_id) + L" ";
      auto sub_folder = std::dynamic_pointer_cast<SleeveFolder>(next_visit_element);
      contains        = sub_folder->ListElements();
      for (auto e : contains) {
        bfs_queue.push_back({e, next_visit.depth + 1, _storage.at(e)->_type == ElementType::FILE});
      }
    } else {
      result += L"FILE: " + next_visit_element->_element_name + L" id:" +
                std::to_wstring(next_visit_element->_element_id) + L" ";
    }
  }
  return result;
}
};  // namespace puerhlab