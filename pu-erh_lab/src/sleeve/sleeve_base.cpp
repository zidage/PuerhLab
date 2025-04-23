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
#include <memory>
#include <optional>
#include <stack>
#include <string>

#include "mapper/sleeve/sleeve_mapper.hpp"
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
SleeveBase::SleeveBase(sleeve_id_t id) : _sleeve_id(id) {
  _next_element_id   = 0;
  _size              = 0;
  _filter_storage[0] = std::make_shared<FilterCombo>();
  _next_filter_id    = 0;
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

/**
 * @brief Access an element by its id from the map
 *
 * @param id
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::AccessElementById(const sl_element_id_t &id) const -> std::optional<std::shared_ptr<SleeveElement>> {
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
auto SleeveBase::AccessElementByPath(const sl_path_t &path) const -> std::optional<std::shared_ptr<SleeveElement>> {
  std::wregex                 re(delimiter);
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
      // Find a file in path's prefix, which is illegal
      return std::nullopt;
    }

    curr_path = path_elements.front();
    path_elements.pop_front();
    auto next_element_id = std::dynamic_pointer_cast<SleeveFolder>(curr_element.value())->GetElementIdByName(curr_path);
    if (!next_element_id.has_value()) {
      // TODO: Log
      return std::nullopt;
    }
    curr_element = AccessElementById(next_element_id.value());
  }
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
auto SleeveBase::CreateElementToPath(const sl_path_t &path, const file_name_t &file_name, const ElementType &type)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto parent_folder_opt = AccessElementByPath(path);
  if (!parent_folder_opt.has_value() || parent_folder_opt.value()->_type == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }
  auto parent_folder = std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value());
  return CreateElementToPath(parent_folder, file_name, type);
}

/**
 * @brief
 *
 * @param parent_folder
 * @param file_name
 * @param type
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::CreateElementToPath(const std::shared_ptr<SleeveFolder> parent_folder, const file_name_t &file_name,
                                     const ElementType &type) -> std::optional<std::shared_ptr<SleeveElement>> {
  if (parent_folder->Contains(file_name)) {
    // TODO: Log
    // TODO: Add repetition handling here
    return std::nullopt;
  }
  std::shared_ptr<SleeveElement> newElement = SleeveElementFactory::CreateElement(type, _next_element_id++, file_name);
  // Update the map
  _storage[newElement->_element_id]         = newElement;
  parent_folder->AddElementToMap(newElement);
  // Increment the reference count of the file
  if (newElement->_type == ElementType::FILE) {
    parent_folder->IncrementFileCount();
  } else {
    parent_folder->IncrementFolderCount();
  }
  ++_size;
  newElement->IncrementRefCount();
  return newElement;
}

/**
 * @brief Remove an element by its full path
 *
 * @param target
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::RemoveElementInPath(const sl_path_t &target) -> std::optional<std::shared_ptr<SleeveElement>> {
  if (target == L"root") {
    // TODO: Log
    return std::nullopt;
  }
  size_t pos               = target.rfind(delimiter);
  auto   parent_folder_opt = AccessElementByPath(target.substr(0, pos));
  if (!parent_folder_opt.has_value() || parent_folder_opt.value()->_type == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }
  auto parent_folder = std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value());
  return RemoveElementInPath(parent_folder, target.substr(pos + 1));
}

/**
 * @brief Remove an element by its parent folder path
 *
 * @param full_path
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::RemoveElementInPath(const sl_path_t &path, const file_name_t &file_name)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto parent_folder_opt = AccessElementByPath(path);
  if (!parent_folder_opt.has_value() || parent_folder_opt.value()->_type == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }
  auto parent_folder = std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value());
  return RemoveElementInPath(parent_folder, file_name);
}

/**
 * @brief
 *
 * @param parent_folder
 * @param file_name
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::RemoveElementInPath(const std::shared_ptr<SleeveFolder> parent_folder, const file_name_t &file_name)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto del_id = parent_folder->GetElementIdByName(file_name);
  if (!del_id.has_value()) {
    // TODO: Log
    return std::nullopt;
  }
  auto del_element = AccessElementById(del_id.value()).value();
  if (del_element->_pinned) {
    // TODO: Log
    return std::nullopt;
  }

  if (del_element->_type == ElementType::FOLDER) {
    auto del_folder   = std::dynamic_pointer_cast<SleeveFolder>(del_element);
    auto del_elements = del_folder->ListElements();
    auto bfs_queue    = std::deque<sl_element_id_t>(del_elements->begin(), del_elements->end());
    // Use BFS to find the elements REACHABLE from the to-delete folder, and decrement their reference count
    while (!bfs_queue.empty()) {
      auto next_del_id      = bfs_queue.front();
      auto next_del_element = _storage[next_del_id];
      bfs_queue.pop_front();
      // If the folder at the current level contains a subfolder, expand the BFS fringe
      if (next_del_element->_type == ElementType::FOLDER) {
        auto sub_folder = std::dynamic_pointer_cast<SleeveFolder>(next_del_element);
        del_elements    = sub_folder->ListElements();
        bfs_queue.insert(bfs_queue.end(), del_elements->begin(), del_elements->end());
        del_folder->ClearFolder();
      }
      next_del_element->DecrementRefCount();
    }
    del_folder->ClearFolder();
    parent_folder->DecrementFolderCount();
  } else {
    // TODO: Add remove code for file elements
    parent_folder->DecrementFileCount();
  }
  parent_folder->RemoveNameFromMap(del_element->_element_name);
  del_element->DecrementRefCount();

  return del_element;
}

/**
 * @brief Acquire the read guard of a element
 *
 * @param target
 * @return std::optional<ElementAccessGuard>
 */
auto SleeveBase::GetReadGuard(const sl_path_t &target) const -> std::optional<ElementAccessGuard> {
  auto read_element = AccessElementByPath(target);
  if (!read_element.has_value()) {
    // TODO: Log
    return std::nullopt;
  }
  return read_element.value();
}

/**
 * @brief Acquire the write guard of an element by its full path
 *
 * @param target
 * @return std::optional<ElementAccessGuard>
 */
auto SleeveBase::GetWriteGuard(const sl_path_t &target) -> std::optional<ElementAccessGuard> {
  // Copy on write, make a copy to the current folder
  if (target == L"root") {
    // TODO: Log
    return std::nullopt;
  }
  size_t pos               = target.rfind(delimiter);
  auto   parent_folder_opt = AccessElementByPath(target.substr(0, pos));
  if (!parent_folder_opt.has_value() || parent_folder_opt.value()->_type == ElementType::FILE) {
    return std::nullopt;
  }
  auto parent_folder = std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value());

  return GetWriteGuard(parent_folder, target.substr(pos + 1));
}

/**
 * @brief Acquire the write guard of an element by its name and parent folder's path
 *
 * @param path
 * @param file_name
 * @return std::optional<ElementAccessGuard>
 */
auto SleeveBase::GetWriteGuard(const sl_path_t &path, const file_name_t &file_name)
    -> std::optional<ElementAccessGuard> {
  auto parent_folder_opt = AccessElementByPath(path);
  if (!parent_folder_opt.has_value() || parent_folder_opt.value()->_type == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }
  auto parent_folder = std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value());
  return GetWriteGuard(parent_folder, file_name);
}

/**
 * @brief Acquire the write guard of an element by its name and parent folder.
 *
 * @param parent_folder
 * @param file_name
 * @return std::optional<ElementAccessGuard>
 */
auto SleeveBase::GetWriteGuard(const std::shared_ptr<SleeveFolder> parent_folder, const file_name_t &file_name)
    -> std::optional<ElementAccessGuard> {
  auto write_file_opt = parent_folder->GetElementIdByName(file_name);
  if (!write_file_opt.has_value()) {
    // TODO: Log
    return std::nullopt;
  }
  auto write_file = _storage.at(write_file_opt.value());
  // For simplicity, only files will have reference count greater than 1
  if (write_file->_ref_count > 1) {
    write_file->DecrementRefCount();
    auto copy = write_file->Copy(_next_element_id++);
    _storage.erase(write_file->_element_id);
    _storage[copy->_element_id] = copy;
    parent_folder->UpdateElementMap(file_name, write_file->_element_id, copy->_element_id);

    return copy;
  }
  return write_file;
}

auto SleeveBase::CopyElement(const sl_path_t &src, const sl_path_t &dest)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto src_read_guard = GetReadGuard(src);
  if (!src_read_guard.has_value()) {
    // TODO: Log
    return std::nullopt;
  }
  auto src_file          = src_read_guard->_access_element;
  // To avoid unnecssary write from mistakenly passed in file path,
  // acquire a read guard here.
  // Multiple references to folder are not allowed, so the read and the write guard have the same behavior here.
  auto target_folder_opt = GetReadGuard(dest);
  if (!target_folder_opt.has_value() || target_folder_opt.value()._access_element->_type == ElementType::FILE) {
    // TODO: Log
    return std::nullopt;
  }
  auto target_folder = std::dynamic_pointer_cast<SleeveFolder>(target_folder_opt.value()._access_element);
  if (target_folder->Contains(src_file->_element_name)) {
    // TODO: Log
    return std::nullopt;
  }
  if (src_file->_type == ElementType::FOLDER) {
    // Folders will always be wholly copied
    auto folder_copy                   = src_file->Copy(_next_element_id++);
    _storage[folder_copy->_element_id] = folder_copy;
    target_folder->AddElementToMap(folder_copy);
    target_folder->IncrementFolderCount();
    return folder_copy;
  }

  // For files, only the reference is copied
  target_folder->AddElementToMap(src_file);
  target_folder->IncrementFileCount();
  return src_file;
}

/**
 * @brief Print the whole file structure under a given folder
 * DEBUG PURPOSE ONLY
 *
 */
auto SleeveBase::Tree(const sl_path_t &path) const -> std::wstring {
  struct TreeNode {
    sl_element_id_t id;
    int             depth;
    bool            is_last;
  };
  std::wstring prefix            = L"";

  auto         target_folder_opt = GetReadGuard(path);
  if (!target_folder_opt.has_value() || target_folder_opt->_access_element->_type == ElementType::FILE) {
    std::wcout << L"Cannot call Tree() on an file" << std::endl;
    return L"";
  }
  std::wstring result       = L"";
  auto         visit_folder = std::dynamic_pointer_cast<SleeveFolder>(target_folder_opt.value()._access_element);
  auto         contains     = visit_folder->ListElements();
  auto         dfs_stack    = std::stack<TreeNode>();
  for (auto e : *contains) {
    dfs_stack.push({e, 0, _storage.at(e)->_type == ElementType::FILE});
  }
  result += visit_folder->_element_name + L"\n";

  while (!dfs_stack.empty()) {
    auto next_visit = dfs_stack.top();
    dfs_stack.pop();
    auto next_visit_element = _storage.at(next_visit.id);

    if (next_visit_element->_type == ElementType::FOLDER) {
      for (int i = 0; i < next_visit.depth; i++) {
        result += L"    ";
      }
      result += L"├── " + next_visit_element->_element_name + L"\n";
      auto sub_folder = std::dynamic_pointer_cast<SleeveFolder>(next_visit_element);
      contains        = sub_folder->ListElements();
      for (auto e : *contains) {
        dfs_stack.push({e, next_visit.depth + 1, _storage.at(e)->_type == ElementType::FILE});
      }
    } else {
      for (int i = 0; i < next_visit.depth; i++) {
        result += L"    ";
      }
      result += L"└── " + next_visit_element->_element_name + L"\n";
    }
  }
  return result;
}
};  // namespace puerhlab