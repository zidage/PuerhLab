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

#include "sleeve/sleeve_base.hpp"

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "mapper/sleeve/sleeve_mapper.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_element_factory.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief Construct a new Sleeve Base:: Sleeve Base object
 *
 * @param id
 */
SleeveBase::SleeveBase(sleeve_id_t id) : _sleeve_id(id) {
  _next_element_id = 0;
  _size            = 0;
  InitializeRoot();
}

/**
 * @brief Initialize the root folder within the base
 *
 */
void SleeveBase::InitializeRoot() {
  if (_root != nullptr) _size += 1;
  _root                        = std::make_shared<SleeveFolder>(_next_element_id++, L"root");
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
      // Find a file in path's prefix, which is illegal
      return std::nullopt;
    }

    curr_path = path_elements.front();
    path_elements.pop_front();
    auto next_element_id = std::dynamic_pointer_cast<SleeveFolder>(curr_element.value())->GetElementByName(curr_path);
    if (!next_element_id.has_value()) {
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
    return std::nullopt;
  }
  std::shared_ptr<SleeveElement> newElement = SleeveElementFactory::CreateElement(type, _next_element_id++, file_name);
  // Update the map
  _storage[newElement->_element_id]         = newElement;
  parent_folder->AddElement(newElement);
  // Increment the reference count of the file
  if (newElement->_type == ElementType::FILE) {
    std::dynamic_pointer_cast<SleeveFile>(newElement)->IncrementRefCount();
    parent_folder->IncrementFileCount();
  } else {
    parent_folder->IncrementFolderCount();
  }

  return newElement;
}

/**
 * @brief
 *
 * @param full_path
 * @return std::optional<std::shared_ptr<SleeveElement>>
 */
auto SleeveBase::RemoveElementInPath(const sl_path_t &path, const file_name_t &file_name)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto parent_folder_opt = AccessElementByPath(path);
  if (!parent_folder_opt.has_value() || parent_folder_opt.value()->_type == ElementType::FILE) {
    return std::nullopt;
  }
  auto parent_folder = std::dynamic_pointer_cast<SleeveFolder>(parent_folder_opt.value());
  return RemoveElementInPath(parent_folder, file_name);
}

auto SleeveBase::RemoveElementInPath(const std::shared_ptr<SleeveFolder> parent_folder, const file_name_t &file_name)
    -> std::optional<std::shared_ptr<SleeveElement>> {
  auto del_id = parent_folder->GetElementByName(file_name);
  if (!del_id.has_value()) {
    return std::nullopt;
  }
  auto del_element = AccessElementById(del_id.value()).value();
  if (del_element->_type == ElementType::FOLDER) {
    auto del_folder = std::dynamic_pointer_cast<SleeveFolder>(del_element);
    del_folder->ClearFolder();
  }
  parent_folder->RemoveElementByName(del_element->_element_name);
  del_element->DecrementRefCount();

  return del_element;
}

};  // namespace puerhlab