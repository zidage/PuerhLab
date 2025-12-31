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

#include <memory>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "storage/controller/controller_types.hpp"
#include "storage/service/sleeve/edit_history/history_service.hpp"
#include "storage/service/sleeve/element/element_service.hpp"
#include "storage/service/sleeve/element/file_service.hpp"
#include "storage/service/sleeve/element/folder_service.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ElementController {
 private:
  ConnectionGuard    _guard;

  ElementService     _element_service;
  FileService        _file_service;
  FolderService      _folder_service;
  EditHistoryService _history_service;

 public:
  ElementController(ConnectionGuard&& guard);

  void AddElement(const std::shared_ptr<SleeveElement> element);

  void AddFolderContent(sl_element_id_t folder_id, sl_element_id_t content_id);
  auto GetFolderContent(const sl_element_id_t folder_id) -> std::vector<sl_element_id_t>;

  void RemoveElement(const sl_element_id_t id);
  void RemoveElement(const std::shared_ptr<SleeveElement> element);
  void UpdateElement(const std::shared_ptr<SleeveElement> element);
  auto GetElementById(const sl_element_id_t id) -> std::shared_ptr<SleeveElement>;

  auto GetElementsInFolderByFilter(const std::shared_ptr<FilterCombo> filter,
                                   const sl_element_id_t              folder_id)
      -> std::vector<std::shared_ptr<SleeveElement>>;

  void EnsureChildrenLoaded(sl_element_id_t folder_id);
};
};  // namespace puerhlab