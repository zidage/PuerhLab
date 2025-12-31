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
#include <unordered_map>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/controller/db_controller.hpp"
#include "storage/controller/image/image_controller.hpp"
#include "storage/controller/sleeve/element_controller.hpp"
#include "type/type.hpp"

namespace puerhlab {
class NodeStorageHandler {
 private:
  ElementController&                                                   _db_ctrl;

  std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& _storage;

 public:
  NodeStorageHandler(ElementController&                                                   db_ctrl,
                     std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& storage);
  void AddToStorage(std::shared_ptr<SleeveElement> new_element);
  void EnsureChildrenLoaded(std::shared_ptr<SleeveFolder> folder);
  auto GetElement(sl_element_id_t id) -> std::shared_ptr<SleeveElement>;
};

class StorageService {
 private:
  DBController      _db_ctrl;
  ElementController _el_ctrl;
  ImageController   _img_ctrl;

 public:
  StorageService(std::filesystem::path db_path);

  auto GetDBController() -> DBController&;
  auto GetElementController() -> ElementController&;
  auto GetImageController() -> ImageController&;
};
};  // namespace puerhlab