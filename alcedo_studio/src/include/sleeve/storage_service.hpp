//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>
#include <unordered_map>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/controller/db_controller.hpp"
#include "storage/controller/image/image_controller.hpp"
#include "storage/controller/sleeve/element_controller.hpp"
#include "type/type.hpp"

namespace alcedo {
class NodeStorageHandler {
 private:
  ElementController&                                                   db_ctrl_;

  std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& storage_;

 public:
  NodeStorageHandler(ElementController&                                                   db_ctrl,
                     std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& storage);
  void AddToStorage(std::shared_ptr<SleeveElement> new_element);
  void EnsureChildrenLoaded(std::shared_ptr<SleeveFolder> folder);
  auto GetElement(sl_element_id_t id) -> std::shared_ptr<SleeveElement>;
  void GarbageCollect();
};

class StorageService {
 private:
  DBController      db_ctrl_;
  ElementController el_ctrl_;
  ImageController   img_ctrl_;

 public:
  StorageService(std::filesystem::path db_path);

  auto GetDBController() -> DBController&;
  auto GetElementController() -> ElementController&;
  auto GetImageController() -> ImageController&;
};
};  // namespace alcedo