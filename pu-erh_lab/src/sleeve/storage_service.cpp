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

#include "sleeve/storage_service.hpp"

#include <exception>

namespace puerhlab {
NodeStorageHandler::NodeStorageHandler(
    ElementController&                                                   db_ctrl,
    std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& storage)
    : db_ctrl_(db_ctrl), storage_(storage) {}

void NodeStorageHandler::AddToStorage(std::shared_ptr<SleeveElement> new_element) {
  storage_[new_element->element_id_] = new_element;
}

auto NodeStorageHandler::GetElement(uint32_t id) -> std::shared_ptr<SleeveElement> {
  if (storage_.contains(id)) {
    return storage_.at(id);
  }
  // If the element is not presented in the memory, get it from the db.
  // Then loaded pointer into the storage
  auto result                   = db_ctrl_.GetElementById(id);
  storage_[result->element_id_] = result;
  return result;
}

void NodeStorageHandler::EnsureChildrenLoaded(std::shared_ptr<SleeveFolder> folder) {
  // Assume all the lazy-loaded folder are empty for now
  if (folder->ContentSize() == 0) {
    try {
      auto folder_content = db_ctrl_.GetFolderContent(folder->element_id_);
      for (auto& content_id : folder_content) {
        auto content = GetElement(content_id);
        folder->AddElementToMap(content);
      }
    } catch (std::exception& e) {
      // TODO: LOG
    }
  }
}

StorageService::StorageService(std::filesystem::path db_path)
    : db_ctrl_(db_path),
      el_ctrl_(db_ctrl_.GetConnectionGuard()),
      img_ctrl_(db_ctrl_.GetConnectionGuard()) {}

auto StorageService::GetElementController() -> ElementController& { return el_ctrl_; }

auto StorageService::GetImageController() -> ImageController& { return img_ctrl_; }

auto StorageService::GetDBController() -> DBController& { return db_ctrl_; }
};  // namespace puerhlab