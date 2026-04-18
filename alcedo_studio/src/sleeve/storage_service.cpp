//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "sleeve/storage_service.hpp"

#include <exception>

namespace alcedo {
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
        // DB-backed children already carry persisted ref counts. Rehydrating the in-memory
        // folder map must not add an extra parent reference or the next write will trigger
        // a bogus copy-on-write clone.
        folder->AddElementToMap(content, false, false);
      }
    } catch (std::exception& e) {
      // TODO: LOG
    }
  }
}

void NodeStorageHandler::GarbageCollect() {
  std::vector<sl_element_id_t> to_delete;
  for (auto& pair : storage_) {
    auto element = pair.second;
    if (element->sync_flag_ == SyncFlag::DELETED) {
      to_delete.push_back(element->element_id_);
    }
  }
  for (auto id : to_delete) {
    storage_.erase(id);
  }
}

StorageService::StorageService(std::filesystem::path db_path)
    : db_ctrl_(db_path),
      el_ctrl_(db_ctrl_.GetConnectionGuard()),
      img_ctrl_(db_ctrl_.GetConnectionGuard()) {}

auto StorageService::GetElementController() -> ElementController& { return el_ctrl_; }

auto StorageService::GetImageController() -> ImageController& { return img_ctrl_; }

auto StorageService::GetDBController() -> DBController& { return db_ctrl_; }
};  // namespace alcedo
