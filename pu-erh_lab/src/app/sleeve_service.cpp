//  Copyright 2026 Yurun Zi
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

#include "app/sleeve_service.hpp"

#include <stdexcept>

namespace puerhlab {
SleeveServiceImpl::SleeveServiceImpl(std::shared_ptr<StorageService> storage_service,
                                     const std::filesystem::path&   db_path,
                                     sl_element_id_t                start_id)
    : storage_service_(std::move(storage_service)), db_path_(db_path) {
  if (!storage_service_) {
    throw std::invalid_argument("StorageService is null");
  }
  fs_ = std::make_unique<FileSystem>(db_path_, *storage_service_, start_id);
  fs_->InitRoot();
}

auto SleeveServiceImpl::Sync() -> SyncResult {
  SyncResult result{true, 0, ""};
  try {
    auto& element_ctrl      = storage_service_->GetElementController();
    auto  modified_elements = fs_->GetModifiedElements();
    auto  unsynced_elements = fs_->GetUnsyncedElements();
    auto  garbage_elements  = fs_->GetDeletedElements();

    // Sync unsynced elements first
    for (auto& element : unsynced_elements) {
      element_ctrl.AddElement(element);
      result.elements_synced_++;
    }
    // Then sync modified elements
    for (auto& element : modified_elements) {
      element_ctrl.UpdateElement(element);
      result.elements_synced_++;
    }

    // Finally, delete the deleted elements
    // TODO: This should be done periodically instead of every sync
    for (auto& element : garbage_elements) {
      element_ctrl.RemoveElement(element->element_id_);
      result.elements_synced_++;
    }
    // Perform garbage collection in the storage
    // The same goes to here, this should be done periodically
    fs_->GarbageCollect();
  } catch (std::exception& e) {
    result.success_ = false;
    result.message_ = e.what();
  }
  return result;
}

auto SleeveServiceImpl::GetCurrentID() const -> sl_element_id_t {
  std::lock_guard<std::mutex> lock(fs_lock_);
  return fs_->GetCurrentID();
}
};  // namespace puerhlab