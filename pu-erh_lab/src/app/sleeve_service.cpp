//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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
      element_ctrl.RemoveElement(element);
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
