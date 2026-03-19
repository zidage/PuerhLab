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

auto SleeveServiceImpl::ResolveElement(const std::filesystem::path& path)
    -> std::shared_ptr<SleeveElement> {
  return Read<std::shared_ptr<SleeveElement>>(
      [path](FileSystem& fs) { return fs.Get(path, false); });
}

auto SleeveServiceImpl::ResolveFolder(const std::filesystem::path& path)
    -> std::shared_ptr<SleeveFolder> {
  auto element = ResolveElement(path);
  if (!element || element->type_ != ElementType::FOLDER ||
      element->sync_flag_ == SyncFlag::DELETED) {
    throw std::runtime_error("SleeveService: Target path is not a folder.");
  }
  return std::static_pointer_cast<SleeveFolder>(element);
}

auto SleeveServiceImpl::ResolveFile(const std::filesystem::path& path)
    -> std::shared_ptr<SleeveFile> {
  auto element = ResolveElement(path);
  if (!element || element->type_ != ElementType::FILE ||
      element->sync_flag_ == SyncFlag::DELETED) {
    throw std::runtime_error("SleeveService: Target path is not a file.");
  }
  auto file = std::dynamic_pointer_cast<SleeveFile>(element);
  if (!file) {
    throw std::runtime_error("SleeveService: Failed to resolve file pointer.");
  }
  return file;
}

auto SleeveServiceImpl::ListFolderEntries(const std::filesystem::path& folder_path)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  return Read<std::vector<std::shared_ptr<SleeveElement>>>([folder_path](FileSystem& fs) {
    std::vector<std::shared_ptr<SleeveElement>> entries;
    const auto ids = fs.ListFolderContent(folder_path, false);
    entries.reserve(ids.size());
    for (const auto id : ids) {
      auto element = fs.Get(id);
      if (!element || element->sync_flag_ == SyncFlag::DELETED) {
        continue;
      }
      entries.push_back(std::move(element));
    }
    return entries;
  });
}

auto SleeveServiceImpl::CreateFolder(const std::filesystem::path& parent_path,
                                     const file_name_t&          name)
    -> std::pair<std::shared_ptr<SleeveFolder>, SyncResult> {
  auto result = Write<std::shared_ptr<SleeveFolder>>([parent_path, name](FileSystem& fs) {
    auto created = fs.Create(parent_path, name, ElementType::FOLDER);
    if (!created || created->type_ != ElementType::FOLDER) {
      throw std::runtime_error("SleeveService: Failed to create folder.");
    }
    return std::dynamic_pointer_cast<SleeveFolder>(created);
  });
  if (!result.first) {
    throw std::runtime_error("SleeveService: Failed to create folder.");
  }
  return result;
}

auto SleeveServiceImpl::DeletePath(const std::filesystem::path& target_path) -> SyncResult {
  return Write<void>([target_path](FileSystem& fs) { fs.Delete(target_path); });
}
};  // namespace puerhlab
