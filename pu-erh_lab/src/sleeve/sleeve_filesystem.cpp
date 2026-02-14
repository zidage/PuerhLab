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

#include "sleeve/sleeve_filesystem.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_element_factory.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
FileSystem::FileSystem(std::filesystem::path db_path, StorageService& storage_service,
                       sl_element_id_t start_id)
    : id_gen_(start_id),
      db_path_(db_path),
      storage_service_(storage_service),
      storage_handler_(storage_service_.GetElementController(), storage_),
      resolver_(storage_handler_, id_gen_) {
  root_ = nullptr;
}

auto FileSystem::InitRoot() -> bool {
  if (root_ != nullptr) {
    throw std::runtime_error("Filesystem: root has already been initialized");
  }
  // root's id is always 0
  std::shared_ptr<SleeveElement> root;
  try {
    root = storage_service_.GetElementController().GetElementById(0);
  } catch (std::exception& e) {
    root = SleeveElementFactory::CreateElement(ElementType::FOLDER, 0, L"");
  }
  storage_[0] = root;
  root_       = std::static_pointer_cast<SleeveFolder>(root);
  resolver_.SetRoot(root_);
  return true;
}

auto FileSystem::Create(std::filesystem::path dest, std::wstring filename, ElementType type)
    -> std::shared_ptr<SleeveElement> {
  auto dest_element = resolver_.ResolveForWrite(dest);
  if (dest_element->type_ != ElementType::FOLDER) {
    throw std::runtime_error("Filesystem: Cannot create element under a file");
  }
  auto dest_folder = std::static_pointer_cast<SleeveFolder>(dest_element);
  while (dest_folder->Contains(filename)) {
    filename = filename + L"@";
  }
  auto new_id      = id_gen_.GenerateID();
  auto new_element = SleeveElementFactory::CreateElement(type, new_id, filename);
  storage_[new_id] = new_element;
  dest_folder->AddElementToMap(new_element);

  return new_element;
}

auto FileSystem::Get(sl_element_id_t id) -> std::shared_ptr<SleeveElement> {
  return storage_handler_.GetElement(id);
}

auto FileSystem::Get(std::filesystem::path target, bool write) -> std::shared_ptr<SleeveElement> {
  if (write) {
    return resolver_.ResolveForWrite(target);
  }
  return resolver_.Resolve(target);
}

auto FileSystem::ListFolderContent(const std::filesystem::path& folder_path, bool write)
    -> std::vector<sl_element_id_t> {
  std::shared_ptr<SleeveElement> folder_element;
  if (write) {
    folder_element = resolver_.ResolveForWrite(folder_path);
  } else {
    folder_element = resolver_.Resolve(folder_path);
  }

  if (!folder_element || folder_element->type_ != ElementType::FOLDER) {
    throw std::runtime_error("Filesystem: Specified path is not a folder");
  }

  auto folder = std::static_pointer_cast<SleeveFolder>(folder_element);
  storage_handler_.EnsureChildrenLoaded(folder);
  const auto& elements = folder->ListElements();
  return std::vector<sl_element_id_t>(elements.begin(), elements.end());
}

auto FileSystem::ListFolderContent(sl_element_id_t folder_id) -> std::vector<sl_element_id_t> {
  auto folder_element = Get(folder_id);
  if (!folder_element || folder_element->type_ != ElementType::FOLDER) {
    throw std::runtime_error("Filesystem: Specified id is not a folder");
  }

  auto folder = std::static_pointer_cast<SleeveFolder>(folder_element);
  storage_handler_.EnsureChildrenLoaded(folder);
  const auto& elements = folder->ListElements();
  return std::vector<sl_element_id_t>(elements.begin(), elements.end());
}

void FileSystem::Delete(std::filesystem::path target) {
  if (!target.has_parent_path()) {
    throw std::runtime_error("Filesystem: root cannot be deleted");
  }
  auto parent           = target.parent_path();
  auto delete_node_name = target.filename();
  if (!resolver_.Contains(parent) || !resolver_.Contains(target)) {
    throw std::runtime_error("Filesystem: Deleting node does not exist");
  }
  // Now re-acquire parent_node using write method
  auto parent_write_node =
      std::static_pointer_cast<SleeveFolder>(resolver_.ResolveForWrite(parent));
  auto delete_node_id = parent_write_node->GetElementIdByName(delete_node_name.wstring());
  auto delete_node    = storage_.at(delete_node_id.value());
  delete_node->DecrementRefCount();

  if (delete_node->ref_count_ <= 0) {
    // Mark the node as deleted in storage handler
    // If it's a folder, recursively decrement children's ref count
    // If the child's ref count reaches 0, decrement its children's ref count, and so on.
    if (delete_node->type_ == ElementType::FOLDER) {
      auto delete_folder = std::static_pointer_cast<SleeveFolder>(delete_node);
      storage_handler_.EnsureChildrenLoaded(delete_folder);
      auto& children = delete_folder->ListElements();
      for (auto& child_id : children) {
        auto child = storage_.at(child_id);
        child->DecrementRefCount();
        if (child->ref_count_ <= 0 && child->type_ == ElementType::FOLDER) {
          // Recursively decrement
          std::filesystem::path child_path =
              target / child->element_name_;  // Construct child's path for resolver
          Delete(child_path);
        }
      }
    }


  }

  parent_write_node->RemoveNameFromMap(delete_node_name.wstring());
}

void FileSystem::Copy(std::filesystem::path from, std::filesystem::path dest) {
  // Path resolver will do the sanity check
  if (resolver_.IsSubpath(from, dest)) {
    throw std::runtime_error(
        "Filesystem: Target folder cannot be a subfolder of the original folder");
  }
  if (!resolver_.Contains(from) || !resolver_.Contains(dest, ElementType::FOLDER)) {
    throw std::runtime_error("Filesystem: Origin path or destination path does not exist");
  }

  auto from_node = resolver_.Resolve(from);
  auto dest_node = std::static_pointer_cast<SleeveFolder>(resolver_.ResolveForWrite(dest));
  dest_node->AddElementToMap(from_node);
}

auto FileSystem::ApplyFilterToFolder(const std::filesystem::path&       folder_path,
                                     const std::shared_ptr<FilterCombo> filter)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  // TODO: Decouple this into a separate FilterService
  if (!resolver_.Contains(folder_path, ElementType::FOLDER)) {
    throw std::runtime_error("Filesystem: Specified folder does not exist");
  }
  // Here we only need to "read" the folder, so no need to use ResolveForWrite
  std::shared_ptr<SleeveElement> folder_element;
  try {
    folder_element = Get(folder_path, false);
  } catch (std::exception& e) {
    throw std::runtime_error("Filesystem: Specified folder does not exist");
  }

  if (folder_element->type_ != ElementType::FOLDER) {
    throw std::runtime_error("Filesystem: Specified path is not a folder");
  }

  auto folder = std::static_pointer_cast<SleeveFolder>(folder_element);

  std::vector<std::shared_ptr<SleeveElement>> result_elements;
  // First check if the folder has cached index for this filter
  if (folder->HasFilterIndex(filter->filter_id_)) {
    auto& cached_ids = folder->ListElementsByFilter(filter->filter_id_);

    result_elements.reserve(cached_ids.size());
    for (const auto& id : cached_ids) {
      result_elements.push_back(Get(id));
    }
    return result_elements;
  }

  auto result_elements_db = storage_service_.GetElementController().GetElementsInFolderByFilter(
      filter, folder->element_id_);
  // Just for cache purpose, which will not affect sync status nor anything else
  folder->CreateIndex(result_elements_db, filter->filter_id_);

  // Then return the result from the storage service
  auto& cached_ids = folder->ListElementsByFilter(filter->filter_id_);
  result_elements.reserve(cached_ids.size());
  for (const auto& id : cached_ids) {
    result_elements.push_back(Get(id));
  }
  return result_elements;
}

auto FileSystem::GetModifiedElements() -> std::vector<std::shared_ptr<SleeveElement>> {
  std::vector<std::shared_ptr<SleeveElement>> modified_elements;
  for (auto& pair : storage_) {
    auto element = pair.second;
    if (element->sync_flag_ == SyncFlag::MODIFIED) {
      modified_elements.push_back(element);
    }
  }
  return modified_elements;
}

auto FileSystem::GetUnsyncedElements() -> std::vector<std::shared_ptr<SleeveElement>> {
  std::vector<std::shared_ptr<SleeveElement>> unsynced_elements;
  for (auto& pair : storage_) {
    auto element = pair.second;
    if (element->sync_flag_ == SyncFlag::UNSYNC) {
      unsynced_elements.push_back(element);
    }
  }
  return unsynced_elements;
}

auto FileSystem::GetDeletedElements() -> std::vector<std::shared_ptr<SleeveElement>> {
  std::vector<std::shared_ptr<SleeveElement>> deleted_elements;
  for (auto& pair : storage_) {
    auto element = pair.second;
    if (element->sync_flag_ == SyncFlag::DELETED) {
      deleted_elements.push_back(element);
    }
  }
  return deleted_elements;
}

void FileSystem::SyncToDB() {
  auto& element_ctrl = storage_service_.GetElementController();
  for (auto& pair : storage_) {
    auto element = pair.second;
    if (element->sync_flag_ == SyncFlag::UNSYNC) {
      element_ctrl.AddElement(element);
    } else if (element->sync_flag_ == SyncFlag::MODIFIED) {
      element_ctrl.UpdateElement(element);
    }
  }
}

void FileSystem::WriteSleeveMeta(const std::filesystem::path& meta_path) {
  nlohmann::json metadata;
  meta_path_            = meta_path;
  metadata["db_path"]   = conv::ToBytes(db_path_.wstring());
  metadata["meta_path"] = conv::ToBytes(meta_path_.wstring());
  metadata["start_id"]  = id_gen_.GetCurrentID();

  std::ofstream file(meta_path);
  if (file.is_open()) {
    file << metadata.dump(4);
    file.close();
  }
}

void FileSystem::ReadSleeveMeta(const std::filesystem::path& meta_path) {
  std::ifstream file(meta_path);
  if (file.is_open()) {
    nlohmann::json metadata;
    file >> metadata;
    db_path_   = std::filesystem::path(conv::FromBytes(metadata["db_path"]));
    meta_path_ = std::filesystem::path(conv::FromBytes(metadata["meta_path"]));
    id_gen_.SetStartID(static_cast<uint32_t>(metadata["start_id"]));
  }
}
auto FileSystem::Tree(const std::filesystem::path& path) -> std::wstring {
  return resolver_.Tree(path);
}
};  // namespace puerhlab
