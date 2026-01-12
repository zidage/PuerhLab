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

#include "storage/controller/sleeve/element_controller.hpp"

#include <memory>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief Construct a new Element Controller:: Element Controller object
 *
 * @param guard
 */
ElementController::ElementController(ConnectionGuard&& guard)
    : guard_(std::move(guard)),
      element_service_(guard_.conn_),
      file_service_(guard_.conn_),
      folder_service_(guard_.conn_),
      history_service_(guard_.conn_),
      pipeline_service_(guard_.conn_) {}
/**
 * @brief Add an element to the database.
 *
 * @param element
 */
void ElementController::AddElement(const std::shared_ptr<SleeveElement> element) {
  element_service_.Insert(element);
  if (element->type_ == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    file_service_.Insert({file->element_id_, file->image_id_});
    if (file->GetEditHistory() != nullptr) {
      auto history = file->GetEditHistory();
      history_service_.Insert(history);
    }
  } else if (element->type_ == ElementType::FOLDER) {
    auto  folder   = std::static_pointer_cast<SleeveFolder>(element);
    auto& contents = folder->ListElements();
    for (auto& content_id : contents) {
      folder_service_.Insert({folder->element_id_, content_id});
    }
  }
  element->sync_flag_ = SyncFlag::SYNCED;
}

/**
 * @brief Add a content to a folder in the database.
 *
 * @param folder_id
 * @param content_id
 */
void ElementController::AddFolderContent(sl_element_id_t folder_id, sl_element_id_t content_id) {
  // TODO: The uniqueness of content_id is not garanteed, SQL statement should be changed
  folder_service_.Insert({folder_id, content_id});
}

/**
 * @brief Get an element by its ID from the database.
 *
 * @param id
 * @return std::shared_ptr<SleeveElement>
 */
auto ElementController::GetElementById(const sl_element_id_t id) -> std::shared_ptr<SleeveElement> {
  auto result = element_service_.GetElementById(id);
  if (result->type_ == ElementType::FILE) {
    auto file    = std::static_pointer_cast<SleeveFile>(result);
    auto history = history_service_.GetEditHistoryByFileId(file->element_id_);
    file->SetEditHistory(history);
  }
  result->SetSyncFlag(SyncFlag::SYNCED);
  return result;
}

/**
 * @brief Get the content of a folder by its ID from the database.
 *
 * @param folder_id
 * @return std::vector<sl_element_id_t>
 */
auto ElementController::GetFolderContent(const sl_element_id_t folder_id)
    -> std::vector<sl_element_id_t> {
  return folder_service_.GetFolderContent(folder_id);
}

/**
 * @brief Remove an element by its ID from the database, only be called when the ref count to the
 * element is 0.
 *
 * @param id
 */
void ElementController::RemoveElement(const sl_element_id_t id) { element_service_.RemoveById(id); }

void ElementController::RemoveElement(const std::shared_ptr<SleeveElement> element) {
  if (element->type_ == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    history_service_.RemoveById(file->element_id_);
    file_service_.RemoveById(file->element_id_);
  } else if (element->type_ == ElementType::FOLDER) {
    auto folder = std::static_pointer_cast<SleeveFolder>(element);
    folder_service_.RemoveById(folder->element_id_);
  }
  element_service_.RemoveById(element->element_id_);
}

/**
 * @brief Update an element in the database.
 *
 * @param element
 */
void ElementController::UpdateElement(const std::shared_ptr<SleeveElement> element) {
  element_service_.Update(element, element->element_id_);
  if (element->type_ == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    file_service_.Update({file->element_id_, file->image_id_}, file->image_id_);
    history_service_.Update(file->GetEditHistory(), file->element_id_);
  } else if (element->type_ == ElementType::FOLDER) {
    auto folder = std::static_pointer_cast<SleeveFolder>(element);
    folder_service_.RemoveById(folder->element_id_);
    for (auto& content_id : folder->ListElements()) {
      AddFolderContent(folder->element_id_, content_id);
    }
  }
  element->sync_flag_ = SyncFlag::SYNCED;
}

auto ElementController::GetElementsInFolderByFilter(const std::shared_ptr<FilterCombo> filter,
                                                    const sl_element_id_t              folder_id)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  // Build SQL query from the filter
  std::wstring filter_sql = filter->GenerateSQLOn(folder_id);
  return element_service_.GetElementsInFolderByFilter(filter_sql);  // for specialized queries only
}

auto ElementController::GetPipelineByElementId(const sl_element_id_t element_id)
    -> std::shared_ptr<CPUPipelineExecutor> {
  return pipeline_service_.GetPipelineParamByFileId(element_id);
}

};  // namespace puerhlab