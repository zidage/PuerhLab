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
    : _guard(std::move(guard)),
      _element_service(_guard._conn),
      _file_service(_guard._conn),
      _folder_service(_guard._conn),
      _history_service(_guard._conn) {}

/**
 * @brief Add an element to the database.
 *
 * @param element
 */
void ElementController::AddElement(const std::shared_ptr<SleeveElement> element) {
  _element_service.Insert(element);
  if (element->_type == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    _file_service.Insert({file->_element_id, file->_image_id});
    if (file->GetEditHistory() != nullptr) {
      auto history = file->GetEditHistory();
      _history_service.Insert(history);
    }
  } else if (element->_type == ElementType::FOLDER) {
    auto folder   = std::static_pointer_cast<SleeveFolder>(element);
    auto& contents = folder->ListElements();
    for (auto& content_id : contents) {
      _folder_service.Insert({folder->_element_id, content_id});
    }
  }
  element->_sync_flag = SyncFlag::SYNCED;
}

/**
 * @brief Add a content to a folder in the database.
 *
 * @param folder_id
 * @param content_id
 */
void ElementController::AddFolderContent(sl_element_id_t folder_id, sl_element_id_t content_id) {
  // TODO: The uniqueness of content_id is not garanteed, SQL statement should be changed
  _folder_service.Insert({folder_id, content_id});
}

/**
 * @brief Get an element by its ID from the database.
 *
 * @param id
 * @return std::shared_ptr<SleeveElement>
 */
auto ElementController::GetElementById(const sl_element_id_t id) -> std::shared_ptr<SleeveElement> {
  auto result = _element_service.GetElementById(id);
  if (result->_type == ElementType::FILE) {
    auto file    = std::static_pointer_cast<SleeveFile>(result);
    auto history = _history_service.GetEditHistoryByFileId(file->_element_id);
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
  return _folder_service.GetFolderContent(folder_id);
}

/**
 * @brief Remove an element by its ID from the database, only be called when the ref count to the
 * element is 0.
 *
 * @param id
 */
void ElementController::RemoveElement(const sl_element_id_t id) { _element_service.RemoveById(id); }

void ElementController::RemoveElement(const std::shared_ptr<SleeveElement> element) {
  if (element->_type == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    _history_service.RemoveById(file->_element_id);
    _file_service.RemoveById(file->_element_id);
  } else if (element->_type == ElementType::FOLDER) {
    auto folder = std::static_pointer_cast<SleeveFolder>(element);
    _folder_service.RemoveById(folder->_element_id);
  }
  _element_service.RemoveById(element->_element_id);
}

/**
 * @brief Update an element in the database.
 *
 * @param element
 */
void ElementController::UpdateElement(const std::shared_ptr<SleeveElement> element) {
  _element_service.Update(element, element->_element_id);
  if (element->_type == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    _file_service.Update({file->_element_id, file->_image_id}, file->_image_id);
    _history_service.Update(file->GetEditHistory(), file->_element_id);
  } else if (element->_type == ElementType::FOLDER) {
    auto folder = std::static_pointer_cast<SleeveFolder>(element);
    _folder_service.RemoveById(folder->_element_id);
    for (auto& content_id : folder->ListElements()) {
      AddFolderContent(folder->_element_id, content_id);
    }
  }
  element->_sync_flag = SyncFlag::SYNCED;
}

auto ElementController::GetElementsInFolderByFilter(const std::shared_ptr<FilterCombo> filter,
                                                    const sl_element_id_t              folder_id)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  // Build SQL query from the filter
  std::wstring filter_sql = filter->GenerateSQLOn(folder_id);
  return _element_service.GetElementsInFolderByFilter(filter_sql); // for specialized queries only
}

};  // namespace puerhlab