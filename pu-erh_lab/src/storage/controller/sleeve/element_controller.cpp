#include "storage/controller/sleeve/element_controller.hpp"

#include <format>
#include <memory>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"

namespace puerhlab {
ElementController::ElementController(ConnectionGuard&& guard)
    : _guard(std::move(guard)),
      _element_service(_guard._conn),
      _file_service(_guard._conn),
      _folder_service(_guard._conn) {}

void ElementController::AddElement(const std::shared_ptr<SleeveElement> element) {
  _element_service.Insert(element);
  if (element->_type == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    _file_service.Insert({file->_element_id, file->_image_id});
  } else if (element->_type == ElementType::FOLDER) {
    auto folder = std::static_pointer_cast<SleeveFolder>(element);
    for (auto& content_id : *folder->ListElements()) {
      _folder_service.Insert({folder->_element_id, content_id});
    }
  }
}

void ElementController::AddFolderContent(sl_element_id_t folder_id, sl_element_id_t content_id) {
  // TODO: The uniqueness of content_id is not garanteed, SQL statement should be changed
  _folder_service.Insert({folder_id, content_id});
}

auto ElementController::GetElementById(const sl_element_id_t id) -> std::shared_ptr<SleeveElement> {
  return _element_service.GetElementById(id);
}

void ElementController::RemoveElement(const sl_element_id_t id) { _element_service.RemoveById(id); }

void ElementController::UpdateElement(const std::shared_ptr<SleeveElement> element) {
  _element_service.Update(element, element->_element_id);
  if (element->_type == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    _file_service.Update({file->_element_id, file->_image_id}, file->_image_id);
  } else if (element->_type == ElementType::FOLDER) {
    auto folder = std::static_pointer_cast<SleeveFolder>(element);
    _folder_service.RemoveById(folder->_element_id);
    for (auto& content_id : *folder->ListElements()) {
      AddFolderContent(folder->_element_id, content_id);
    }
  }
}

};  // namespace puerhlab