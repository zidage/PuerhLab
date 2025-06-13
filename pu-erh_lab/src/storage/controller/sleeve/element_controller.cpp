#include "storage/controller/sleeve/element_controller.hpp"

#include <memory>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"

namespace puerhlab {
ElementController::ElementController(ConnectionGuard&& guard)
    : _guard(guard),
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

};  // namespace puerhlab