#include "sleeve/storage_service.hpp"

#include <exception>

namespace puerhlab {
NodeStorageHandler::NodeStorageHandler(
    ElementController&                                                   db_ctrl,
    std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& storage)
    : _db_ctrl(db_ctrl), _storage(storage) {}

void NodeStorageHandler::AddToStorage(std::shared_ptr<SleeveElement> new_element) {
  _storage[new_element->_element_id] = new_element;
}

auto NodeStorageHandler::GetElement(uint32_t id) -> std::shared_ptr<SleeveElement> {
  if (_storage.contains(id)) {
    return _storage.at(id);
  }
  // If the element is not presented in the memory, get it from the db.
  // Then loaded pointer into the storage
  auto result                   = _db_ctrl.GetElementById(id);
  _storage[result->_element_id] = result;
  return result;
}

void NodeStorageHandler::EnsureChildrenLoaded(std::shared_ptr<SleeveFolder> folder) {
  // Assume all the lazy-loaded folder are empty for now
  if (folder->ContentSize() == 0) {
    try {
      auto folder_content = _db_ctrl.GetFolderContent(folder->_element_id);
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
    : _db_ctrl(db_path),
      _el_ctrl(_db_ctrl.GetConnectionGuard()),
      _img_ctrl(_db_ctrl.GetConnectionGuard()) {}

auto StorageService::GetElementController() -> ElementController& { return _el_ctrl; }

auto StorageService::GetImageController() -> ImageController& { return _img_ctrl; }
};  // namespace puerhlab