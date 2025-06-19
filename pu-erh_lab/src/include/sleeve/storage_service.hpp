#pragma once

#include <memory>
#include <unordered_map>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/controller/db_controller.hpp"
#include "storage/controller/image/image_controller.hpp"
#include "storage/controller/sleeve/element_controller.hpp"
#include "type/type.hpp"

namespace puerhlab {
class LazyNodeHandler {
 private:
  ElementController&                                                   _db_ctrl;

  std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& _storage;

 public:
  LazyNodeHandler(ElementController&                                                   db_ctrl,
                  std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& storage);
  void EnsureChildrenLoaded(std::shared_ptr<SleeveFolder> folder);
  auto GetElement(sl_element_id_t id) -> std::shared_ptr<SleeveElement>;
};

class StorageService {
 private:
  DBController      _db_ctrl;
  ElementController _el_ctrl;
  ImageController   _img_ctrl;

 public:
  StorageService(std::filesystem::path db_path);

  auto GetElementController() -> ElementController&;
  auto GetImageController() -> ImageController&;
};
};  // namespace puerhlab