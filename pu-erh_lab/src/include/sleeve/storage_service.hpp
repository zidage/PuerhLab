#pragma once

#include <memory>
#include <unordered_map>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/controller/sleeve/element_controller.hpp"
#include "type/type.hpp"

namespace puerhlab {
class LazyNodeHandler {
  ElementController                                                    _db_ctrl;
  std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>& _storage;

 public:
  void EnsureChildrenLoaded(std::shared_ptr<SleeveFolder> folder);
  auto GetElement(sl_element_id_t id) -> std::shared_ptr<SleeveElement>;
};
};  // namespace puerhlab