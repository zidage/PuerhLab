#pragma once

#include <memory>

#include "sleeve_element.hpp"
#include "sleeve_file.hpp"
#include "sleeve_folder.hpp"

namespace puerhlab {
class SleeveElementFactory {
 public:
  static std::shared_ptr<SleeveElement> CreateElement(const ElementType &type, sl_element_id_t id,
                                                      file_name_t element_name);
};
};  // namespace puerhlab