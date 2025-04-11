#pragma once

#include <unordered_map>

#include "sleeve_element.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief A type of element that contains files or folders of its kind
 *
 */
class SleeveFolder : SleeveElement {
  std::unordered_map<sl_element_id_t, SleeveElement> _contents;
};
};  // namespace puerhlab