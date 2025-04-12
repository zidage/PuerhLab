#pragma once

#include <unordered_map>
#include <vector>

#include "sleeve_element.hpp"
#include "sleeve_filter.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief A type of element that contains files or folders of its kind
 *
 */
class SleeveFolder : SleeveElement {
  std::unordered_map<sl_element_id_t, SleeveElement>             _contents;
  std::unordered_map<SleeveFilter, std::vector<sl_element_id_t>> _indicies_cache;

  explicit SleeveFolder(sl_element_id_t id);
};
};  // namespace puerhlab