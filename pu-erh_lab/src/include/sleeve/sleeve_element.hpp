#pragma once

#include <cstdint>
#include <ctime>

#include "type/type.hpp"

namespace puerhlab {
enum class ElementType { FILE, FOLDER };

/**
 * @brief Abstract objects residing in a sleeve, it can be files or folders
 *
 */
class SleeveElement {
 protected:
  sl_element_id_t _element_id;
  ElementType     _e_type;

  std::time_t _added_time;
  std::time_t _last_modified_time;

  explicit SleeveElement(sl_element_id_t id);

  void SetAddTime();
  void SetLastModifiedTime();
};
};  // namespace puerhlab
