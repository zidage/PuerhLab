#pragma once

#include <cstdint>

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
};
};  // namespace puerhlab
