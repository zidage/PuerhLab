#pragma once

#include <memory>

#include "image/image.hpp"
#include "sleeve_element.hpp"

namespace puerhlab {

/**
 * @brief A type of element, it contains an image file, its edit history, and other metadata used in this software
 *
 */
class SleeveFile : SleeveElement {
 protected:
  std::shared_ptr<Image> _image;

  std::time_t _added_time;
  std::time_t _last_modified_time;

 public:
  explicit SleeveFile(std::shared_ptr<Image> image);
};
};  // namespace puerhlab