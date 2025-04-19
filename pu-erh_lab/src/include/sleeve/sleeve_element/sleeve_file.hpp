#pragma once

#include <cstdint>
#include <memory>

#include "edit/history/edit_history.hpp"
#include "image/image.hpp"
#include "sleeve_element.hpp"
#include "type/type.hpp"

namespace puerhlab {

/**
 * @brief A type of element, it contains an image file, its edit history, and other metadata used in this software
 *
 */
class SleeveFile : public SleeveElement {
 private:
  std::shared_ptr<Image>       _image;

  std::shared_ptr<EditHistory> _edit_history;
  std::shared_ptr<Version>     _current_version;

  uint32_t                     _ref_count;

 public:
  ElementType _type = ElementType::FILE;
  explicit SleeveFile(sl_element_id_t id, file_name_t element_name);
  explicit SleeveFile(sl_element_id_t id, file_name_t element_name, std::shared_ptr<Image> image);

  ~SleeveFile();
};
};  // namespace puerhlab