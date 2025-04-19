#include "sleeve/sleeve_element/sleeve_file.hpp"

#include "sleeve/sleeve_element/sleeve_element.hpp"

namespace puerhlab {
SleeveFile::~SleeveFile() {}
SleeveFile::SleeveFile(sl_element_id_t id, file_name_t element_name) : SleeveElement(id, element_name) {}
SleeveFile::SleeveFile(sl_element_id_t id, file_name_t element_name, std::shared_ptr<Image> image)
    : SleeveElement(id, element_name) {
  _image = image;
}
};  // namespace puerhlab
