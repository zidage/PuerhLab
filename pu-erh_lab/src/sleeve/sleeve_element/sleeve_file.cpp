#include "sleeve/sleeve_element/sleeve_file.hpp"

#include <cstdint>
#include <memory>

#include "sleeve/sleeve_element/sleeve_element.hpp"

namespace puerhlab {
SleeveFile::~SleeveFile() {}
SleeveFile::SleeveFile(sl_element_id_t id, file_name_t element_name) : SleeveElement(id, element_name) {
  _type = ElementType::FILE;
}
SleeveFile::SleeveFile(sl_element_id_t id, file_name_t element_name, std::shared_ptr<Image> image)
    : SleeveElement(id, element_name) {
  _image = image;
  _type  = ElementType::FILE;
}

auto SleeveFile::Clear() -> bool {
  // FIXME: Add implementation
  return true;
}

auto SleeveFile::Copy(uint32_t new_id) const -> std::shared_ptr<SleeveElement> {
  std::shared_ptr<SleeveFile> new_file = std::make_shared<SleeveFile>(new_id, _element_name);
  new_file->_edit_history              = _edit_history;
  // TODO: Update the current_version pointer once finish implementing edit history module
  new_file->_current_version           = nullptr;
  // The image object is still reused
  new_file->_image                     = _image;
  return new_file;
}

auto SleeveFile::GetImage() -> std::shared_ptr<Image> { return _image; }

void SleeveFile::SetImage(const std::shared_ptr<Image> img) { _image = img; }
};  // namespace puerhlab
