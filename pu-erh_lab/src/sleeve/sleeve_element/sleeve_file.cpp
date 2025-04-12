#include "sleeve/sleeve_element/sleeve_file.hpp"

namespace puerhlab {
SleeveFile::SleeveFile(sl_element_id_t id, std::shared_ptr<Image> image) : SleeveElement(id), _image(image) {}
};  // namespace puerhlab