#include "sleeve/sleeve_element/sleeve_element_factory.hpp"

#include <cstddef>
#include <memory>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"

namespace puerhlab {
std::shared_ptr<SleeveElement> SleeveElementFactory::CreateElement(const ElementType& type,
                                                                   uint32_t           id,
                                                                   std::wstring element_name) {
  std::shared_ptr<SleeveElement> new_element = nullptr;
  switch (type) {
    case ElementType::FILE:
      new_element = std::make_shared<SleeveFile>(id, element_name);
      break;
    case ElementType::FOLDER:
      new_element = std::make_shared<SleeveFolder>(id, element_name);
  }
  return new_element;
}
};  // namespace puerhlab