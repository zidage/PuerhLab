//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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