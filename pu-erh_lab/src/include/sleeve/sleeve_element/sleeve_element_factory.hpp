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

#pragma once

#include <memory>

#include "sleeve_element.hpp"
#include "sleeve_file.hpp"
#include "sleeve_folder.hpp"

namespace puerhlab {
class SleeveElementFactory {
 public:
  static std::shared_ptr<SleeveElement> CreateElement(const ElementType& type, sl_element_id_t id,
                                                      file_name_t element_name);
};
};  // namespace puerhlab