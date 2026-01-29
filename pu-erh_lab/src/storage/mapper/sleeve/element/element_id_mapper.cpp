//  Copyright 2026 Yurun Zi
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

#include "storage/mapper/sleeve/element/element_id_mapper.hpp"

#include <stdexcept>

namespace puerhlab {
auto ElementIdMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> ElementIdMapperParams {
  if (data.size() != FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for ElementIdMapper");
  }

  auto id = std::get_if<sl_element_id_t>(&data[0]);
  if (id == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }

  return {*id};
}
}  // namespace puerhlab
