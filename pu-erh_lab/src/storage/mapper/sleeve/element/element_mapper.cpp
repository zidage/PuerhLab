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

#include "storage/mapper/sleeve/element/element_mapper.hpp"

#include <cstdint>
#include <format>
#include <stdexcept>
#include <variant>

#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "type/type.hpp"

namespace puerhlab {
// struct ElementMapperParams {
//   sl_element_id_t id;
//   uint32_t        type;
//   const char*     element_name;
//   const char*     added_time;
//   const char*     modified_time;
//   uint32_t        ref_count;
// };
auto ElementMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> ElementMapperParams {
  if (data.size() != FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for SleeveElement");
  }
  auto id            = std::get_if<sl_element_id_t>(&data[0]);
  auto type          = std::get_if<uint32_t>(&data[1]);
  auto element_name  = std::get_if<std::unique_ptr<std::string>>(&data[2]);
  auto added_time    = std::get_if<std::unique_ptr<std::string>>(&data[3]);
  auto modified_time = std::get_if<std::unique_ptr<std::string>>(&data[4]);
  auto ref_count     = std::get_if<uint32_t>(&data[5]);

  if (id == nullptr || type == nullptr || element_name == nullptr || added_time == nullptr ||
      modified_time == nullptr || ref_count == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }
  return {
      *id,       *type, std::move(*element_name), std::move(*added_time), std::move(*modified_time),
      *ref_count};
}
};  // namespace puerhlab