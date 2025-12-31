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

#include "storage/mapper/sleeve/element/file_mapper.hpp"

#include <duckdb.h>

#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "type/type.hpp"

namespace puerhlab {
auto FileMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> FileMapperParams {
  if (data.size() != FileMapper::FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for SleeveFile");
  }

  auto file_id = std::get_if<sl_element_id_t>(&data[0]);
  auto img_id  = std::get_if<image_id_t>(&data[1]);
  if (file_id == nullptr || img_id == nullptr) {
    throw std::runtime_error("Unmatching types occured when parsing the data from the DB");
  }
  return {*file_id, *img_id};
}
};  // namespace puerhlab