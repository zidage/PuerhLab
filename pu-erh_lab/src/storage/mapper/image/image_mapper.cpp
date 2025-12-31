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

#include "storage/mapper/image/image_mapper.hpp"

#include <cstdint>
#include <memory>
#include <variant>

// struct ImageParams {
//   image_id_t  id;
//   const char* image_path;
//   const char* file_name;
//   uint32_t    type;
//   const char* metadata;
// };

namespace puerhlab {
auto ImageMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> ImageMapperParams {
  if (data.size() != FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for Image");
  }
  auto id         = std::get_if<sl_element_id_t>(&data[0]);
  auto image_path = std::get_if<std::unique_ptr<std::string>>(&data[1]);
  auto file_name  = std::get_if<std::unique_ptr<std::string>>(&data[2]);
  auto type       = std::get_if<uint32_t>(&data[3]);
  auto metadata   = std::get_if<std::unique_ptr<std::string>>(&data[4]);

  if (id == nullptr || image_path == nullptr || file_name == nullptr || type == nullptr ||
      metadata == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }
  return {*id, std::move(*image_path), std::move(*file_name), *type, std::move(*metadata)};
}
};  // namespace puerhlab