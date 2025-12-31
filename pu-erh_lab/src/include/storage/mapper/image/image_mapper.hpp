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

#include <cstdint>
#include <memory>

#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE Image (id BIGINT PRIMARY KEY, image_path TEXT, file_name TEXT, type INTEGER,
// metadata JSON);
struct ImageMapperParams {
  image_id_t                   id;
  std::unique_ptr<std::string> image_path;
  std::unique_ptr<std::string> file_name;
  uint32_t                     type;
  std::unique_ptr<std::string> metadata;
};

class ImageMapper : public MapperInterface<ImageMapper, ImageMapperParams, image_id_t>,
                    public FieldReflectable<ImageMapper> {
 private:
  static constexpr uint32_t                                         _field_count      = 5;
  static constexpr const char*                                      _table_name       = "Image";
  static constexpr const char*                                      _prime_key_clause = "id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs      = {
      FIELD(ImageMapperParams, id, UINT32), FIELD(ImageMapperParams, image_path, VARCHAR),
      FIELD(ImageMapperParams, file_name, VARCHAR), FIELD(ImageMapperParams, type, UINT32),
      FIELD(ImageMapperParams, metadata, VARCHAR)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> ImageMapperParams;
  friend struct FieldReflectable<ImageMapper>;
  using MapperInterface::MapperInterface;
};
}  // namespace puerhlab