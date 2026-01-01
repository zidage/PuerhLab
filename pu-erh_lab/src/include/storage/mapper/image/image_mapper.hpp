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
  image_id_t                   id_;
  std::unique_ptr<std::string> image_path_;
  std::unique_ptr<std::string> file_name_;
  uint32_t                     type_;
  std::unique_ptr<std::string> metadata_;
};

class ImageMapper : public MapperInterface<ImageMapper, ImageMapperParams, image_id_t>,
                    public FieldReflectable<ImageMapper> {
 private:
  static constexpr uint32_t                                         field_count_      = 5;
  static constexpr const char*                                      table_name_       = "Image";
  static constexpr const char*                                      prime_key_clause_ = "id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_      = {
      FIELD(ImageMapperParams, id_, UINT32), FIELD(ImageMapperParams, image_path_, VARCHAR),
      FIELD(ImageMapperParams, file_name_, VARCHAR), FIELD(ImageMapperParams, type_, UINT32),
      FIELD(ImageMapperParams, metadata_, VARCHAR)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> ImageMapperParams;
  friend struct FieldReflectable<ImageMapper>;
  using MapperInterface::MapperInterface;
};
}  // namespace puerhlab