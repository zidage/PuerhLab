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

#include <array>
#include <cstdint>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE FileImage (file_id BIGINT, image_id BIGINT);
struct FileMapperParams {
  sl_element_id_t file_id;
  image_id_t      image_id;
};
class FileMapper : public MapperInterface<FileMapper, FileMapperParams, sl_element_id_t>,
                   public FieldReflectable<FileMapper> {
 private:
  static constexpr uint32_t    field_count_                                      = 2;
  static constexpr const char* table_name_                                       = "FileImage";
  static constexpr const char* prime_key_clause_                                 = "file_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_ = {
      FIELD(FileMapperParams, file_id, UINT32), FIELD(FileMapperParams, image_id, UINT32)};
 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> FileMapperParams;
  friend struct FieldReflectable<FileMapper>;
  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab