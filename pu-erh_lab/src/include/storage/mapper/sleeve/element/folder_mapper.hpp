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

#include <duckdb.h>

#include <array>

#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE FolderContent (folder_id BIGINT, element_id BIGINT);
struct FolderMapperParams {
  sl_element_id_t folder_id;
  sl_element_id_t element_id;
};

class FolderMapper : public MapperInterface<FolderMapper, FolderMapperParams, sl_element_id_t>,
                     public FieldReflectable<FolderMapper> {
 private:
  static constexpr uint32_t    _field_count                                      = 2;
  static constexpr const char* _table_name                                       = "FolderContent";
  static constexpr const char* _prime_key_clause                                 = "folder_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs = {
      FIELD(FolderMapperParams, folder_id, UINT32), FIELD(FolderMapperParams, element_id, UINT32)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> FolderMapperParams;
  friend struct FieldReflectable<FolderMapper>;
  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab