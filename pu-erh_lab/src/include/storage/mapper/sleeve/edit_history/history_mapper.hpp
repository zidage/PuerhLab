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

#include "image/image.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE EditHistory (file_id PRIMARY KEY BIGINT, history JSON);
struct EditHistoryMapperParams {
  sl_element_id_t              file_id;
  std::unique_ptr<std::string> history;
};

class EditHistoryMapper
    : public MapperInterface<EditHistoryMapper, EditHistoryMapperParams, sl_element_id_t>,
      public FieldReflectable<EditHistoryMapper> {
 private:
  static constexpr uint32_t    _field_count                                      = 2;
  static constexpr const char* _table_name                                       = "EditHistory";
  static constexpr const char* _prime_key_clause                                 = "file_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs = {
      FIELD(EditHistoryMapperParams, file_id, UINT32),
      FIELD(EditHistoryMapperParams, history, VARCHAR)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> EditHistoryMapperParams;
  friend struct FieldReflectable<EditHistoryMapper>;
  using MapperInterface::MapperInterface;
};

};  // namespace puerhlab