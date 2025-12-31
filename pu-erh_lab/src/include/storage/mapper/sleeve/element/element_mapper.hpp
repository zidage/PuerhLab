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
#include <memory>

#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// Element (id BIGINT PRIMARY KEY, type INTEGER, element_name TEXT, added_time "
//     "TIMESTAMP, modified_time "
//     "TIMESTAMP, "
//     "ref_count BIGINT);"
struct ElementMapperParams {
  sl_element_id_t              id;
  uint32_t                     type;
  std::unique_ptr<std::string> element_name;
  std::unique_ptr<std::string> added_time;
  std::unique_ptr<std::string> modified_time;
  uint32_t                     ref_count;
};
class ElementMapper : public MapperInterface<ElementMapper, ElementMapperParams, sl_element_id_t>,
                      public FieldReflectable<ElementMapper> {
 private:
  static constexpr uint32_t                                         _field_count      = 6;
  static constexpr const char*                                      _table_name       = "Element";
  static constexpr const char*                                      _prime_key_clause = "id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs      = {
      FIELD(ElementMapperParams, id, UINT32),
      FIELD(ElementMapperParams, type, UINT32),
      FIELD(ElementMapperParams, element_name, VARCHAR),
      FIELD(ElementMapperParams, added_time, TIMESTAMP),
      FIELD(ElementMapperParams, modified_time, TIMESTAMP),
      FIELD(ElementMapperParams, ref_count, UINT32)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> ElementMapperParams;
  friend struct FieldReflectable<ElementMapper>;

  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab