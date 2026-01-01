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
  sl_element_id_t              id_;
  uint32_t                     type_;
  std::unique_ptr<std::string> element_name_;
  std::unique_ptr<std::string> added_time_;
  std::unique_ptr<std::string> modified_time_;
  uint32_t                     ref_count_;
};
class ElementMapper : public MapperInterface<ElementMapper, ElementMapperParams, sl_element_id_t>,
                      public FieldReflectable<ElementMapper> {
 private:
  static constexpr uint32_t                                         field_count_      = 6;
  static constexpr const char*                                      table_name_       = "Element";
  static constexpr const char*                                      prime_key_clause_ = "id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_      = {
      FIELD(ElementMapperParams, id_, UINT32),
      FIELD(ElementMapperParams, type_, UINT32),
      FIELD(ElementMapperParams, element_name_, VARCHAR),
      FIELD(ElementMapperParams, added_time_, TIMESTAMP),
      FIELD(ElementMapperParams, modified_time_, TIMESTAMP),
      FIELD(ElementMapperParams, ref_count_, UINT32)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> ElementMapperParams;
  friend struct FieldReflectable<ElementMapper>;

  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab