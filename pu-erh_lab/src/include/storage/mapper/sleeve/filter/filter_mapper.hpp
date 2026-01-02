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

#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE Filter (combo_id BIGINT, type INTEGER, data JSON);
struct FilterMapperParams {
  uint32_t    combo_id;
  uint32_t    type;
  const char* data;
};
class FilterMapper : MapperInterface<FilterMapper, FilterMapperParams, sl_element_id_t>,
                     FieldReflectable<FilterMapper> {
 private:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> FilterMapperParams;
  static constexpr uint32_t    field_count_                                      = 3;
  static constexpr const char* table_name_                                       = "Filter";
  static constexpr const char* prime_key_clause_                                 = "combo_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_ = {
      FIELD(FilterMapperParams, combo_id, UINT32), FIELD(FilterMapperParams, type, UINT32),
      FIELD(FilterMapperParams, data, VARCHAR)};

 public:
  using MapperInterface::MapperInterface;
  friend struct FieldReflectable<FilterMapper>;
};

// CREATE TABLE ComboFolder (combo_id BIGINT, folder_id BIGINT);
struct ComboMapperParams {
  uint32_t combo_id_;
  uint32_t folder_id_;
};
class ComboMapper : MapperInterface<ComboMapper, ComboMapperParams, sl_element_id_t>,
                    FieldReflectable<ComboMapper> {
 private:
  static constexpr uint32_t    field_count_                                      = 2;
  static constexpr const char* table_name_                                       = "ComboFolder";
  static constexpr const char* prime_key_clause_                                 = "combo_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_ = {
      FIELD(ComboMapperParams, combo_id_, UINT32), FIELD(ComboMapperParams, folder_id_, UINT32)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> ComboMapperParams;
  friend struct FieldReflectable<ComboMapper>;
  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab