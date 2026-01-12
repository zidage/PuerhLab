//  Copyright 2026 Yurun Zi
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
#include <array>

#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE PipelineParam (file_id BIGINT PRIMARY KEY, param_json JSON);
struct PipelineMapperParams {
  sl_element_id_t              file_id;
  std::unique_ptr<std::string> param_json;
};

class PipelineMapper
    : public MapperInterface<PipelineMapper, PipelineMapperParams, sl_element_id_t>,
      public FieldReflectable<PipelineMapper> {
 private:
  static constexpr uint32_t    field_count_                                      = 2;
  static constexpr const char* table_name_                                       = "PipelineParam";
  static constexpr const char* prime_key_clause_                                 = "file_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_ = {
      FIELD(PipelineMapperParams, file_id, UINT32),
      FIELD(PipelineMapperParams, param_json, VARCHAR)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> PipelineMapperParams;
  friend struct FieldReflectable<PipelineMapper>;
  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab