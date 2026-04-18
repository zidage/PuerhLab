//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstdint>
#include <memory>
#include <array>

#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace alcedo {
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
};  // namespace alcedo