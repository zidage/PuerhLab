//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <array>

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
  static constexpr uint32_t    field_count_                                     = 2;
  static constexpr const char* table_name_                                      = "EditHistory";
  static constexpr const char* prime_key_clause_                                = "file_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_ = {
      FIELD(EditHistoryMapperParams, file_id, UINT32),
      FIELD(EditHistoryMapperParams, history, VARCHAR)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> EditHistoryMapperParams;
  friend struct FieldReflectable<EditHistoryMapper>;
  using MapperInterface::MapperInterface;
};

};  // namespace puerhlab