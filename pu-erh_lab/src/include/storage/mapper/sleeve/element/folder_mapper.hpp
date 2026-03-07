//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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
  static constexpr uint32_t    field_count_                                      = 2;
  static constexpr const char* table_name_                                       = "FolderContent";
  static constexpr const char* prime_key_clause_                                 = "folder_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_ = {
      FIELD(FolderMapperParams, folder_id, UINT32), FIELD(FolderMapperParams, element_id, UINT32)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> FolderMapperParams;
  friend struct FieldReflectable<FolderMapper>;
  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab