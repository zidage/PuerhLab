//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <array>

#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE Sleeve (id BIGINT PRIMARY KEY);
struct BaseMapperParams {
  sleeve_id_t id;
};
class BaseMapper : public MapperInterface<BaseMapper, BaseMapperParams, sleeve_id_t>,
                   public FieldReflectable<BaseMapper> {
 private:
  static constexpr uint32_t                                         field_count_     = 1;
  static constexpr const char*                                      table_name_       = "Sleeve";
  static constexpr const char*                                      prime_key_clause_ = "id={}";

  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_     = {
      FIELD(BaseMapperParams, id, UINT32)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> BaseMapperParams;
  friend struct FieldReflectable<BaseMapper>;
  using MapperInterface::MapperInterface;
};

// CREATE TABLE SleeveRoot (id BIGINT PRIMARY KEY);"
struct RootMapperParams {
  sl_element_id_t id_;
};
class RootMapper : MapperInterface<RootMapper, RootMapperParams, sl_element_id_t>,
                   FieldReflectable<RootMapper> {
 private:
  static constexpr uint32_t                                         field_count_ = 1;
  static constexpr const char*                                      table_name_  = "SleeveRoot";
  static constexpr const char*                                      prime_key_clause_ = "id={}";

  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_      = {
      FIELD(RootMapperParams, id_, UINT32)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> RootMapperParams;
  friend struct FieldReflectable<RootMapper>;
  friend class MapperInterface<RootMapper, RootMapperParams, sl_element_id_t>;
  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab