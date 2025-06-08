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
  static constexpr uint32_t    _field_count                                      = 3;
  static constexpr const char* _table_name                                       = "Filter";
  static constexpr const char* _prime_key_clause                                 = "combo_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs = {
      FIELD(FilterMapperParams, combo_id, UINT32), FIELD(FilterMapperParams, type, UINT32),
      FIELD(FilterMapperParams, data, VARCHAR)};

 public:
  using MapperInterface::MapperInterface;
  friend struct FieldReflectable<FilterMapper>;
};

// CREATE TABLE ComboFolder (combo_id BIGINT, folder_id BIGINT);
struct ComboMapperParams {
  uint32_t combo_id;
  uint32_t folder_id;
};
class ComboMapper : MapperInterface<ComboMapper, ComboMapperParams, sl_element_id_t>,
                    FieldReflectable<ComboMapper> {
 private:
  auto                      FromRawData(std::vector<duckorm::VarTypes>&& data) -> ComboMapperParams;
  static constexpr uint32_t _field_count                                         = 2;
  static constexpr const char* _table_name                                       = "ComboFolder";
  static constexpr const char* _prime_key_clause                                 = "combo_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs = {
      FIELD(ComboMapperParams, combo_id, UINT32), FIELD(ComboMapperParams, folder_id, UINT32)};

 public:
  friend struct FieldReflectable<ComboMapper>;
  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab