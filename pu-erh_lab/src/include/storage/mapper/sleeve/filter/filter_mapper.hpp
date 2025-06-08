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
class FilterMapper : MapperInterface<FilterMapperParams, sl_element_id_t>,
                     FieldReflectable<FilterMapper> {
 private:
  auto FromRawData(std::vector<VarTypes>&& data) -> FilterMapperParams;

  static constexpr std::array<DuckFieldDesc, 3> kFieldDescs = {
      FIELD(FilterMapperParams, combo_id, UINT32), FIELD(FilterMapperParams, type, UINT32),
      FIELD(FilterMapperParams, data, VARCHAR)};

 public:
  friend struct FieldReflectable<FilterMapper>;
  using MapperInterface<FilterMapperParams, sl_element_id_t>::MapperInterface;

  void Insert(const FilterMapperParams params);
  auto Get(const sl_element_id_t id) -> std::vector<FilterMapperParams>;
  auto Get(const char* where_clause) -> std::vector<FilterMapperParams>;
  void Remove(const sl_element_id_t element_id);
  void Update(const sl_element_id_t element_id, const FilterMapperParams updated);
};

struct ComboMapperParams {
  uint32_t combo_id;
  uint32_t folder_id;
};
class ComboMapper : MapperInterface<ComboMapperParams, sl_element_id_t>,
                    FieldReflectable<ComboMapper> {
 private:
  auto FromRawData(std::vector<VarTypes>&& data) -> ComboMapperParams;

  static constexpr std::array<DuckFieldDesc, 2> kFieldDescs = {
      FIELD(ComboMapperParams, combo_id, UINT32), FIELD(ComboMapperParams, folder_id, UINT32)};

 public:
  friend struct FieldReflectable<ComboMapper>;
  using MapperInterface::MapperInterface;

  void Insert(const ComboMapperParams params);
  auto Get(const sl_element_id_t id) -> std::vector<ComboMapperParams>;
  auto Get(const char* where_clause) -> std::vector<ComboMapperParams>;
  void Remove(const sl_element_id_t element_id);
  void Update(const sl_element_id_t element_id, const ComboMapperParams updated);
};
};  // namespace puerhlab