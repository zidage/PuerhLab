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
                      FieldReflectable<ElementMapper> {
 private:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> ElementMapperParams;

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
  friend struct FieldReflectable<ElementMapper>;

  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab