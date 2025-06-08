#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct SleeveElementParams {
  sl_element_id_t _element_id;
  uint32_t        _type;
  const char*     _element_name;
  const char*     _added_time;
  const char*     _modified_time;
  uint32_t        _ref_count;
};
class ElementMapper : MapperInterface<SleeveElementParams, sl_element_id_t>,
                      FieldReflectable<ElementMapper> {
 private:
  auto FromRawData(std::vector<VarTypes>&& data) -> SleeveElementParams;

  static constexpr std::array<duckorm::DuckFieldDesc, 6> kFieldDescs = {
      FIELD(SleeveElementParams, _element_id, UINT32),
      FIELD(SleeveElementParams, _type, UINT32),
      FIELD(SleeveElementParams, _element_name, VARCHAR),
      FIELD(SleeveElementParams, _added_time, TIMESTAMP),
      FIELD(SleeveElementParams, _modified_time, TIMESTAMP),
      FIELD(SleeveElementParams, _ref_count, UINT32)};

 public:
  using MapperInterface<SleeveElementParams, sl_element_id_t>::MapperInterface;

  void Insert(const SleeveElementParams element);
  auto Get(const sl_element_id_t id) -> std::vector<SleeveElementParams>;
  auto Get(const char* where_clause) -> std::vector<SleeveElementParams>;
  void Remove(const sl_element_id_t element_id);
  void Update(const sl_element_id_t element_id, const SleeveElementParams);
};
};  // namespace puerhlab