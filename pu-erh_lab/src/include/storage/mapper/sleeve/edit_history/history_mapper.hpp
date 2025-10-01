#pragma once

#include <array>

#include "image/image.hpp"
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
  static constexpr uint32_t    _field_count                                      = 2;
  static constexpr const char* _table_name                                       = "EditHistory";
  static constexpr const char* _prime_key_clause                                 = "file_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs = {
      FIELD(EditHistoryMapperParams, file_id, UINT32),
      FIELD(EditHistoryMapperParams, history, VARCHAR)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> EditHistoryMapperParams;
  friend struct FieldReflectable<EditHistoryMapper>;
  using MapperInterface::MapperInterface;
};

};  // namespace puerhlab