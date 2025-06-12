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

class FolderMapper : MapperInterface<FolderMapper, FolderMapperParams, sl_element_id_t>,
                     FieldReflectable<FolderMapper> {
 private:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> FolderMapperParams;

  static constexpr uint32_t    _field_count                                      = 2;
  static constexpr const char* _table_name                                       = "FolderContent";
  static constexpr const char* _prime_key_clause                                 = "folder_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs = {
      FIELD(FolderMapperParams, folder_id, UINT32), FIELD(FolderMapperParams, element_id, UINT32)};

 public:
  friend struct FieldReflectable<FolderMapper>;
  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab