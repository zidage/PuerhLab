#pragma once

#include <duckdb.h>

#include <array>

#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE FolderContent (folder_id BIGINT, element_name TEXT, element_id BIGINT);
struct FolderMapperParams {
  sl_element_id_t folder_id;
  const char*     element_name;
  sl_element_id_t element_id;
};

class FolderMapper : MapperInterface<FolderMapperParams, sl_element_id_t>,
                     FieldReflectable<FolderMapper> {
 private:
  auto FromRawData(std::vector<VarTypes>&& data) -> FolderMapperParams;

  static constexpr std::array<duckorm::DuckFieldDesc, 3> kFieldDescs = {
      FIELD(FolderMapperParams, folder_id, UINT32),
      FIELD(FolderMapperParams, element_name, VARCHAR),
      FIELD(FolderMapperParams, element_id, UINT32)};

 public:
  friend struct FieldReflectable<FolderMapper>;
  using MapperInterface<FolderMapperParams, sl_element_id_t>::MapperInterface;

  void Insert(const FolderMapperParams params);
  auto Get(const sl_element_id_t id) -> std::vector<FolderMapperParams>;
  auto Get(const char* where_clause) -> std::vector<FolderMapperParams>;
  void Remove(const sl_element_id_t id);
  void Update(const sl_element_id_t id, const FolderMapperParams updated);
};
};  // namespace puerhlab