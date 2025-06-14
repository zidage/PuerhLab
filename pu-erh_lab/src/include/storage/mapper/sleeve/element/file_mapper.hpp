#pragma once

#include <array>
#include <cstdint>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE FileImage (file_id BIGINT, image_id BIGINT);
struct FileMapperParams {
  sl_element_id_t file_id;
  image_id_t      image_id;
};
class FileMapper : public MapperInterface<FileMapper, FileMapperParams, sl_element_id_t>,
                   public FieldReflectable<FileMapper> {
 private:
  static constexpr uint32_t    _field_count                                      = 2;
  static constexpr const char* _table_name                                       = "FileImage";
  static constexpr const char* _prime_key_clause                                 = "file_id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs = {
      FIELD(FileMapperParams, file_id, UINT32), FIELD(FileMapperParams, image_id, UINT32)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> FileMapperParams;
  friend struct FieldReflectable<FileMapper>;
  using MapperInterface::MapperInterface;
};
};  // namespace puerhlab