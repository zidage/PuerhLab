#pragma once

#include <duckdb.h>

#include <array>

#include "mapper_interface.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct FileMapperParams {
  sl_element_id_t _file_id;
  image_id_t      _img_id;
};
class FileMapper : MapperInterface<FileMapperParams, sl_element_id_t>,
                   FieldReflectable<FileMapper> {
 private:
  auto FromDesc(std::vector<DuckFieldDesc>&& fields) -> FileMapperParams;

  static constexpr std::array<DuckFieldDesc, 2> kFieldDescs = {
      FIELD(FileMapperParams, _file_id, UINT32), FIELD(FileMapperParams, _img_id, UINT32)};

 public:
  using MapperInterface::MapperInterface;

  void Insert(const FileMapperParams params);
  auto Get(const sl_element_id_t id) -> std::vector<FileMapperParams>;
  auto Get(const char* where_clause) -> std::vector<FileMapperParams>;
  void Remove(const sl_element_id_t element_id);
  void Update(const sl_element_id_t element_id, const FileMapperParams updated);
};
};  // namespace puerhlab