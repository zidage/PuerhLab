#pragma once

#include <array>

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
class FileMapper : MapperInterface<FileMapperParams, sl_element_id_t>,
                   FieldReflectable<FileMapper> {
 private:
  auto FromRawData(std::vector<VarTypes>&& data) -> FileMapperParams;

  static constexpr std::array<duckorm::DuckFieldDesc, 2> kFieldDescs = {
      FIELD(FileMapperParams, file_id, UINT32), FIELD(FileMapperParams, image_id, UINT32)};

 public:
  friend struct FieldReflectable<FileMapper>;
  using MapperInterface<FileMapperParams, sl_element_id_t>::MapperInterface;

  void Insert(const FileMapperParams params);
  auto Get(const sl_element_id_t id) -> std::vector<FileMapperParams>;
  auto Get(const char* where_clause) -> std::vector<FileMapperParams>;
  void Remove(const sl_element_id_t element_id);
  void Update(const sl_element_id_t element_id, const FileMapperParams updated);
};
};  // namespace puerhlab