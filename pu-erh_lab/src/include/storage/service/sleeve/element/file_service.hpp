#pragma once

#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "storage/mapper/sleeve/element/file_mapper.hpp"
#include "type/type.hpp"

namespace puerhlab {
class FileService {
  duckdb_connection _conn;
  FileMapper        _mapper;

  void              InsertFileParams(const FileMapperParams& param);

 public:
  explicit FileService(duckdb_connection conn) : _conn(conn), _mapper(_conn) {}

  auto ToParams(const SleeveFile& source) -> FileMapperParams;
  auto FromParams(const FileMapperParams&& param) -> std::shared_ptr<SleeveFile>;

  void InsertFile(const SleeveFile& element);

  auto GetFileById(const sl_element_id_t id) -> sl_element_id_t;
  auto GetBoundImageById(const sl_element_id_t id) -> image_id_t;

  void RemoveFileById(sl_element_id_t id);

  void UpdateFile(const SleeveFile& updated);
};
};  // namespace puerhlab