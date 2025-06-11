#pragma once

#include <vector>

#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/mapper/sleeve/element/folder_mapper.hpp"
#include "type/type.hpp"

namespace puerhlab {

class FolderService {
  duckdb_connection _conn;
  FolderMapper      _mapper;

  void              InsertFileParams(const FolderMapperParams& param);

 public:
  explicit FolderService(duckdb_connection conn) : _conn(conn), _mapper(_conn) {}

  auto ToParams(const SleeveFolder& source) -> FolderMapperParams;
  auto FromParams(const FolderMapperParams&& param) -> std::shared_ptr<SleeveFolder>;

  void InsertFolder(const SleeveFolder& element);

  auto GetFolderById(const sl_element_id_t id) -> sl_element_id_t;
  auto GetContentById(const sl_element_id_t id) -> std::vector<sl_element_id_t>;

  void RemoveFolderByParentId(sl_element_id_t id);
  void RemoveFolderContentById(sl_element_id_t id);

  void UpdateFolder(const SleeveFolder& updated);
};
};  // namespace puerhlab