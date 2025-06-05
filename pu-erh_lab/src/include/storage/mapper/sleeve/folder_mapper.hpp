#pragma once

#include <duckdb.h>

#include "mapper_interface.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct FolderMapperParams {
  sl_element_id_t _folder_id;
  sl_element_id_t _element_id;
};
class FolderMapper : MapperInterface<FolderMapperParams, sl_element_id_t> {
 private:
  auto FromDesc(std::vector<DuckFieldDesc>&& fields) -> FolderMapperParams;
  auto ToDesc() -> std::vector<DuckFieldDesc>;

 public:
  using MapperInterface::MapperInterface;

  void Insert(const FolderMapperParams params);
  auto Get(const sl_element_id_t id) -> std::vector<FolderMapperParams>;
  auto Get(const char* where_clause) -> std::vector<FolderMapperParams>;
  void Remove(const sl_element_id_t element_id);
  void Update(const sl_element_id_t element_id, const FolderMapperParams updated);
};
};  // namespace puerhlab