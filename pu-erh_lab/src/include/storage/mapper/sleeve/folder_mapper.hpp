#pragma once

#include <duckdb.h>

#include "mapper_interface.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"

namespace puerhlab {
class FolderMapper : MapperInterface<SleeveFolder, sl_element_id_t> {
 private:
  auto FromDesc(std::vector<DuckFieldDesc> &&fields) -> std::shared_ptr<SleeveFolder>;
  auto ToDesc(const SleeveFolder &file) -> std::vector<DuckFieldDesc>;

 public:
  using MapperInterface::MapperInterface;

  void Insert(const SleeveFolder &folder);
  auto Get(const sl_element_id_t id) -> std::vector<std::shared_ptr<SleeveFolder>>;
  auto Get(const char *where_clause) -> std::vector<std::shared_ptr<SleeveFolder>>;
  void Remove(const sl_element_id_t element_id);
  void Update(const sl_element_id_t element_id, const SleeveFolder &updated);
};
};  // namespace puerhlab