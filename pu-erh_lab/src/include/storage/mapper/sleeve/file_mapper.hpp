#pragma once

#include <duckdb.h>

#include "mapper_interface.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "type/type.hpp"

namespace puerhlab {
class FileMapper : MapperInterface<SleeveFile, sl_element_id_t> {
 private:
  auto FromDesc(std::vector<DuckFieldDesc> &&fields) -> std::shared_ptr<SleeveFile>;
  auto ToDesc(const SleeveFile &file) -> std::vector<DuckFieldDesc>;

 public:
  using MapperInterface::MapperInterface;

  void Insert(const SleeveFile &file);
  auto Get(const sl_element_id_t id) -> std::vector<std::shared_ptr<SleeveFile>>;
  auto Get(const char *where_clause) -> std::vector<std::shared_ptr<SleeveFile>>;
  void Remove(const sl_element_id_t element_id);
  void Update(const sl_element_id_t element_id, const SleeveFile &updated);
};
};  // namespace puerhlab