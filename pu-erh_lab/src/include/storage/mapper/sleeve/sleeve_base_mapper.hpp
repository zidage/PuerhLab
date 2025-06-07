#pragma once

#include <duckdb.h>

#include <memory>

#include "file_mapper.hpp"
#include "folder_mapper.hpp"
#include "mapper_interface.hpp"
#include "sleeve/sleeve_base.hpp"
#include "sleeve_mapper.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct SleeveBaseMapperParams {
  sleeve_id_t sleeve_id;
};
class SleeveBaseMapper : MapperInterface<SleeveBaseMapperParams, sleeve_id_t> {
 private:
  auto FromRawData(std::vector<VarTypes>&& data) -> SleeveBaseMapperParams;
  auto ToDesc() -> std::vector<DuckFieldDesc>;

 public:
  SleeveBaseMapper(duckdb_connection& conn) : MapperInterface(conn) {};

  void CaputureSleeveBase(std::shared_ptr<SleeveBase> base);

  void Insert(const SleeveBaseMapperParams params);
  auto Get(const sleeve_id_t id) -> std::vector<SleeveBaseMapperParams>;
  auto Get(const char* where_clause) -> std::vector<SleeveBaseMapperParams>;
  void Remove(const sleeve_id_t sleeve_id);
  void Update(const sleeve_id_t sleeve_id, const SleeveBaseMapperParams updated);
};
};  // namespace puerhlab