#pragma once

#include <duckdb.h>

#include <memory>

#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"

namespace puerhlab {
using namespace duckorm;
template <typename Mappable, typename ID>
class MapperInterface {
 private:
  duckdb_connection       &_conn;
  bool                     _is_connected;

  std::vector<std::string> _query_cache;

  virtual auto             FromDesc(std::vector<DuckFieldDesc> &&fields) -> std::shared_ptr<Mappable> = 0;
  virtual auto             ToDesc(const Mappable &obj) -> std::vector<DuckFieldDesc>;

 public:
  MapperInterface(duckdb_connection &conn) : _conn(conn) {}
  virtual void Insert(const Mappable &obj)                                             = 0;
  virtual void Remove(const ID remove_id)                                              = 0;
  virtual auto Get(const ID target_id) -> std::vector<std::shared_ptr<Mappable>>       = 0;
  virtual auto Get(const char *where_clause) -> std::vector<std::shared_ptr<Mappable>> = 0;
  virtual void Update(const ID target_id, const Mappable &updated)                     = 0;
};
};  // namespace puerhlab