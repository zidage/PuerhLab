#pragma once

#include <duckdb.h>

#include <memory>
#include <span>

#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"

namespace puerhlab {
using namespace duckorm;
template <typename Mappable, typename ID>
class MapperInterface {
 private:
  duckdb_connection&       _conn;
  bool                     _is_connected;

  std::vector<std::string> _query_cache;

  virtual auto             FromRawData(std::vector<VarTypes>&& data) -> Mappable = 0;

 public:
  MapperInterface(duckdb_connection& conn) : _conn(conn) {}
  virtual void Insert(const Mappable obj)                             = 0;
  virtual void Remove(const ID remove_id)                             = 0;
  virtual auto Get(const ID target_id) -> std::vector<Mappable>       = 0;
  virtual auto Get(const char* where_clause) -> std::vector<Mappable> = 0;
  virtual void Update(const ID target_id, const Mappable updated)     = 0;
};

// Don't understand what heck this is... They call it CRTP (C++ Recurring Tremendous Pain, maybe).
template <typename Derived>
struct FieldReflectable {
  using FieldArrayType = std::span<const DuckFieldDesc>;
  static constexpr FieldArrayType GetDesc() { return Derived::kFieldDescs; }
};
};  // namespace puerhlab