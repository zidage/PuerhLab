#pragma once

#include <duckdb.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>

namespace duckorm {
enum class DuckDBType : uint8_t {
  INT32,
  INT64,
  UINT32,
  UINT64,
  DOUBLE,
  VARCHAR,
  JSON,
  BOOLEAN,
  TIMESTAMP,
};

class StatementPrepare {
 private:
  void RecycleResources();

 public:
  duckdb_result             _result;
  duckdb_prepared_statement _stmt;
  duckdb_connection&        _con;

  bool                      _prepared = false;
  StatementPrepare(duckdb_connection& con);
  StatementPrepare();
  ~StatementPrepare();
  auto GetStmtGuard(const std::string& prepare_query) -> duckdb_prepared_statement&;
  void SetConnection(duckdb_connection& con);
};

struct DuckFieldDesc {
  const char* name;
  DuckDBType  type;
  size_t      offset;
};

#define FIELD(type, field, field_type) \
  duckorm::DuckFieldDesc { #field, duckorm::DuckDBType::field_type, offsetof(type, field) }

using VarTypes =
    std::variant<int32_t, int64_t, uint32_t, uint64_t, double, std::unique_ptr<std::string>>;
};  // namespace duckorm