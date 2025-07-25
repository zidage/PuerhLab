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

/**
 * @brief A RAII class for managing prepared statements in DuckDB.
 *
 */
class PreparedStatement {
 private:
  void RecycleResources();

 public:
  duckdb_result             _result;
  duckdb_prepared_statement _stmt;
  duckdb_connection&        _con;

  bool                      _prepared = false;
  PreparedStatement(duckdb_connection& con);
  PreparedStatement(duckdb_connection& con, const std::string& prepare_query);
  PreparedStatement();
  ~PreparedStatement();
  auto GetStmtGuard(const std::string& prepare_query) -> duckdb_prepared_statement&;
  void SetConnection(duckdb_connection& con);
};

/**
 * @brief Field descriptor for DuckDB ORM.
 *
 */
struct DuckFieldDesc {
  const char* name;
  DuckDBType  type;
  size_t      offset;
};

// Macro to define a field descriptor for a specific type and field.
#define FIELD(type, field, field_type) \
  duckorm::DuckFieldDesc { #field, duckorm::DuckDBType::field_type, offsetof(type, field) }

// brief Type alias for a variant that can hold various DuckDB-supported types.
using VarTypes =
    std::variant<int32_t, int64_t, uint32_t, uint64_t, double, std::unique_ptr<std::string>>;
};  // namespace duckorm