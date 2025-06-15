#include "storage/mapper/duckorm/duckdb_types.hpp"

#include <duckdb.h>

#include <exception>

namespace duckorm {
void PreparedStatement::RecycleResources() {
  if (_stmt) {
    duckdb_destroy_prepare(&_stmt);
    _stmt = nullptr;
  }
  if (_result.deprecated_columns) duckdb_destroy_result(&_result);
}

PreparedStatement::PreparedStatement(duckdb_connection& con) : _stmt(), _con(con) {
  std::memset(&_result, 0, sizeof(_result));
}

PreparedStatement::PreparedStatement(duckdb_connection& con, const std::string& prepare_query)
    : _stmt(), _con(con) {
  std::memset(&_result, 0, sizeof(_result));
  // std::memset(&_stmt, 0, sizeof(_stmt));

  if (duckdb_prepare(_con, prepare_query.c_str(), &_stmt) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("PreparedStatement failed when inserting images");
  }
  _prepared = true;
}

PreparedStatement::~PreparedStatement() { RecycleResources(); }

auto PreparedStatement::GetStmtGuard(const std::string& prepare_query)
    -> duckdb_prepared_statement& {
  if (duckdb_prepare(_con, prepare_query.c_str(), &_stmt) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("PreparedStatement failed when inserting images");
  }
  _prepared = true;
  return _stmt;
}

void PreparedStatement::SetConnection(duckdb_connection& con) { _con = con; }
}  // namespace duckorm
