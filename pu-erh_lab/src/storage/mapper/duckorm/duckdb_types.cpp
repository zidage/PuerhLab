#include "storage/mapper/duckorm/duckdb_types.hpp"

namespace duckorm {
void StatementPrepare::RecycleResources() {
  if (_stmt) {
    duckdb_destroy_prepare(&_stmt);
  }
  duckdb_destroy_result(&_result);
}

StatementPrepare::StatementPrepare(duckdb_connection& con) : _stmt(), _con(con) {
  std::memset(&_result, 0, sizeof(_result));
}

StatementPrepare::~StatementPrepare() { RecycleResources(); }

auto StatementPrepare::GetStmtGuard(const std::string& prepare_query)
    -> duckdb_prepared_statement& {
  if (duckdb_prepare(_con, prepare_query.c_str(), &_stmt) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("StatementPrepare failed when inserting images");
  }
  _prepared = true;
  return _stmt;
}

void StatementPrepare::SetConnection(duckdb_connection& con) { _con = con; }
}  // namespace duckorm
