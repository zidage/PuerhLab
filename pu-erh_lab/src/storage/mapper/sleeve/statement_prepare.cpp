#include "storage/mapper/sleeve/statement_prepare.hpp"

#include "storage/mapper/sleeve/query_prepare.hpp"

namespace puerhlab {
void Prepare::RecycleResources() {
  if (_stmt) {
    duckdb_destroy_prepare(&_stmt);
  }
  duckdb_destroy_result(&_result);
}

Prepare::Prepare(duckdb_connection &con) : _stmt(), _con(con) { std::memset(&_result, 0, sizeof(_result)); }

Prepare::~Prepare() { RecycleResources(); }

auto Prepare::GetStmtGuard(const std::string &prepare_query) -> duckdb_prepared_statement & {
  if (duckdb_prepare(_con, prepare_query.c_str(), &_stmt) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("Prepare failed when inserting images");
  }
  return _stmt;
}

};  // namespace puerhlab