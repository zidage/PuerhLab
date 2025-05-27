#include <duckdb.h>

#include <string>

#pragma once
namespace puerhlab {
class Prepare {
 private:
  void RecycleResources();

 public:
  duckdb_result             _result;
  duckdb_prepared_statement _stmt;
  duckdb_connection        &_con;
  Prepare(duckdb_connection &con);
  ~Prepare();
  auto GetStmtGuard(const std::string &prepare_query) -> duckdb_prepared_statement &;
};
};  // namespace puerhlab