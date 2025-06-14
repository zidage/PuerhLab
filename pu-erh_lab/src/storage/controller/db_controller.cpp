#include "storage/controller/db_controller.hpp"

#include <duckdb.h>

#include <stdexcept>

namespace puerhlab {
DBController::DBController(file_path_t& db_path) : _avail_conns(16), _db_path(db_path) {
  if (std::filesystem::exists(db_path)) {
    _initialized = true;
  }

  for (int i = 0; i < 8; ++i) {
    _avail_conns.push({});
  }
}

DBController::~DBController() { duckdb_close(&_db); }

auto DBController::GetConnectionGuard() -> ConnectionGuard {
  if (_avail_conns.empty()) {
    throw std::runtime_error(
        "DB Controller: Maximum number of connections reached; cannot initialize more connections");
  }
  ConnectionGuard guard{_avail_conns.pop()};

  if (duckdb_open(conv.to_bytes(_db_path.wstring()).c_str(), &_db) != DuckDBSuccess) {
    throw std::exception("DB cannot be created");
  }
  if (duckdb_connect(_db, &guard._conn) != DuckDBSuccess) {
    throw std::exception("DB cannot be connected");
  }

  return guard;
}

void DBController::InitializeDB() {
  auto guard = GetConnectionGuard();
  if (_initialized) return;

  duckdb_result result;
  if (duckdb_query(guard._conn, init_table_query, &result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&result);
    duckdb_destroy_result(&result);
    throw std::exception(error_message);
  }
  _initialized = true;
}

void DBController::ReturnConnectionGuard() { _avail_conns.push({}); }
};  // namespace puerhlab