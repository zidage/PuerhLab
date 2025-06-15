#include "storage/controller/db_controller.hpp"

#include <duckdb.h>
#include <utf8.h>

#include <iterator>
#include <stdexcept>

#include "utf8/checked.h"

namespace puerhlab {
DBController::DBController(file_path_t& db_path) : _db_path(db_path), _initialized(false) {
  if (std::filesystem::exists(db_path)) {
    _initialized = true;
  }
}

DBController::~DBController() { duckdb_close(&_db); }

auto DBController::GetConnectionGuard() -> ConnectionGuard {
  ConnectionGuard guard{{}};

  if (duckdb_connect(_db, &guard._conn) != DuckDBSuccess) {
    throw std::exception("DB cannot be connected");
  }

  return guard;
}

void DBController::InitializeDB() {
  std::string  utf8_str;
  std::wstring path_wstr = _db_path.wstring();
  // TODO: Cross-platform
  utf8::utf16to8(path_wstr.begin(), path_wstr.end(), std::back_inserter(utf8_str));
  if (duckdb_open(utf8_str.c_str(), &_db) != DuckDBSuccess) {
    throw std::exception("DB cannot be created");
  }

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

};  // namespace puerhlab