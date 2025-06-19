#include "storage/controller/db_controller.hpp"

#include <duckdb.h>
#include <utf8.h>

#include <iterator>
#include <stdexcept>

#include "utf8/checked.h"
#include "utils/string/convert.hpp"

namespace puerhlab {
/**
 * @brief Construct a new DBController::DBController object
 *
 * @param db_path
 */
DBController::DBController(file_path_t& db_path) : _db_path(db_path), _initialized(false) {
  if (std::filesystem::exists(db_path)) {
    _initialized = true;
  }
  InitializeDB();
}

/**
 * @brief Destroy the DBController::DBController object
 *
 */
DBController::~DBController() { duckdb_close(&_db); }

/**
 * @brief Get a connection guard for the database.
 *
 * @return ConnectionGuard
 */
auto DBController::GetConnectionGuard() -> ConnectionGuard {
  ConnectionGuard guard{{}};

  if (duckdb_connect(_db, &guard._conn) != DuckDBSuccess) {
    throw std::exception("DB cannot be connected");
  }

  return guard;
}

/**
 * @brief Initialize the database by creating necessary tables.
 *
 */
void DBController::InitializeDB() {
  // SQL query to create the necessary tables
  std::string utf8_str = conv::ToBytes(_db_path.wstring());
  if (duckdb_open(utf8_str.c_str(), &_db) != DuckDBSuccess) {
    throw std::exception("DB cannot be created");
  }

  // SQL query to create the tables
  auto guard = GetConnectionGuard();
  if (_initialized) return;

  duckdb_result result;

  // Run the SQL query to create the tables
  if (duckdb_query(guard._conn, init_table_query, &result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&result);
    duckdb_destroy_result(&result);
    throw std::exception(error_message);
  }
  _initialized = true;
}

};  // namespace puerhlab