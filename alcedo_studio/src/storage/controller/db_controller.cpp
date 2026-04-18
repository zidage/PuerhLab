//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "storage/controller/db_controller.hpp"

#include <duckdb.h>
#include <utf8.h>

#include <iterator>
#include <stdexcept>

#include "utf8/checked.h"
#include "utils/string/convert.hpp"

namespace alcedo {
/**
 * @brief Construct a new DBController::DBController object
 *
 * @param db_path
 */
DBController::DBController(file_path_t& db_path) : db_path_(db_path), initialized_(false) {
  if (std::filesystem::exists(db_path)) {
    initialized_ = true;
  }
  InitializeDB();
}

/**
 * @brief Destroy the DBController::DBController object
 *
 */
DBController::~DBController() { duckdb_close(&db_); }

/**
 * @brief Get a connection guard for the database.
 *
 * @return ConnectionGuard
 */
auto DBController::GetConnectionGuard() -> ConnectionGuard {
  ConnectionGuard guard{{}};

  if (duckdb_connect(db_, &guard.conn_) != DuckDBSuccess) {
    throw std::runtime_error("DB cannot be connected");
  }

  return guard;
}

/**
 * @brief Initialize the database by creating necessary tables.
 *
 */
void DBController::InitializeDB() {
  // SQL query to create the necessary tables
  
  std::string utf8_str = conv::ToBytes(db_path_.wstring());
  auto        state    = duckdb_open(utf8_str.c_str(), &db_);
  if (state != DuckDBSuccess) {
    throw std::runtime_error("DB cannot be opened or created" );
  }

  // SQL query to create the tables
  auto guard = GetConnectionGuard();
  if (initialized_) return;

  duckdb_result result;

  // Run the SQL query to create the tables
  if (duckdb_query(guard.conn_, init_table_query, &result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&result);
    duckdb_destroy_result(&result);
    throw std::runtime_error(error_message);
  }
  initialized_ = true;
}

};  // namespace alcedo