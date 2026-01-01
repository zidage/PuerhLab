//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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
  if (duckdb_open(utf8_str.c_str(), &db_) != DuckDBSuccess) {
    throw std::runtime_error("DB cannot be created");
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

};  // namespace puerhlab