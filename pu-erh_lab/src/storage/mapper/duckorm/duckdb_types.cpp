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

#include "storage/mapper/duckorm/duckdb_types.hpp"

#include <duckdb.h>

#include <exception>
#include <iostream>
#include <stdexcept>

namespace duckorm {
void PreparedStatement::RecycleResources() {
  if (stmt_) {
    duckdb_destroy_prepare(&stmt_);
    stmt_ = nullptr;
  }
  if (result_.deprecated_columns) duckdb_destroy_result(&result_);
}

PreparedStatement::PreparedStatement(duckdb_connection& con) : stmt_(), con_(con) {
  std::memset(&result_, 0, sizeof(result_));
}

PreparedStatement::PreparedStatement(duckdb_connection& con, const std::string& prepare_query)
    : stmt_(), con_(con) {
  std::memset(&result_, 0, sizeof(result_));

  // FIXME: Unified error handling
  try {
    if (duckdb_prepare(con_, prepare_query.c_str(), &stmt_) != DuckDBSuccess) {
      const char* err = duckdb_prepare_error(stmt_);
      std::string msg = "PreparedStatement failed in GetStmtGuard";
      if (err && std::strlen(err) > 0) {
        msg += ": ";
        msg += err;
      }
      RecycleResources();
      throw std::runtime_error(msg);
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception in PreparedStatement constructor: " << e.what() << std::endl;
    RecycleResources();
    throw;
  }
  prepared_ = true;
}

PreparedStatement::~PreparedStatement() { RecycleResources(); }

auto PreparedStatement::GetStmtGuard(const std::string& prepare_query)
    -> duckdb_prepared_statement& {
  if (duckdb_prepare(con_, prepare_query.c_str(), &stmt_) != DuckDBSuccess) {
    RecycleResources();
    throw std::runtime_error("PreparedStatement failed when inserting images");
  }
  prepared_ = true;
  return stmt_;
}

void PreparedStatement::SetConnection(duckdb_connection& con) { con_ = con; }
}  // namespace duckorm
