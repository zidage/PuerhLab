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
#include <stdexcept>

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

  // FIXME: Unified error handling
  if (duckdb_prepare(_con, prepare_query.c_str(), &_stmt) != DuckDBSuccess) {
    const char* err = duckdb_prepare_error(_stmt);
    std::string msg = "PreparedStatement failed in GetStmtGuard";
    if (err && std::strlen(err) > 0) {
      msg += ": ";
      msg += err;
    }
    RecycleResources();
    throw std::runtime_error(msg);
  }
  _prepared = true;
}

PreparedStatement::~PreparedStatement() { RecycleResources(); }

auto PreparedStatement::GetStmtGuard(const std::string& prepare_query)
    -> duckdb_prepared_statement& {
  if (duckdb_prepare(_con, prepare_query.c_str(), &_stmt) != DuckDBSuccess) {
    RecycleResources();
    throw std::runtime_error("PreparedStatement failed when inserting images");
  }
  _prepared = true;
  return _stmt;
}

void PreparedStatement::SetConnection(duckdb_connection& con) { _con = con; }
}  // namespace duckorm
