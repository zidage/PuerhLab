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

#pragma once

#include <duckdb.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>

namespace duckorm {
enum class DuckDBType : uint8_t {
  INT32,
  INT64,
  UINT32,
  UINT64,
  DOUBLE,
  VARCHAR,
  JSON,
  BOOLEAN,
  TIMESTAMP,
};

/**
 * @brief A RAII class for managing prepared statements in DuckDB.
 *
 */
class PreparedStatement {
 private:
  void RecycleResources();

 public:
  duckdb_result             result_;
  duckdb_prepared_statement stmt_;
  duckdb_connection&        con_;

  bool                      prepared_ = false;
  PreparedStatement(duckdb_connection& con);
  PreparedStatement(duckdb_connection& con, const std::string& prepare_query);
  PreparedStatement();
  ~PreparedStatement();
  auto GetStmtGuard(const std::string& prepare_query) -> duckdb_prepared_statement&;
  void SetConnection(duckdb_connection& con);
};

/**
 * @brief Field descriptor for DuckDB ORM.
 *
 */
struct DuckFieldDesc {
  const char* name_;
  DuckDBType  type_;
  size_t      offset_;
};

// Macro to define a field descriptor for a specific type and field.
#define FIELD(type, field, field_type) \
  duckorm::DuckFieldDesc { #field, duckorm::DuckDBType::field_type, offsetof(type, field) }

// brief Type alias for a variant that can hold various DuckDB-supported types.
using VarTypes =
    std::variant<int32_t, int64_t, uint32_t, uint64_t, double, std::unique_ptr<std::string>>;
};  // namespace duckorm