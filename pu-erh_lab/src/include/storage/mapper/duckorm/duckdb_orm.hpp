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

#include <span>
#include <string>
#include <vector>

#include "duckdb_types.hpp"

namespace duckorm {
duckdb_state insert(duckdb_connection& conn, const char* table, const void* obj,
                    std::span<const DuckFieldDesc> fields, size_t field_count);

duckdb_state update(duckdb_connection& conn, const char* table, const void* obj,
                    std::span<const DuckFieldDesc> fields, size_t field_count,
                    const char* where_clause);

duckdb_state remove(duckdb_connection& conn, const char* table, const char* where_clause);

std::vector<std::vector<VarTypes>> select(duckdb_connection& conn, const std::string table,
                                          std::span<const DuckFieldDesc> sample_fields,
                                          size_t field_count, const char* where_clause);

std::vector<std::vector<VarTypes>> select_by_query(duckdb_connection&             conn,
                                                   std::span<const DuckFieldDesc> sample_fields,
                                                   size_t field_count, const std::string& sql);
}  // namespace duckorm