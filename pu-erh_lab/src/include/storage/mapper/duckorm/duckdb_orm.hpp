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
                                          std::span<const DuckFieldDesc> fields, size_t field_count,
                                          const std::string where_clause);
}  // namespace duckorm