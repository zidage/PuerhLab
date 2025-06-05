#pragma once
#include <duckdb.h>

#include <string>
#include <vector>

#include "duckdb_types.hpp"

namespace duckorm {
duckdb_state insert(duckdb_connection& conn, const char* table, const void* obj,
                    const std::vector<DuckFieldDesc>&& fields, size_t field_count);

duckdb_state update(duckdb_connection& conn, const char* table, const void* obj,
                    const std::vector<DuckFieldDesc>&& fields, size_t field_count,
                    const char* where_clause);

duckdb_state remove(duckdb_connection& conn, const char* table, const char* where_clause);

template <typename T>
std::vector<T> select(duckdb_connection& conn, const std::string table,
                      const DuckFieldDesc* sample_fields, size_t field_count,
                      const std::string where_clause);
}  // namespace duckorm