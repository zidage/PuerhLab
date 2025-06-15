#include "storage/mapper/duckorm/duckdb_orm.hpp"

#include <duckdb.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "storage/mapper/sleeve/statement_prepare.hpp"

namespace duckorm {
/**
 * @brief Insert an object into a DuckDB table.
 *
 * @param conn a reference to the DuckDB connection
 * @param table the name of the table to insert into
 * @param obj the object to insert, which should be a pointer to a struct
 *            with fields matching the DuckFieldDesc descriptions
 * @param fields a span of DuckFieldDesc describing the fields of the object
 * @param field_count the number of fields in the object
 * @return duckdb_state
 */
duckdb_state insert(duckdb_connection& conn, const char* table, const void* obj,
                    std::span<const DuckFieldDesc> fields, size_t field_count) {
  // Construct the SQL insert statement
  std::ostringstream sql;
  sql << "INSERT INTO " << table << " (";
  for (size_t i = 0; i < field_count; ++i) {
    sql << fields[i].name;
    if (i < field_count - 1) {
      sql << ",";
    }
  }
  sql << ") VALUES (";
  for (size_t i = 0; i < field_count; ++i) {
    sql << "?";
    if (i < field_count - 1) {
      sql << ",";
    }
  }
  sql << ");";
  std::string       sql_str = sql.str();
  PreparedStatement insert_pre{conn, sql_str};

  // Bind parameters
  for (size_t i = 0; i < field_count; ++i) {
    const DuckFieldDesc& field = fields[i];
    const char*          ptr   = reinterpret_cast<const char*>(obj) + field.offset;
    switch (field.type) {
      case DuckDBType::INT32: {
        int32_t value = *reinterpret_cast<const int32_t*>(ptr);
        duckdb_bind_int32(insert_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::INT64: {
        int64_t value = *reinterpret_cast<const int64_t*>(ptr);
        duckdb_bind_int64(insert_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::UINT32: {
        uint32_t value =
            *reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(obj) + field.offset);
        duckdb_bind_uint32(insert_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::UINT64: {
        uint64_t value =
            *reinterpret_cast<const uint64_t*>(reinterpret_cast<const char*>(obj) + field.offset);
        duckdb_bind_uint64(insert_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::DOUBLE: {
        double value =
            *reinterpret_cast<const double*>(reinterpret_cast<const char*>(obj) + field.offset);
        duckdb_bind_double(insert_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::TIMESTAMP:
      case DuckDBType::JSON:
      case DuckDBType::VARCHAR: {
        // For TIMESTAMP, VARCHAR and JSON, we assume the field is a std::unique_ptr<std::string>
        const std::unique_ptr<std::string>* ptr_to_unique =
            reinterpret_cast<const std::unique_ptr<std::string>*>(
                reinterpret_cast<const char*>(obj) + field.offset);
        duckdb_bind_varchar(insert_pre._stmt, i + 1, ptr_to_unique->get()->c_str());
        break;
      }
      case DuckDBType::BOOLEAN:
        duckdb_bind_boolean(
            insert_pre._stmt, i + 1,
            *reinterpret_cast<const bool*>(reinterpret_cast<const char*>(obj) + field.offset));
        break;
      default:
        throw std::runtime_error("Unsupported DuckFieldType in insert()");  // Unsupported type
    }
  }

  duckdb_state state = duckdb_execute_prepared(insert_pre._stmt, &insert_pre._result);
  if (state != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&insert_pre._result);
    throw std::runtime_error(error_message);
  }
  return state;
}

/**
 * @brief Update an object in a DuckDB table.
 *
 * @param conn
 * @param table
 * @param obj
 * @param fields
 * @param field_count
 * @param where_clause
 * @return duckdb_state
 */
duckdb_state update(duckdb_connection& conn, const char* table, const void* obj,
                    std::span<const DuckFieldDesc> fields, size_t field_count,
                    const char* where_clause) {
  std::ostringstream sql;
  sql << "UPDATE " << table << " SET ";
  for (size_t i = 0; i < field_count; ++i) {
    sql << fields[i].name << " = ?";
    if (i < field_count - 1) {
      sql << ", ";
    }
  }
  sql << " WHERE " << where_clause << ";";
  std::string       sql_str = sql.str();

  PreparedStatement update_pre(conn, sql_str);

  // Bind parameters
  for (size_t i = 0; i < field_count; ++i) {
    const DuckFieldDesc& field = fields[i];
    const char*          ptr   = reinterpret_cast<const char*>(obj) + field.offset;
    switch (field.type) {
      case DuckDBType::INT32: {
        int32_t value = *reinterpret_cast<const int32_t*>(ptr);
        duckdb_bind_int32(update_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::INT64: {
        int64_t value = *reinterpret_cast<const int64_t*>(ptr);
        duckdb_bind_int64(update_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::UINT32: {
        uint32_t value = *reinterpret_cast<const uint32_t*>(ptr);
        duckdb_bind_uint32(update_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::UINT64: {
        uint64_t value = *reinterpret_cast<const uint64_t*>(ptr);
        duckdb_bind_uint64(update_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::DOUBLE: {
        double value = *reinterpret_cast<const double*>(ptr);
        duckdb_bind_double(update_pre._stmt, i + 1, value);
        break;
      }
      case DuckDBType::TIMESTAMP:
      case DuckDBType::JSON:
      case DuckDBType::VARCHAR: {
        auto member_ptr = reinterpret_cast<const std::unique_ptr<std::string>*>(ptr);
        auto c_str      = member_ptr->get()->c_str();
        duckdb_bind_varchar(update_pre._stmt, i + 1, c_str);
        break;
      }
      case DuckDBType::BOOLEAN:
        duckdb_bind_boolean(
            update_pre._stmt, i + 1,
            *reinterpret_cast<const bool*>(reinterpret_cast<const char*>(obj) + field.offset));
        break;
      default:
        duckdb_destroy_prepare(&update_pre._stmt);
        throw std::runtime_error("Unsupported DuckFieldType in update()");  // Unsupported type
    }
  }

  duckdb_state state = duckdb_execute_prepared(update_pre._stmt, &update_pre._result);
  if (state != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&update_pre._result);
    throw std::runtime_error(error_message);
  }

  return state;
}

/**
 * @brief Remove an object from a DuckDB table based on a where clause.
 *
 * @param conn
 * @param table
 * @param where_clause
 * @return duckdb_state
 */
duckdb_state remove(duckdb_connection& conn, const char* table, const char* where_clause) {
  std::ostringstream sql;
  sql << "DELETE FROM " << table << " WHERE " << where_clause << ";";
  std::string       sql_str = sql.str();

  PreparedStatement delete_pre(conn, sql_str);

  duckdb_state      state = duckdb_execute_prepared(delete_pre._stmt, &delete_pre._result);
  if (state != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&delete_pre._result);
    throw std::runtime_error(error_message);
  }

  return state;
}

/**
 * @brief Select rows from a DuckDB table based on a where clause.
 *
 * @param conn
 * @param table
 * @param sample_fields
 * @param field_count
 * @param where_clause
 * @return std::vector<std::vector<VarTypes>>
 */
std::vector<std::vector<VarTypes>> select(duckdb_connection& conn, const std::string table,
                                          std::span<const DuckFieldDesc> sample_fields,
                                          size_t field_count, const char* where_clause) {
  std::ostringstream sql;
  sql << "SELECT * FROM " << table << " WHERE " << where_clause << ";";

  std::vector<std::vector<VarTypes>> results;
  PreparedStatement                  select_pre(conn, sql.str());

  if (duckdb_execute_prepared(select_pre._stmt, &select_pre._result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&select_pre._result);
    throw std::runtime_error(error_message);
  }

  if (duckdb_column_count(&select_pre._result) != field_count) {
    throw std::runtime_error("Column count mismatch in select query");
  }

  idx_t row_count = duckdb_row_count(&select_pre._result);
  if (row_count == 0) {
    return results;  // No rows found
  }
  results.resize(row_count);
  for (idx_t i = 0; i < row_count; ++i) {
    // Construct a field describe
    results[i].resize(field_count);
    for (size_t j = 0; j < field_count; ++j) {
      switch (sample_fields[j].type) {
        case DuckDBType::INT32: {
          int32_t value = duckdb_value_int32(&select_pre._result, j, i);
          results[i][j] = value;
          break;
        };
        case DuckDBType::INT64: {
          int64_t value = duckdb_value_int64(&select_pre._result, j, i);
          results[i][j] = value;
          break;
        }
        case DuckDBType::UINT32: {
          uint32_t value = duckdb_value_uint32(&select_pre._result, j, i);
          results[i][j]  = value;
          break;
        }
        case DuckDBType::UINT64: {
          uint64_t value = duckdb_value_uint64(&select_pre._result, j, i);
          results[i][j]  = value;
          break;
        }
        case DuckDBType::DOUBLE: {
          double value  = duckdb_value_double(&select_pre._result, j, i);
          results[i][j] = value;
          break;
        }
        case DuckDBType::VARCHAR:
        case DuckDBType::JSON:
        case DuckDBType::BOOLEAN:
        case DuckDBType::TIMESTAMP: {
          const char* value = duckdb_value_varchar(&select_pre._result, j, i);
          results[i][j]     = std::make_unique<std::string>(value);
          break;
        }
        default:
          throw std::runtime_error("Unsupported DuckFieldType in select()");
      }
    }
  }

  return results;
}
};  // namespace duckorm