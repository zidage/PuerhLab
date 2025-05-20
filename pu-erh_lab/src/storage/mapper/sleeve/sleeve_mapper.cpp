#include "storage/mapper/sleeve/sleeve_mapper.hpp"

#include <duckdb.h>

#include <exception>
#include <filesystem>

#include "type/type.hpp"

namespace puerhlab {

static std::string init_table =
    "CREATE TABLE Sleeve (id INTEGER);"
    "CREATE TABLE Image (id INTEGER, image_path TEXT, file_name TEXT, type INTEGER);"
    "CREATE TABLE SleeveRoot (id INTEGER);"
    "CREATE TABLE Element (id INTEGER, type INTEGER, element_name TEXT, added_time TIMESTAMP, modified_time "
    "TIMESTAMP, "
    "ref_count INTEGER);"
    "CREATE TABLE FolderContent (folder_id INTEGER, element_name TEXT, element_id INTEGER);"
    "CREATE TABLE Filter (combo_id INTEGER, folder_id INTEGER, type INTEGER, data JSON);"
    "CREATE TABLE EditHistory (history_id INTEGER, file_id INTEGER, added_time TIMESTAMP, modified_time TIMESTAMP);"
    "CREATE TABLE Version (history_id INTEGER, hash INTEGER, parent_hash INTEGER, content JSON)";

SleeveMapper::SleeveMapper() {}

SleeveMapper::SleeveMapper(file_path_t db_path) {
  if (std::filesystem::exists(db_path)) {
    _initialized = true;
  }
  ConnectDB(db_path);
}

SleeveMapper::~SleeveMapper() {
  duckdb_disconnect(&_con);
  duckdb_close(&_db);
}

void SleeveMapper::ConnectDB(file_path_t db_path) {
  if (_db_connected) return;
  if (duckdb_open(db_path.string().c_str(), &_db) != DuckDBSuccess) {
    throw std::exception("DB cannot be created");
  }
  if (duckdb_connect(_db, &_con) != DuckDBSuccess) {
    throw std::exception("DB cannot be connected");
  }
  _db_connected = true;
}

void SleeveMapper::InitDB() {
  if (!_db_connected) {
    throw std::exception("Must connect to a DB before initialization");
  }
  if (_initialized) {
    return;
  }

  duckdb_result result;
  if (duckdb_query(_con, init_table.c_str(), &result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&result);
    duckdb_destroy_result(&result);
    throw std::exception(error_message);
  }
}

};  // namespace puerhlab