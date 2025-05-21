#include "storage/mapper/sleeve/sleeve_mapper.hpp"

#include <duckdb.h>

#include <codecvt>
#include <ctime>
#include <exception>
#include <filesystem>

#include "type/type.hpp"

namespace puerhlab {

static std::string init_table =
    "CREATE TABLE Sleeve (id PRIMARY KEY BIGINT);"
    "CREATE TABLE Image (id PRIMARY KEY BIGINT, image_path TEXT, file_name TEXT, type INTEGER);"
    "CREATE TABLE SleeveRoot (id PRIMARY KEY BIGINT);"
    "CREATE TABLE Element (id BIGINT, type INTEGER, element_name TEXT, added_time TIMESTAMP, modified_time "
    "TIMESTAMP, "
    "ref_count BIGINT);"
    "CREATE TABLE FolderContent (folder_id BIGINT, element_name TEXT, element_id BIGINT);"
    "CREATE TABLE Filter (combo_id BIGINT, folder_id BIGINT, type INTEGER, data JSON);"
    "CREATE TABLE EditHistory (history_id BIGINT, file_id BIGINT, added_time TIMESTAMP, modified_time TIMESTAMP);"
    "CREATE TABLE Version (history_id BIGINT, hash BIGINT, parent_hash INTEGER, content JSON)";

static std::string element_insert =
    "INSERT OR REPLACE INTO Element (id,type,element_name,added_time,modified_time,ref_count) VALUES "
    "(?,?,?,?,?,?)";

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

void SleeveMapper::CaptureSleeve(const std::shared_ptr<SleeveBase> sleeve_base) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }

  std::unordered_map<uint32_t, std::shared_ptr<SleeveElement>> &storage = sleeve_base->GetStorage();
  duckdb_result                                                 result;
  if (_has_sleeve && _captured_sleeve_id != sleeve_base->_sleeve_id) {
    throw std::exception("A sleeve has already been captured and cannot be overwritten");
  }

  if (!_has_sleeve) {
    // Try to insert the sleeve base if there is no sleeve captured
    if (duckdb_query(_con,
                     std::format("INSERT OR REPLACE INTO Sleeve (id) VALUES ({})", sleeve_base->_sleeve_id).c_str(),
                     &result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&result);
      duckdb_destroy_result(&result);
      throw std::exception(error_message);
    }
  }

  duckdb_prepared_statement stmt_element;
  if (duckdb_prepare(_con, element_insert.c_str(), &stmt_element) != DuckDBSuccess) {
    throw std::exception("Prepare failed when inserting sleeve elements");
  }
  // For each elements in the sleeve, insert it to the Element table
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  for (auto &val : storage) {
    auto element = val.second;
    // INSERT OR REPLACE INTO Element (id,type,element_name,added_time,modified_time,ref_count) VALUES
    // (?,?,?,?,?,?)
    char added_time[32];
    char modified_time[32];
    std::strftime(added_time, sizeof(added_time), "%Y-%m-%d %H:%M:%S", std::gmtime(&element->_added_time));
    std::strftime(modified_time, sizeof(modified_time), "%Y-%m-%d %H:%M:%S",
                  std::gmtime(&element->_last_modified_time));
    duckdb_bind_int64(stmt_element, 1, static_cast<int64_t>(element->_element_id));
    duckdb_bind_varchar(stmt_element, 2, conv.to_bytes(element->_element_name).c_str());
    duckdb_bind_varchar(stmt_element, 3, added_time);
    duckdb_bind_varchar(stmt_element, 4, modified_time);
    duckdb_bind_int64(stmt_element, 5, static_cast<int64_t>(element->_ref_count));
  }
}
};  // namespace puerhlab