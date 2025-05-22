#include "storage/mapper/sleeve/sleeve_mapper.hpp"

#include <duckdb.h>

#include <codecvt>
#include <ctime>
#include <exception>
#include <filesystem>
#include <memory>
#include <unordered_set>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "type/type.hpp"

namespace puerhlab {

static std::string init_table_query =
    "CREATE TABLE Sleeve (id PRIMARY KEY BIGINT);"
    "CREATE TABLE Image (id PRIMARY KEY BIGINT, image_path TEXT, file_name TEXT, type INTEGER);"
    "CREATE TABLE SleeveRoot (id PRIMARY KEY BIGINT);"
    "CREATE TABLE Element (id BIGINT, type INTEGER, element_name TEXT, added_time TIMESTAMP, modified_time "
    "TIMESTAMP, "
    "ref_count BIGINT);"
    "CREATE TABLE FolderContent (folder_id BIGINT, element_name TEXT, element_id BIGINT);"
    "CREATE TABLE FileImage (file_id BIGINT, image_id BIGINT);"
    "CREATE TABLE FilterCombo (combo_id PRIMARY KEY BIGINT, folder_id BIGINT);"
    "CREATE TABLE Filter (combo_id BIGINT, type INTEGER, data JSON);"
    "CREATE TABLE EditHistory (history_id PRIMARY KEY BIGINT, file_id BIGINT, added_time TIMESTAMP, modified_time "
    "TIMESTAMP);"
    "CREATE TABLE Version (hash PRIMARY KEY BIGINT, history_id BIGINT, parent_hash BIGINT, content JSON)";

static std::string base_insert_query = "INSERT OR REPLACE INTO Sleeve (id) VALUES (?)";

static std::string element_insert_query =
    "INSERT OR REPLACE INTO Element (id,type,element_name,added_time,modified_time,ref_count) VALUES "
    "(?,?,?,?,?,?)";

static std::string root_insert_query =
    "INSERT INTO SleeveRoot (id) VALUES "
    "(?)";

static std::string folder_insert_query =
    "INSERT INTO FolderContent (folder_id,element_id) VALUES "
    "(?,?)";

static std::string filtercombo_insert_query =
    "INSERT OR REPLACE INTO FilterCombo (combo_id,folder_id) VALUES "
    "(?,?)";

static std::string filter_insert_query =
    "INSERT INTO Filter (combo_id,types,data) VALUES "
    "(?,?,?,?)";

static std::string edit_history_insert_query =
    "INSERT OR REPLACE INTO EditHistory (history_id,file_id,added_time,modified_time) VALUES "
    "(?,?,?,?)";

static std::string version_insert_query =
    "INSERT INTO Version (hash,history_id,parent_hash,content) VALUES "
    "(?,?,?,?)";

class SleeveCaptureResources {
 private:
  void RecycleResources() {
    duckdb_destroy_prepare(&stmt_base);
    duckdb_destroy_prepare(&stmt_element);
    duckdb_destroy_prepare(&stmt_root);
    duckdb_destroy_prepare(&stmt_folder);
    duckdb_destroy_prepare(&stmt_filtercombo);
    duckdb_destroy_prepare(&stmt_filter);
    duckdb_destroy_prepare(&stmt_history);
    duckdb_destroy_prepare(&stmt_version);
    duckdb_destroy_result(&result);
  }

 public:
  duckdb_result             result;
  duckdb_prepared_statement stmt_base;
  duckdb_prepared_statement stmt_element;
  duckdb_prepared_statement stmt_root;
  duckdb_prepared_statement stmt_folder;
  duckdb_prepared_statement stmt_filtercombo;
  duckdb_prepared_statement stmt_filter;
  duckdb_prepared_statement stmt_history;
  duckdb_prepared_statement stmt_version;
  SleeveCaptureResources(duckdb_connection &con) {
    if (duckdb_prepare(con, element_insert_query.c_str(), &stmt_base) != DuckDBSuccess) {
      RecycleResources();
      throw std::exception("Prepare failed when inserting sleeve base");
    }
    if (duckdb_prepare(con, element_insert_query.c_str(), &stmt_element) != DuckDBSuccess) {
      RecycleResources();
      throw std::exception("Prepare failed when inserting sleeve elements");
    }
    if (duckdb_prepare(con, root_insert_query.c_str(), &stmt_root) != DuckDBSuccess) {
      RecycleResources();
      throw std::exception("Prepare failed when inserting sleeve root");
    }
    if (duckdb_prepare(con, element_insert_query.c_str(), &stmt_folder) != DuckDBSuccess) {
      RecycleResources();
      throw std::exception("Prepare failed when inserting folder contents");
    }
    if (duckdb_prepare(con, element_insert_query.c_str(), &stmt_filtercombo) != DuckDBSuccess) {
      RecycleResources();
      throw std::exception("Prepare failed when inserting folder filter combos");
    }
    if (duckdb_prepare(con, element_insert_query.c_str(), &stmt_filter) != DuckDBSuccess) {
      RecycleResources();
      throw std::exception("Prepare failed when inserting filters");
    }
    if (duckdb_prepare(con, element_insert_query.c_str(), &stmt_history) != DuckDBSuccess) {
      RecycleResources();
      throw std::exception("Prepare failed when inserting edit history");
    }
    if (duckdb_prepare(con, element_insert_query.c_str(), &stmt_version) != DuckDBSuccess) {
      RecycleResources();
      throw std::exception("Prepare failed when inserting edit version");
    }
  }

  ~SleeveCaptureResources() { RecycleResources(); }
};

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
  if (duckdb_query(_con, init_table_query.c_str(), &result) != DuckDBSuccess) {
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
  if (_has_sleeve && _captured_sleeve_id != sleeve_base->_sleeve_id) {
    throw std::exception("A sleeve has already been captured and cannot be overwritten");
  }

  SleeveCaptureResources res{_con};
  if (!_has_sleeve) {
    // Try to insert the sleeve base if there is no sleeve captured
    duckdb_bind_uint64(res.stmt_base, 1, sleeve_base->_sleeve_id);
    if (duckdb_execute_prepared(res.stmt_element, &res.result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&res.result);
      throw std::exception(error_message);
    }
  }
  duckdb_destroy_result(&res.result);

  // For each elements in the sleeve, insert it to the Element table
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

  // Insert the root
  duckdb_bind_uint64(res.stmt_root, 1, static_cast<uint64_t>(sleeve_base->_root->_element_id));
  if (duckdb_execute_prepared(res.stmt_root, &res.result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&res.result);
    throw std::exception(error_message);
  }
  duckdb_destroy_result(&res.result);

  std::unordered_set<sl_element_id_t> visited;

  // Insert elements
  for (auto &val : storage) {
    auto element = val.second;
    // INSERT OR REPLACE INTO Element (id,type,element_name,added_time,modified_time,ref_count) VALUES
    // (?,?,?,?,?,?)
    char added_time[32];
    char modified_time[32];
    std::strftime(added_time, sizeof(added_time), "%Y-%m-%d %H:%M:%S", std::gmtime(&element->_added_time));
    std::strftime(modified_time, sizeof(modified_time), "%Y-%m-%d %H:%M:%S",
                  std::gmtime(&element->_last_modified_time));
    duckdb_bind_uint64(res.stmt_element, 1, static_cast<uint64_t>(element->_element_id));
    duckdb_bind_uint32(res.stmt_element, 2, static_cast<uint32_t>(element->_type));
    duckdb_bind_varchar(res.stmt_element, 3, conv.to_bytes(element->_element_name).c_str());
    duckdb_bind_varchar(res.stmt_element, 4, added_time);
    duckdb_bind_varchar(res.stmt_element, 5, modified_time);
    duckdb_bind_uint64(res.stmt_element, 6, static_cast<uint64_t>(element->_ref_count));

    // Execute insertion of an element
    if (duckdb_execute_prepared(res.stmt_element, &res.result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&res.result);
      throw std::exception(error_message);
    }

    duckdb_destroy_result(&res.result);

    // Try to record all the file contents, history, and filters
    if (element->_type == ElementType::FOLDER) {
      // FolderContent (folder_id BIGINT, element_id BIGINT);
      auto folder         = std::static_pointer_cast<SleeveFolder>(element);
      auto folder_content = folder->ListElements();
      for (auto &content_id : *folder_content) {
        duckdb_bind_uint64(res.stmt_folder, 1, element->_element_id);
        duckdb_bind_uint64(res.stmt_folder, 2, content_id);
        // Execute insertion of an folder content
        if (duckdb_execute_prepared(res.stmt_folder, &res.result) != DuckDBSuccess) {
          auto error_message = duckdb_result_error(&res.result);
          throw std::exception(error_message);
        }
      }
      if (!visited.contains(element->_element_id)) {
        // Insert filters:
      }
    }
  }
}
};  // namespace puerhlab