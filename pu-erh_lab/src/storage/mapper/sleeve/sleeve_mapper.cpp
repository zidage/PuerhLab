#include "storage/mapper/sleeve/sleeve_mapper.hpp"

#include <duckdb.h>

#include <codecvt>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <memory>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"

namespace puerhlab {

static std::string init_table_query =
    "CREATE TABLE Sleeve (id BIGINT PRIMARY KEY);"
    "CREATE TABLE Image (id BIGINT PRIMARY KEY, image_path TEXT, file_name TEXT, type INTEGER, metadata JSON);"
    "CREATE TABLE SleeveRoot (id BIGINT PRIMARY KEY);"
    "CREATE TABLE Element (id BIGINT PRIMARY KEY, type INTEGER, element_name TEXT, added_time TIMESTAMP, modified_time "
    "TIMESTAMP, "
    "ref_count BIGINT);"
    "CREATE TABLE FolderContent (folder_id BIGINT, element_name TEXT, element_id BIGINT);"
    "CREATE TABLE FileImage (file_id BIGINT, image_id BIGINT);"
    "CREATE TABLE FilterCombo (combo_id BIGINT PRIMARY KEY, folder_id BIGINT);"
    "CREATE TABLE Filter (combo_id BIGINT, type INTEGER, data JSON);"
    "CREATE TABLE EditHistory (history_id BIGINT PRIMARY KEY, file_id BIGINT, added_time TIMESTAMP, modified_time "
    "TIMESTAMP);"
    "CREATE TABLE Version (hash BIGINT PRIMARY KEY, history_id BIGINT, parent_hash BIGINT, content JSON)";

static std::string base_insert_query = "INSERT OR REPLACE INTO Sleeve (id) VALUES (?);";

static std::string element_insert_query =
    "INSERT OR REPLACE INTO Element (id,type,element_name,added_time,modified_time,ref_count) VALUES "
    "(?,?,?,?,?,?);";

static std::string root_insert_query =
    "INSERT INTO SleeveRoot (id) VALUES "
    "(?);";

static std::string folder_insert_query =
    "INSERT INTO FolderContent (folder_id,element_id) VALUES "
    "(?,?);";

static std::string file_insert_query =
    "INSERT INTO FileImage (file_id,image_id) VALUES "
    "(?,?);";

static std::string filter_insert_query =
    "INSERT INTO Filter (combo_id,type,data) VALUES "
    "(?,?,?);";

static std::string edit_history_insert_query =
    "INSERT OR REPLACE INTO EditHistory (history_id,file_id,added_time,modified_time) VALUES "
    "(?,?,?,?);";

static std::string version_insert_query =
    "INSERT INTO Version (hash,history_id,parent_hash,content) VALUES "
    "(?,?,?,?);";

static std::string image_insert_query =
    "INSERT INTO Image (id,image_path,file_name,type,metadata) VALUES "
    "(?,?,?,?,?);";

void SleeveCaptureResources::RecycleResources() {
  if (stmt_base) {
    duckdb_destroy_prepare(&stmt_base);
  }
  if (stmt_element) {
    duckdb_destroy_prepare(&stmt_element);
  }
  if (stmt_root) {
    duckdb_destroy_prepare(&stmt_root);
  }
  if (stmt_folder) {
    duckdb_destroy_prepare(&stmt_folder);
  }
  if (stmt_file) {
    duckdb_destroy_prepare(&stmt_file);
  }
  if (stmt_filter) {
    duckdb_destroy_prepare(&stmt_filter);
  }
  if (stmt_history) {
    duckdb_destroy_prepare(&stmt_history);
  }
  if (stmt_version) {
    duckdb_destroy_prepare(&stmt_version);
  }
  duckdb_destroy_result(&result);
}

SleeveCaptureResources::SleeveCaptureResources(duckdb_connection &con)
    : stmt_base(),
      stmt_element(),
      stmt_root(),
      stmt_folder(),
      stmt_file(),
      stmt_filter(),
      stmt_history(),
      stmt_version() {
  std::memset(&result, 0, sizeof(result));
  if (duckdb_prepare(con, base_insert_query.c_str(), &stmt_base) != DuckDBSuccess) {
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
  if (duckdb_prepare(con, folder_insert_query.c_str(), &stmt_folder) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("Prepare failed when inserting folder contents");
  }
  if (duckdb_prepare(con, file_insert_query.c_str(), &stmt_file) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("Prepare failed when inserting files");
  }
  if (duckdb_prepare(con, filter_insert_query.c_str(), &stmt_filter) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("Prepare failed when inserting filters");
  }
  if (duckdb_prepare(con, edit_history_insert_query.c_str(), &stmt_history) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("Prepare failed when inserting edit history");
  }
  if (duckdb_prepare(con, version_insert_query.c_str(), &stmt_version) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("Prepare failed when inserting edit version");
  }
}

SleeveCaptureResources::~SleeveCaptureResources() { RecycleResources(); }

void ImageCaptureResources::RecycleResources() {
  if (stmt_img) {
    duckdb_destroy_prepare(&stmt_img);
  }
}

ImageCaptureResources::ImageCaptureResources(duckdb_connection &con) : stmt_img() {
  std::memset(&result, 0, sizeof(result));
  if (duckdb_prepare(con, image_insert_query.c_str(), &stmt_img) != DuckDBSuccess) {
    RecycleResources();
    throw std::exception("Prepare failed when inserting images");
  }
}

ImageCaptureResources::~ImageCaptureResources() { RecycleResources(); }

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
  _initialized = true;
}

void SleeveMapper::CaptureElement(std::unordered_map<uint32_t, std::shared_ptr<SleeveElement>> &storage,
                                  SleeveCaptureResources                                       &stmts) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  for (auto &val : storage) {
    auto element = val.second;
    char added_time[32];
    char modified_time[32];
    std::strftime(added_time, sizeof(added_time), "%Y-%m-%d %H:%M:%S", std::gmtime(&element->_added_time));
    std::strftime(modified_time, sizeof(modified_time), "%Y-%m-%d %H:%M:%S",
                  std::gmtime(&element->_last_modified_time));

    duckdb_bind_uint32(stmts.stmt_element, 1, element->_element_id);
    duckdb_bind_uint32(stmts.stmt_element, 2, static_cast<uint32_t>(element->_type));
    duckdb_bind_varchar(stmts.stmt_element, 3, conv.to_bytes(element->_element_name).c_str());
    duckdb_bind_varchar(stmts.stmt_element, 4, added_time);
    duckdb_bind_varchar(stmts.stmt_element, 5, modified_time);
    duckdb_bind_uint32(stmts.stmt_element, 6, element->_ref_count);

    // Execute insertion of an element
    if (duckdb_execute_prepared(stmts.stmt_element, &stmts.result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&stmts.result);
      throw std::exception(error_message);
    }

    duckdb_destroy_result(&stmts.result);

    if (element->_type == ElementType::FOLDER) {
      CaptureFolder(std::static_pointer_cast<SleeveFolder>(element), stmts);
    } else if (element->_type == ElementType::FILE) {
      CaptureFile(std::static_pointer_cast<SleeveFile>(element), stmts);
    }
  }
}

void SleeveMapper::CaptureFolder(std::shared_ptr<SleeveFolder> folder, SleeveCaptureResources &stmts) {
  auto folder_content = folder->ListElements();
  for (auto &content_id : *folder_content) {
    // FolderContent (folder_id BIGINT, element_id BIGINT);
    duckdb_bind_uint32(stmts.stmt_folder, 1, folder->_element_id);
    duckdb_bind_uint32(stmts.stmt_folder, 2, content_id);
    // Execute insertion of an folder content
    if (duckdb_execute_prepared(stmts.stmt_folder, &stmts.result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&stmts.result);
      throw std::exception(error_message);
    }
  }
  duckdb_destroy_result(&stmts.result);
}

void SleeveMapper::CaptureFile(std::shared_ptr<SleeveFile> file, SleeveCaptureResources &stmts) {
  duckdb_bind_uint32(stmts.stmt_file, 1, file->_element_id);
  duckdb_bind_uint32(stmts.stmt_file, 2, file->GetImage()->_image_id);
  if (duckdb_execute_prepared(stmts.stmt_file, &stmts.result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&stmts.result);
    throw std::exception(error_message);
  }
  // TODO: handle with edit histories
  // For now edit history will not be stored in DB, since I haven't started working on the editor.
  duckdb_destroy_result(&stmts.result);
}

void SleeveMapper::CaptureFilters(std::unordered_map<uint32_t, std::shared_ptr<FilterCombo>> &filter_storage,
                                  SleeveCaptureResources                                     &stmts) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  for (auto &combo_it : filter_storage) {
    auto  combo    = combo_it.second;
    auto  combo_id = combo_it.first;
    auto &filters  = combo->GetFilters();
    for (auto &filter : filters) {
      // INSERT INTO Filter (combo_id,types,data)
      duckdb_bind_uint32(stmts.stmt_filter, 1, combo_id);
      duckdb_bind_uint32(stmts.stmt_filter, 2, static_cast<uint32_t>(filter._type));
      duckdb_bind_varchar(stmts.stmt_filter, 3, conv.to_bytes(filter.ToJSON()).c_str());
    }
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

  SleeveCaptureResources stmts{_con};
  if (!_has_sleeve) {
    // Try to insert the sleeve base if there is no sleeve captured
    duckdb_bind_uint32(stmts.stmt_base, 1, sleeve_base->_sleeve_id);
    if (duckdb_execute_prepared(stmts.stmt_base, &stmts.result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&stmts.result);
      throw std::exception(error_message);
    }
  }
  duckdb_destroy_result(&stmts.result);

  // Insert the root
  duckdb_bind_uint64(stmts.stmt_root, 1, static_cast<uint64_t>(sleeve_base->_root->_element_id));
  if (duckdb_execute_prepared(stmts.stmt_root, &stmts.result) != DuckDBSuccess) {
    auto error_message = duckdb_result_error(&stmts.result);
    throw std::exception(error_message);
  }
  duckdb_destroy_result(&stmts.result);

  // Capture elements
  CaptureElement(storage, stmts);

  // Capture filters
  CaptureFilters(sleeve_base->GetFilterStorage(), stmts);
}

void SleeveMapper::CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool) {
  if (!_initialized || !_db_connected) {
    throw std::exception("Cannot connect to a valid sleeve db");
  }
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  ImageCaptureResources                            stmts{_con};
  auto                                            &pool = image_pool->GetPool();
  for (auto &pool_val : pool) {
    auto img = pool_val.second;
    // INSERT INTO Image (id,image_path,file_name,type,metadata)
    duckdb_bind_uint32(stmts.stmt_img, 1, img->_image_id);
    duckdb_bind_varchar(stmts.stmt_img, 2, img->_image_path.string().c_str());
    duckdb_bind_varchar(stmts.stmt_img, 3, conv.to_bytes(img->_image_name).c_str());
    duckdb_bind_uint32(stmts.stmt_img, 4, static_cast<uint32_t>(img->_image_type));
    duckdb_bind_varchar(stmts.stmt_img, 5, img->ExifToJson().c_str());
    if (duckdb_execute_prepared(stmts.stmt_img, &stmts.result) != DuckDBSuccess) {
      auto error_message = duckdb_result_error(&stmts.result);
      throw std::exception(error_message);
    }
    duckdb_destroy_result(&stmts.result);
  }
}
};  // namespace puerhlab