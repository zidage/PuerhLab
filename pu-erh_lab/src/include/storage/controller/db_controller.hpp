#pragma once

#include <duckdb.h>

#include <codecvt>
#include <filesystem>
#include <string>

#include "controller_types.hpp"
#include "type/type.hpp"
#include "utils/queue/queue.hpp"

namespace puerhlab {
class DBController {
 private:
  duckdb_database              _db;

  file_path_t                  _db_path;

  bool                         _initialized;

  constexpr static const char* init_table_query =
      "CREATE TABLE Sleeve (id BIGINT PRIMARY KEY);"
      "CREATE TABLE Image (id BIGINT PRIMARY KEY, image_path TEXT, file_name TEXT, type INTEGER, "
      "metadata JSON);"
      "CREATE TABLE SleeveRoot (id BIGINT PRIMARY KEY);"
      "CREATE TABLE Element (id BIGINT PRIMARY KEY, type INTEGER, element_name TEXT, added_time "
      "TIMESTAMP, modified_time "
      "TIMESTAMP, "
      "ref_count BIGINT);"
      "CREATE TABLE FolderContent (folder_id BIGINT, element_id BIGINT);"
      "CREATE TABLE FileImage (file_id BIGINT, image_id BIGINT);"
      "CREATE TABLE ComboFolder (combo_id BIGINT, folder_id BIGINT);"
      "CREATE TABLE Filter (combo_id BIGINT, type INTEGER, data JSON);"
      "CREATE TABLE EditHistory (file_id BIGINT, history JSON);"
      "CREATE TABLE Version (hash BIGINT PRIMARY KEY, history_id BIGINT, parent_hash BIGINT, "
      "content "
      "JSON)";

 public:
  explicit DBController(file_path_t& db_path);
  ~DBController();

  void InitializeDB();

  auto GetConnectionGuard() -> ConnectionGuard;
};
};  // namespace puerhlab