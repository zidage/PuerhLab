//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <duckdb.h>

#include <codecvt>
#include <filesystem>
#include <string>

#include "controller_types.hpp"
#include "type/type.hpp"
#include "utils/queue/queue.hpp"

namespace alcedo {
class DBController {
 private:
  duckdb_database              db_;

  file_path_t                  db_path_;

  bool                         initialized_;

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
      "CREATE TABLE EditHistory (file_id BIGINT PRIMARY KEY, history JSON);"
      "CREATE TABLE Version (hash BIGINT PRIMARY KEY, history_id BIGINT, parent_hash BIGINT, "
      "content "
      "JSON);"
      "CREATE TABLE PipelineParam(file_id BIGINT PRIMARY KEY, param_json JSON);";

 public:
  explicit DBController(file_path_t& db_path);
  ~DBController();

  void InitializeDB();

  auto GetConnectionGuard() -> ConnectionGuard;
};
};  // namespace alcedo