#include "storage/mapper/sleeve/query_prepare.hpp"

namespace puerhlab {
const std::string Queries::init_table_query =
    "CREATE TABLE Sleeve (id BIGINT PRIMARY KEY);"
    "CREATE TABLE Image (id BIGINT PRIMARY KEY, image_path TEXT, file_name TEXT, type INTEGER, metadata JSON);"
    "CREATE TABLE SleeveRoot (id BIGINT PRIMARY KEY);"
    "CREATE TABLE Element (id BIGINT PRIMARY KEY, type INTEGER, element_name TEXT, added_time TIMESTAMP, modified_time "
    "TIMESTAMP, "
    "ref_count BIGINT);"
    "CREATE TABLE FolderContent (folder_id BIGINT, element_name TEXT, element_id BIGINT);"
    "CREATE TABLE FileImage (file_id BIGINT, image_id BIGINT);"
    "CREATE TABLE ComboFolder (combo_id BIGINT, folder_id BIGINT);"
    "CREATE TABLE Filter (combo_id BIGINT, type INTEGER, data JSON);"
    "CREATE TABLE EditHistory (history_id BIGINT PRIMARY KEY, file_id BIGINT, added_time TIMESTAMP, modified_time "
    "TIMESTAMP);"
    "CREATE TABLE Version (hash BIGINT PRIMARY KEY, history_id BIGINT, parent_hash BIGINT, content JSON)";

const std::string Queries::base_insert_query = "INSERT OR REPLACE INTO Sleeve (id) VALUES (?);";

const std::string Queries::element_insert_query =
    "INSERT OR REPLACE INTO Element (id,type,element_name,added_time,modified_time,ref_count) VALUES "
    "(?,?,?,?,?,?);";

const std::string Queries::root_insert_query =
    "INSERT INTO SleeveRoot (id) VALUES "
    "(?);";

const std::string Queries::folder_insert_query =
    "INSERT INTO FolderContent (folder_id,element_id) VALUES "
    "(?,?);";

const std::string Queries::file_insert_query =
    "INSERT INTO FileImage (file_id,image_id) VALUES "
    "(?,?);";

const std::string Queries::combo_insert_query =
    "INSERT INTO ComboFolder (combo_id, folder_id) VALUES "
    "(?,?);";

const std::string Queries::filter_insert_query =
    "INSERT INTO Filter (combo_id,type,data) VALUES "
    "(?,?,?);";

const std::string Queries::edit_history_insert_query =
    "INSERT OR REPLACE INTO EditHistory (history_id,file_id,added_time,modified_time) VALUES "
    "(?,?,?,?);";

const std::string Queries::version_insert_query =
    "INSERT INTO Version (hash,history_id,parent_hash,content) VALUES "
    "(?,?,?,?);";

const std::string Queries::image_insert_query =
    "INSERT INTO Image (id,image_path,file_name,type,metadata) VALUES "
    "(?,?,?,?,?);";
};  // namespace puerhlab