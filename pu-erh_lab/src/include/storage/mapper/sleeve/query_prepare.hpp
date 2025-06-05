#pragma once

#include <string>

namespace puerhlab {
class Queries {
 public:
  static const std::string init_table_query;

  static const std::string base_insert_query;
  static const std::string base_lookup_query;

  static const std::string element_insert_query;

  static const std::string root_insert_query;
  static const std::string root_lookup_query;

  static const std::string folder_insert_query;
  static const std::string folder_content_lookup_query;

  static const std::string file_insert_query;

  static const std::string combo_insert_query;

  static const std::string filter_insert_query;

  static const std::string edit_history_insert_query;

  static const std::string version_insert_query;

  static const std::string image_insert_query;

  static const std::string image_update_query;

  static const std::string image_lookup_query;

  static const std::string image_delete_query;

  static const std::string element_delete_query;

  static const std::string folder_delete_query;
  static const std::string file_delete_query;
  static const std::string combo_delete_query_combo_id;
  static const std::string combo_delete_query_folder_id;

  static const std::string element_update_query;
  static const std::string element_lookup_query;
};
};  // namespace puerhlab
