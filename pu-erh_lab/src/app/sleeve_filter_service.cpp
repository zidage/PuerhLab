//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "app/sleeve_filter_service.hpp"

#include <duckdb.h>

#include <format>
#include <memory>

#include "utils/string/convert.hpp"

namespace puerhlab {
namespace {
auto RunGroupByQuery(duckdb_connection conn, const std::string& sql) -> std::vector<StatsBucket> {
  std::vector<StatsBucket> rows;
  duckdb_result            result;
  if (duckdb_query(conn, sql.c_str(), &result) != DuckDBSuccess) {
    duckdb_destroy_result(&result);
    return rows;
  }

  const auto row_count = duckdb_row_count(&result);
  rows.reserve(static_cast<size_t>(row_count));
  for (idx_t r = 0; r < row_count; ++r) {
    char*       label_raw = duckdb_value_varchar(&result, 0, r);
    StatsBucket row;
    if (label_raw) {
      row.label_ = label_raw;
      duckdb_free(label_raw);
    }
    row.count_ = static_cast<int>(duckdb_value_int64(&result, 1, r));
    rows.push_back(std::move(row));
  }

  duckdb_destroy_result(&result);
  return rows;
}

auto RunScalarInt64(duckdb_connection conn, const std::string& sql) -> int64_t {
  duckdb_result result;
  if (duckdb_query(conn, sql.c_str(), &result) != DuckDBSuccess) {
    duckdb_destroy_result(&result);
    return 0;
  }
  int64_t value = 0;
  if (duckdb_row_count(&result) > 0) {
    value = duckdb_value_int64(&result, 0, 0);
  }
  duckdb_destroy_result(&result);
  return value;
}
}  // namespace

auto SleeveFilterService::CreateFilterCombo(const FilterNode& root) -> filter_id_t {
  filter_id_t new_id = filter_id_generator_.GenerateID();
  filter_storage_.RecordAccess(new_id, std::make_shared<FilterCombo>(new_id, root));
  return new_id;
}

auto SleeveFilterService::GetFilterCombo(filter_id_t filter_id)
    -> std::optional<std::shared_ptr<FilterCombo>> {
  auto combo_opt = filter_storage_.AccessElement(filter_id);
  if (combo_opt.has_value()) {
    return combo_opt.value();
  } else {
    return std::nullopt;
  }
}

void SleeveFilterService::RemoveFilterCombo(filter_id_t filter_id) {
  // If there is no record, this is a no-op.
  filter_storage_.RemoveRecord(filter_id);
  // The same goes for the result cache.
  filter_result_cache_.RemoveRecord(filter_id);
}

auto SleeveFilterService::ApplyFilterOn(filter_id_t filter_id, sl_element_id_t parent_id)
    -> std::optional<std::vector<sl_element_id_t>> {
  // First, check if the filter combo exists.
  auto combo_opt = filter_storage_.AccessElement(filter_id);
  if (!combo_opt.has_value()) {
    return std::nullopt;
  }
  auto combo      = combo_opt.value();

  // Next, check if we have a cached result for this filter.
  auto         result_opt = filter_result_cache_.AccessElement(filter_id);
  if (result_opt.has_value()) {
    return result_opt;
  }

  // No cached result, we need to execute the filter.
  auto result_ids =
      storage_service_->GetElementController().GetElementIdsInFolderByFilter(combo, parent_id);
  // Cache the result for future use.
  filter_result_cache_.RecordAccess(filter_id, result_ids);
  return result_ids;
}

auto SleeveFilterService::BuildFolderStats(
    sl_element_id_t parent_id, const std::optional<FilterNode>& extra_filter) -> AlbumStatsView {
  AlbumStatsView out;

  std::string extra_where;
  if (extra_filter.has_value()) {
    const auto where_w = FilterSQLCompiler::Compile(*extra_filter);
    if (!where_w.empty()) {
      extra_where = " AND (" + conv::ToBytes(where_w) + ")";
    }
  }

  auto guard = storage_service_->GetDBController().GetConnectionGuard();
  const auto base_join = std::format(
      "FROM FolderContent fc "
      "JOIN Element e ON fc.element_id = e.id "
      "JOIN FileImage fi ON fi.file_id = e.id "
      "JOIN Image i ON i.id = fi.image_id "
      "WHERE fc.folder_id = {} AND e.type = 0{}",
      parent_id, extra_where);

  out.total_photo_count_ =
      static_cast<int>(RunScalarInt64(guard.conn_, std::format("SELECT COUNT(*) {}", base_join)));

  out.date_stats_ = RunGroupByQuery(
      guard.conn_,
      std::format(
          "SELECT CAST(json_extract(i.metadata, '$.DateTimeString') AS DATE)::VARCHAR AS d, "
          "COUNT(*) AS c {} "
          "GROUP BY d ORDER BY d DESC",
          base_join));

  out.camera_stats_ = RunGroupByQuery(
      guard.conn_,
      std::format(
          "SELECT COALESCE(NULLIF(json_extract_string(i.metadata, '$.Model'), ''), '(unknown)') "
          "AS m, COUNT(*) AS c {} "
          "GROUP BY m ORDER BY c DESC",
          base_join));

  out.lens_stats_ = RunGroupByQuery(
      guard.conn_,
      std::format(
          "SELECT COALESCE(NULLIF(json_extract_string(i.metadata, '$.Lens'), ''), '(unknown)') "
          "AS l, COUNT(*) AS c {} "
          "GROUP BY l ORDER BY c DESC",
          base_join));

  return out;
}
}  // namespace puerhlab
