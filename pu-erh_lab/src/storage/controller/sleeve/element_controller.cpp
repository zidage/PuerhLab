//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "storage/controller/sleeve/element_controller.hpp"

#include <duckdb.h>

#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
namespace {
auto RunGroupByQuery(duckdb_connection conn, const std::string& sql) -> std::vector<StorageStatsBucket> {
  std::vector<StorageStatsBucket> rows;
  duckdb_result                   result;
  if (duckdb_query(conn, sql.c_str(), &result) != DuckDBSuccess) {
    duckdb_destroy_result(&result);
    return rows;
  }

  const auto row_count = duckdb_row_count(&result);
  rows.reserve(static_cast<size_t>(row_count));
  for (idx_t r = 0; r < row_count; ++r) {
    char*              label_raw = duckdb_value_varchar(&result, 0, r);
    StorageStatsBucket row;
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

/**
 * @brief Construct a new Element Controller:: Element Controller object
 *
 * @param guard
 */
ElementController::ElementController(ConnectionGuard&& guard)
    : guard_(std::move(guard)),
      element_service_(guard_.conn_),
      element_id_service_(guard_.conn_),
      file_service_(guard_.conn_),
      folder_service_(guard_.conn_),
      history_service_(guard_.conn_),
      pipeline_service_(guard_.conn_),
      edit_history_service_(guard_.conn_) {}
/**
 * @brief Add an element to the database.
 *
 * @param element
 */
void ElementController::AddElement(const std::shared_ptr<SleeveElement> element) {
  element_service_.Insert(element);
  if (element->type_ == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    file_service_.Insert({file->element_id_, file->image_id_});
    if (file->GetEditHistory() != nullptr) {
      auto history = file->GetEditHistory();
      history_service_.Insert(history);
    }
  } else if (element->type_ == ElementType::FOLDER) {
    auto  folder   = std::static_pointer_cast<SleeveFolder>(element);
    auto& contents = folder->ListElements();
    for (auto& content_id : contents) {
      folder_service_.Insert({folder->element_id_, content_id});
    }
  }
  element->sync_flag_ = SyncFlag::SYNCED;
}

/**
 * @brief Add a content to a folder in the database.
 *
 * @param folder_id
 * @param content_id
 */
void ElementController::AddFolderContent(sl_element_id_t folder_id, sl_element_id_t content_id) {
  // TODO: The uniqueness of content_id is not garanteed, SQL statement should be changed
  folder_service_.Insert({folder_id, content_id});
}

/**
 * @brief Get an element by its ID from the database.
 *
 * @param id
 * @return std::shared_ptr<SleeveElement>
 */
auto ElementController::GetElementById(const sl_element_id_t id) -> std::shared_ptr<SleeveElement> {
  auto result = element_service_.GetElementById(id);
  if (result->type_ == ElementType::FILE) {
    auto file    = std::static_pointer_cast<SleeveFile>(result);
    try {
      file->image_id_ = file_service_.GetBoundImageById(file->element_id_);
    } catch (...) {
      file->image_id_ = 0;
    }
    auto history = history_service_.GetEditHistoryByFileId(file->element_id_);
    file->SetEditHistory(history);
  }
  result->SetSyncFlag(SyncFlag::SYNCED);
  return result;
}

/**
 * @brief Get the content of a folder by its ID from the database.
 *
 * @param folder_id
 * @return std::vector<sl_element_id_t>
 */
auto ElementController::GetFolderContent(const sl_element_id_t folder_id)
    -> std::vector<sl_element_id_t> {
  return folder_service_.GetFolderContent(folder_id);
}

/**
 * @brief Remove an element by its ID from the database, only be called when the ref count to the
 * element is 0.
 *
 * @param id
 */
void ElementController::RemoveElement(const sl_element_id_t id) { element_service_.RemoveById(id); }

void ElementController::RemoveElement(const std::shared_ptr<SleeveElement> element) {
  if (element->type_ == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    history_service_.RemoveById(file->element_id_);
    file_service_.RemoveById(file->element_id_);
  } else if (element->type_ == ElementType::FOLDER) {
    auto folder = std::static_pointer_cast<SleeveFolder>(element);
    folder_service_.RemoveById(folder->element_id_);
  }
  element_service_.RemoveById(element->element_id_);
}

/**
 * @brief Update an element in the database.
 *
 * @param element
 */
void ElementController::UpdateElement(const std::shared_ptr<SleeveElement> element) {
  element_service_.Update(element, element->element_id_);
  if (element->type_ == ElementType::FILE) {
    auto file = std::static_pointer_cast<SleeveFile>(element);
    file_service_.Update({file->element_id_, file->image_id_}, file->image_id_);
    history_service_.Update(file->GetEditHistory(), file->element_id_);
  } else if (element->type_ == ElementType::FOLDER) {
    auto folder = std::static_pointer_cast<SleeveFolder>(element);
    folder_service_.RemoveById(folder->element_id_);
    for (auto& content_id : folder->ListElements()) {
      AddFolderContent(folder->element_id_, content_id);
    }
  }
  element->sync_flag_ = SyncFlag::SYNCED;
}

auto ElementController::GetElementsInFolderByFilter(const std::shared_ptr<FilterCombo> filter,
                                                    const sl_element_id_t              folder_id)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  // Build SQL query from the filter
  std::wstring filter_sql = filter->GenerateSQLOn(folder_id);
  return element_service_.GetElementsInFolderByFilter(filter_sql);  // for specialized queries only
}

auto ElementController::GetElementIdsInFolderByFilter(const std::shared_ptr<FilterCombo> filter,
                                                         const sl_element_id_t folder_id)
    -> std::vector<sl_element_id_t> {
  // Build SQL query from the filter
  std::wstring filter_sql = filter->GenerateIdSQLOn(folder_id);
  return element_id_service_.GetElementIdsByQuery(filter_sql);  // for specialized queries only
}

auto ElementController::BuildFolderStats(
    sl_element_id_t folder_id, const std::optional<std::wstring>& extra_filter_where)
    -> FolderStatsView {
  FolderStatsView out;

  std::string extra_where;
  if (extra_filter_where.has_value() && !extra_filter_where->empty()) {
    extra_where = " AND (" + conv::ToBytes(*extra_filter_where) + ")";
  }

  const auto base_join = std::format(
      "FROM FolderContent fc "
      "JOIN Element e ON fc.element_id = e.id "
      "JOIN FileImage fi ON fi.file_id = e.id "
      "JOIN Image i ON i.id = fi.image_id "
      "WHERE fc.folder_id = {} AND e.type = 0{}",
      folder_id, extra_where);

  out.total_photo_count_ =
      static_cast<int>(RunScalarInt64(guard_.conn_, std::format("SELECT COUNT(*) {}", base_join)));

  out.date_stats_ = RunGroupByQuery(
      guard_.conn_,
      std::format(
          "SELECT CAST(json_extract(i.metadata, '$.DateTimeString') AS DATE)::VARCHAR AS d, "
          "COUNT(*) AS c {} "
          "GROUP BY d ORDER BY d DESC",
          base_join));

  out.camera_stats_ = RunGroupByQuery(
      guard_.conn_,
      std::format(
          "SELECT COALESCE(NULLIF(json_extract_string(i.metadata, '$.Model'), ''), '(unknown)') "
          "AS m, COUNT(*) AS c {} "
          "GROUP BY m ORDER BY c DESC",
          base_join));

  out.lens_stats_ = RunGroupByQuery(
      guard_.conn_,
      std::format(
          "SELECT COALESCE(NULLIF(json_extract_string(i.metadata, '$.Lens'), ''), '(unknown)') "
          "AS l, COUNT(*) AS c {} "
          "GROUP BY l ORDER BY c DESC",
          base_join));

  return out;
}

auto ElementController::GetPipelineByElementId(const sl_element_id_t element_id)
    -> std::shared_ptr<CPUPipelineExecutor> {
  return pipeline_service_.GetPipelineParamByFileId(element_id);
}

auto ElementController::UpdatePipelineByElementId(
    const sl_element_id_t                      element_id,
    const std::shared_ptr<CPUPipelineExecutor> pipeline) -> void {
  pipeline_service_.UpdatePipelineParamByFileId(element_id, pipeline);
}

auto ElementController::RemovePipelineByElementId(const sl_element_id_t element_id) -> void {
  pipeline_service_.RemoveById(element_id);
}

auto ElementController::GetEditHistoryByFileId(const sl_element_id_t file_id)
    -> std::shared_ptr<EditHistory> {
  return history_service_.GetEditHistoryByFileId(file_id);
}

auto ElementController::UpdateEditHistoryByFileId(const sl_element_id_t              file_id,
                                                  const std::shared_ptr<EditHistory> history)
    -> void {
  history_service_.Update(history, file_id);
}

auto ElementController::RemoveEditHistoryByFileId(const sl_element_id_t file_id) -> void {
  history_service_.RemoveById(file_id);
}

};  // namespace puerhlab
