//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "app/sleeve_filter_service.hpp"

#include <memory>

namespace puerhlab {

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
  std::optional<std::wstring> extra_where;
  if (extra_filter.has_value()) {
    const auto where_w = FilterSQLCompiler::Compile(*extra_filter);
    if (!where_w.empty()) {
      extra_where = where_w;
    }
  }

  const auto storage_stats =
      storage_service_->GetElementController().BuildFolderStats(parent_id, extra_where);

  AlbumStatsView out;
  out.total_photo_count_ = storage_stats.total_photo_count_;

  out.date_stats_.reserve(storage_stats.date_stats_.size());
  for (const auto& bucket : storage_stats.date_stats_) {
    out.date_stats_.push_back({bucket.label_, bucket.count_});
  }

  out.camera_stats_.reserve(storage_stats.camera_stats_.size());
  for (const auto& bucket : storage_stats.camera_stats_) {
    out.camera_stats_.push_back({bucket.label_, bucket.count_});
  }

  out.lens_stats_.reserve(storage_stats.lens_stats_.size());
  for (const auto& bucket : storage_stats.lens_stats_) {
    out.lens_stats_.push_back({bucket.label_, bucket.count_});
  }

  return out;
}
}  // namespace puerhlab
