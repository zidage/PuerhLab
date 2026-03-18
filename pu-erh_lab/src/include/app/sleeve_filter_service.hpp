//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "sleeve/storage_service.hpp"
#include "type/type.hpp"
#include "utils/cache/lru_cache.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
struct StatsBucket {
  std::string label_{};
  int         count_ = 0;
};

struct AlbumStatsView {
  int                      total_photo_count_ = 0;
  std::vector<StatsBucket> date_stats_{};
  std::vector<StatsBucket> camera_stats_{};
  std::vector<StatsBucket> lens_stats_{};
};

// This service should not be used in multi-threaded scenarios.
class SleeveFilterService {
 private:
  std::shared_ptr<StorageService>                     storage_service_;

  // Filter will not be saved in DB for now. It will be only stored in memory for the lifetime of
  // the application.
  IncrID::IDGenerator<filter_id_t>                    filter_id_generator_;

  LRUCache<filter_id_t, std::shared_ptr<FilterCombo>>                  filter_storage_;
  LRUCache<filter_id_t, std::vector<sl_element_id_t>> filter_result_cache_;

 public:
  // Disable all copy operations
  SleeveFilterService()                                      = delete;
  SleeveFilterService(const SleeveFilterService&)            = delete;
  SleeveFilterService& operator=(const SleeveFilterService&) = delete;

  SleeveFilterService(std::shared_ptr<StorageService> storage_service)
      : storage_service_(std::move(storage_service)), filter_id_generator_(0) {}

  auto CreateFilterCombo(const FilterNode& root) -> filter_id_t;
  auto GetFilterCombo(filter_id_t filter_id) -> std::optional<std::shared_ptr<FilterCombo>>;
  void RemoveFilterCombo(filter_id_t filter_id);
  auto ApplyFilterOn(filter_id_t filter_id, sl_element_id_t parent_id)
      -> std::optional<std::vector<sl_element_id_t>>;
  auto BuildFolderStats(sl_element_id_t parent_id,
                        const std::optional<FilterNode>& extra_filter = std::nullopt)
      -> AlbumStatsView;
};
}  // namespace puerhlab
