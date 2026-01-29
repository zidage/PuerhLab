//  Copyright 2026 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "sleeve/storage_service.hpp"
#include "type/type.hpp"
#include "utils/cache/lru_cache.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
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
};
}  // namespace puerhlab