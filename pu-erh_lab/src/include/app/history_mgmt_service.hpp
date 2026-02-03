//  Copyright 2025 Yurun Zi
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
#include <mutex>
#include <unordered_map>

#include "edit/history/edit_history.hpp"
#include "sleeve/storage_service.hpp"
#include "type/type.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {
struct EditHistoryGuard {
  sl_element_id_t              file_id_;
  std::shared_ptr<EditHistory> history_;

  bool                         dirty_  = false;
  bool                         pinned_ = false;
};

class EditHistoryMgmtService final {
 private:
  std::shared_ptr<StorageService>            storage_service_;

  LRUCache<sl_element_id_t, sl_element_id_t> cache_;
  std::unordered_map<sl_element_id_t, std::shared_ptr<EditHistoryGuard>> cached_histories_;

  std::mutex                                  lock_;

  static constexpr size_t                     default_cache_capacity_ = 16;

  void HandleEviction(sl_element_id_t evicted_id);

 public:
  EditHistoryMgmtService() = delete;
  explicit EditHistoryMgmtService(std::shared_ptr<StorageService> storage_service)
      : storage_service_(std::move(storage_service)),
        cache_(default_cache_capacity_),
        cached_histories_() {}

  auto LoadHistory(sl_element_id_t file_id) -> std::shared_ptr<EditHistoryGuard>;

  void SaveHistory(const std::shared_ptr<EditHistoryGuard>& history_guard);

  void Sync();
};
};  // namespace puerhlab
