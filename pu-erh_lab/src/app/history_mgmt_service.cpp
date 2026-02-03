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

#include "app/history_mgmt_service.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include "type/type.hpp"

namespace puerhlab {
void EditHistoryMgmtService::HandleEviction(sl_element_id_t evicted_id) {
  // If the would-be evicted history is pinned, keep it and evict another entry instead.
  // This service is typically single-session/single-history, so pinned guards are expected.
  sl_element_id_t candidate    = evicted_id;
  const size_t    max_attempts = cached_histories_.empty() ? 1 : (cached_histories_.size() + 1);

  for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
    auto it = cached_histories_.find(candidate);
    if (it == cached_histories_.end()) {
      return;
    }

    auto history_guard = it->second;
    if (!history_guard->pinned_) {
      if (history_guard->dirty_) {
        storage_service_->GetElementController().UpdateEditHistoryByFileId(candidate,
                                                                          history_guard->history_);
        history_guard->dirty_ = false;
      }
      cached_histories_.erase(it);
      cache_.RemoveRecord(candidate);
      return;
    }

    // Pinned: put it back into the LRU and evict a different entry.
    auto next = cache_.RecordAccess_WithEvict(candidate, candidate);
    if (!next.has_value()) {
      return;
    }
    candidate = next.value();
  }

  // Fallback: if everything is pinned, allow temporary growth to avoid evicting in-use histories.
  auto keys = cache_.GetLRUKeys();
  cache_.Resize(static_cast<uint32_t>(keys.size() + 5));
  cache_.RecordAccess(evicted_id, evicted_id);
}

auto EditHistoryMgmtService::LoadHistory(sl_element_id_t file_id)
    -> std::shared_ptr<EditHistoryGuard> {
  std::unique_lock<std::mutex> guard(lock_);

  if (cache_.Contains(file_id)) {
    auto cached_id = cache_.AccessElement(file_id);
    if (cached_id.has_value()) {
      auto it = cached_histories_.find(cached_id.value());
      if (it != cached_histories_.end()) {
        // Always pin on load; this service is intended for a single active history per session.
        it->second->pinned_  = true;
        it->second->file_id_ = file_id;
        return it->second;
      }
    }
  }

  std::shared_ptr<EditHistory>      history;
  std::shared_ptr<EditHistoryGuard> history_guard;
  try {
    history = storage_service_->GetElementController().GetEditHistoryByFileId(file_id);
  } catch (std::exception& e) {
    throw std::runtime_error(
        "[ERROR] EditHistoryMgmtService: Failed to load edit history from storage for file ID " +
        std::to_string(file_id) + ": " + e.what());
  }

  if (!history) {
    history = std::make_shared<EditHistory>(file_id);
  }

  history_guard            = std::make_shared<EditHistoryGuard>();
  history_guard->file_id_  = file_id;
  history_guard->history_  = std::move(history);
  history_guard->dirty_    = false;
  history_guard->pinned_   = true;

  std::optional<sl_element_id_t> evicted = cache_.RecordAccess_WithEvict(file_id, file_id);
  if (evicted.has_value()) {
    HandleEviction(evicted.value());
  }

  cached_histories_[file_id] = history_guard;

  // If no eviction happened, and the cache size is still in "boost" range, resize it
  if (!evicted.has_value() && cached_histories_.size() + 1 > default_cache_capacity_) {
    cache_.Resize(static_cast<uint32_t>(cached_histories_.size() - 1));
  }

  return history_guard;
}

void EditHistoryMgmtService::SaveHistory(const std::shared_ptr<EditHistoryGuard>& history_guard) {
  if (!history_guard) {
    return;
  }

  std::unique_lock<std::mutex> guard(lock_);

  const sl_element_id_t file_id = history_guard->file_id_;

  // Ensure the guard remains tracked by the cache even if constructed externally.
  std::optional<sl_element_id_t> evicted = cache_.RecordAccess_WithEvict(file_id, file_id);
  if (evicted.has_value()) {
    HandleEviction(evicted.value());
  }

  cached_histories_[file_id] = history_guard;

  if (history_guard->dirty_) {
    storage_service_->GetElementController().UpdateEditHistoryByFileId(file_id,
                                                                      history_guard->history_);
    history_guard->dirty_ = false;
  }

  // Return to cache (unpinned)
  history_guard->pinned_ = false;

  // If eviction did not happen, but the cache size is still in "boost" range, resize it
  if (!evicted.has_value() && cached_histories_.size() + 1 > default_cache_capacity_) {
    cache_.Resize(static_cast<uint32_t>(cached_histories_.size() - 1));
  }
}

void EditHistoryMgmtService::Sync() {
  std::unique_lock<std::mutex> guard(lock_);
  for (auto& [file_id, history_guard] : cached_histories_) {
    if (!history_guard || !history_guard->dirty_) {
      continue;
    }
    storage_service_->GetElementController().UpdateEditHistoryByFileId(file_id,
                                                                      history_guard->history_);
    history_guard->dirty_ = false;
  }
}
}  // namespace puerhlab

