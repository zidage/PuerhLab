//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "sleeve/dentry_cache_manager.hpp"

#include <cstdint>
#include <optional>

#include "type/type.hpp"

namespace alcedo {
DCacheManager::DCacheManager() : capacity_(512) {}
DCacheManager::DCacheManager(uint32_t capacity) : capacity_(capacity) {}

/**
 * @brief Check if a path exists in the cache
 *
 * @param path
 * @return true
 * @return false
 */
auto DCacheManager::Contains(const sl_path_t& path) -> bool {
  return cache_map_.find(path) != cache_map_.end();
}

/**
 * @brief Access an element in the cache.
 *
 * @param path
 * @return std::optional<sl_element_id_t> an element_id to the element or null if path not presents
 * in the cache
 */
auto DCacheManager::AccessElement(const sl_path_t path) -> std::optional<sl_element_id_t> {
  auto it = cache_map_.find(path);
  if (it == cache_map_.end()) {
    return std::nullopt;
  }
  cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
  return it->second->second;
}

void DCacheManager::RecordAccess(const sl_path_t path, const sl_element_id_t element_id) {
  auto it = cache_map_.find(path);
  if (it != cache_map_.end()) {
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    if (cache_list_.front().second != element_id) {
      cache_list_.front() = {path, element_id};
    }
  } else {
    if (cache_list_.size() >= capacity_) {
      Evict();
    }

    cache_list_.push_front({path, element_id});
    cache_map_[path] = cache_list_.begin();
  }
}

void DCacheManager::RemoveRecord(const sl_path_t path) {
  auto it = cache_map_.find(path);
  if (it != cache_map_.end()) {
    cache_list_.erase(it->second);
    cache_map_.erase(it);
  }
}

auto DCacheManager::Evict() -> std::optional<sl_element_id_t> {
  if (cache_list_.empty()) {
    return std::nullopt;
  }
  auto last = cache_list_.end();
  --last;
  cache_map_.erase(last->first);
  auto evicted_id = last->second;
  cache_list_.pop_back();
  ++evict_count_;
  if (access_count_ != 0 && (double)evict_count_ / (double)access_count_ > 0.8) {
    Resize(capacity_ * 1.2);
  }
  return evicted_id;
}

void DCacheManager::Resize(uint32_t new_capacity) {
  if (new_capacity > capacity_) {
    Flush();
  }
  capacity_ = new_capacity;
}

void DCacheManager::Flush() {
  cache_map_.clear();
  cache_list_.clear();
}
};  // namespace alcedo