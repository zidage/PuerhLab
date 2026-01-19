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

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>
namespace puerhlab {
template <typename K>
concept Hashable = std::copy_constructible<K> && std::equality_comparable<K> && requires(K key) {
  { std::hash<K>{}(key) } -> std::convertible_to<std::size_t>;
};
template <Hashable K, typename V>
class LRUCache {
  using ListIterator = std::list<std::pair<K, V>>::iterator;

 private:
  std::unordered_map<K, ListIterator> cache_map_;
  std::list<std::pair<K, V>>          cache_list_;
  size_t                              capacity_;

  uint32_t                            evict_count_  = 0;
  uint32_t                            access_count_ = 0;

 public:
  static const uint32_t default_capacity_ = 256;

  explicit LRUCache() : capacity_(default_capacity_) {}
  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  auto Contains(const K& key) -> bool { return cache_map_.contains(key); }

  auto AccessElement(const K& key) -> std::optional<V> {
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
      return std::nullopt;
    }
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    return it->second->second;
  }

  void RecordAccess(const K& key, const V& val) {
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
      cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
      if (cache_list_.front().second != val) {
        cache_list_.front() = {key, val};
      }
    } else {
      if (cache_list_.size() >= capacity_) {
        Evict();
      }

      cache_list_.push_front({key, val});
      cache_map_[key] = cache_list_.begin();
    }
  }

  auto RecordAccess_WithEvict(const K& key, const V& val) -> std::optional<V> {
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
      cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
      if (cache_list_.front().second != val) {
        cache_list_.front() = {key, val};
      }
      return std::nullopt;
    } else {
      std::optional<V> evicted;
      if (cache_list_.size() >= capacity_) {
        evicted = Evict();
      }

      cache_list_.push_front({key, val});
      cache_map_[key] = cache_list_.begin();
      return evicted;
    }
    return std::nullopt;
  }

  void RemoveRecord(const K& path) {
    auto it = cache_map_.find(path);
    if (it != cache_map_.end()) {
      cache_list_.erase(it->second);
      cache_map_.erase(it);
    }
  }

  auto Evict() -> std::optional<V> {
    if (cache_list_.empty()) {
      return std::nullopt;
    }
    auto last = cache_list_.end();
    --last;
    cache_map_.erase(last->first);
    auto evicted_id = last->second;
    cache_list_.pop_back();
    ++evict_count_;
    // Resize
    if (access_count_ != 0 && (double)evict_count_ / (double)access_count_ > 0.8) {
      Resize(static_cast<uint32_t>(capacity_ * 1.2));
    }
    return evicted_id;
  }

  void Resize(uint32_t new_capacity) {
    if (new_capacity > capacity_) {
      auto all_keys = GetLRUKeys();
      Flush();
      capacity_ = new_capacity;
      for (const auto& key : all_keys) {
        RecordAccess(key, key);
      }
    } else {
      capacity_ = new_capacity;
    }
  }

  void Flush() {
    cache_map_.clear();
    cache_list_.clear();
  }

  auto GetLRUKeys() const -> std::vector<K> {
    std::vector<K> keys;
    keys.reserve(cache_list_.size());
    for (auto it = cache_list_.rbegin(); it != cache_list_.rend(); ++it) {
      keys.push_back(it->first);
    }
    return keys;
  }
};
};  // namespace puerhlab