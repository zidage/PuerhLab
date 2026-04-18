//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>

#include "sleeve_element/sleeve_element.hpp"
#include "type/type.hpp"

#pragma once

namespace alcedo {
class DCacheManager {
  using ListIterator = std::list<std::pair<sl_path_t, sl_element_id_t>>::iterator;

 private:
  std::unordered_map<sl_path_t, ListIterator>      cache_map_;
  std::list<std::pair<sl_path_t, sl_element_id_t>> cache_list_;
  uint32_t                                         capacity_;

  uint32_t                                         evict_count_  = 0;
  uint32_t                                         access_count_ = 0;

 public:
  static const uint32_t default_capacity_ = 256;
  explicit DCacheManager();
  explicit DCacheManager(uint32_t capacity);

  auto AccessElement(const sl_path_t path) -> std::optional<sl_element_id_t>;
  void RecordAccess(const sl_path_t path, const sl_element_id_t element_id);
  void RemoveRecord(const sl_path_t path);
  auto Evict() -> std::optional<sl_element_id_t>;
  auto Contains(const sl_path_t& path) -> bool;

  void Flush();
  void Resize(uint32_t new_capacity);
};
};  // namespace alcedo