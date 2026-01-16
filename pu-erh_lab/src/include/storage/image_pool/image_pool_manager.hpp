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

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "image/image.hpp"
#include "type/type.hpp"
#include "utils/cache/lru_cache.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {

class ImagePoolManager {
  using ListIterator = std::list<image_id_t>::iterator;

 private:
  IncrID::IDGenerator<image_id_t>                        id_generator_{0};
  std::unordered_map<image_id_t, std::shared_ptr<Image>> image_pool_;

  LRUCache<image_id_t, image_id_t>                       thumb_cache_;

 public:
  static const uint32_t default_capacity_thumb_ = 64;
  static const uint32_t default_capacity_full_  = 3;

  explicit ImagePoolManager();
  explicit ImagePoolManager(uint32_t start_id);
  explicit ImagePoolManager(uint32_t capacity_thumb, uint32_t start_id);

  auto GetPool() -> std::unordered_map<image_id_t, std::shared_ptr<Image>>&;
  void Insert(const std::shared_ptr<Image> img);
  auto InsertEmpty() -> std::shared_ptr<Image>;
  auto PoolContains(const image_id_t& id) -> bool;

  auto Capacity() -> uint32_t;

  void RecordAccess(const image_id_t& id);
  void RemoveRecord(const image_id_t& id);
  auto CacheContains(const image_id_t& id) -> bool;

  void ResizeCache(const uint32_t new_capacity);

  auto GetCurrentID() -> image_id_t { return id_generator_.GetCurrentID(); }

  void Flush();
  void Clear();
};
};  // namespace puerhlab