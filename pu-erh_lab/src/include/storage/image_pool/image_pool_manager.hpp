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

namespace puerhlab {

enum class AccessType { THUMB, FULL_IMG, META };

class ImagePoolManager {
  using ListIterator = std::list<image_id_t>::iterator;

 private:
  std::unordered_map<image_id_t, std::shared_ptr<Image>> _image_pool;

  std::unordered_map<image_id_t, ListIterator>           _cache_map_thumb;
  std::list<image_id_t>                                  _cache_list_thumb;

  std::unordered_map<image_id_t, ListIterator>           _cache_map_full;
  std::list<image_id_t>                                  _cache_list_full;

  uint32_t                                               _capacity_thumb;
  uint32_t                                               _capacity_full;

  std::unordered_set<image_id_t>                         _with_thumb;
  std::unordered_set<image_id_t>                         _with_full;

 public:
  static const uint32_t _default_capacity_thumb = 64;
  static const uint32_t _default_capacity_full  = 3;

  explicit ImagePoolManager();
  explicit ImagePoolManager(uint32_t capacity_thumb, uint32_t capacity_full);

  auto GetPool() -> std::unordered_map<image_id_t, std::shared_ptr<Image>>&;
  void Insert(const std::shared_ptr<Image> img);
  auto PoolContains(const image_id_t& id) -> bool;

  auto Capacity(const AccessType type) -> uint32_t;
  auto AccessElement(const image_id_t& id, const AccessType type)
      -> std::optional<std::weak_ptr<Image>>;
  void RecordAccess(const image_id_t& id, const AccessType type);
  void RemoveRecord(const image_id_t& id, const AccessType type);
  auto Evict(const AccessType type) -> std::optional<std::weak_ptr<Image>>;
  auto CacheContains(const image_id_t& id, const AccessType type) -> bool;

  void ResizeCache(const uint32_t new_capacity, const AccessType type);

  void Flush();
  void Clear();
};
};  // namespace puerhlab