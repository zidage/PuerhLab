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

#include "storage/image_pool/image_pool_manager.hpp"

#include <cstdint>
#include <memory>
#include <optional>

#include "decoders/raw_decoder.hpp"
#include "type/type.hpp"

namespace puerhlab {
ImagePoolManager::ImagePoolManager()
    : capacity_thumb_(default_capacity_thumb_), capacity_full_(default_capacity_full_) {}

ImagePoolManager::ImagePoolManager(uint32_t capacity_thumb, uint32_t capacity_full)
    : capacity_thumb_(capacity_thumb), capacity_full_(capacity_full) {}

/**
 * @brief
 *
 * @return std::unordered_map<image_id_t, std::shared_ptr<Image>>
 */
auto ImagePoolManager::GetPool() -> std::unordered_map<image_id_t, std::shared_ptr<Image>>& {
  return image_pool_;
}

/**
 * @brief Insert an image into the image pool
 *
 * @param img
 */
void ImagePoolManager::Insert(const std::shared_ptr<Image> img) {
  image_pool_[img->image_id_] = img;
}

/**
 * @brief Check whether an image with the given id exists in the image pool
 *
 * @param id
 * @return true image exists
 * @return false image not exists
 */
auto ImagePoolManager::PoolContains(const image_id_t& id) -> bool {
  return image_pool_.contains(id);
}

auto ImagePoolManager::Capacity(AccessType type) -> uint32_t {
  switch (type) {
    case AccessType::FULL_IMG: {
      return capacity_full_;
    }
    case AccessType::THUMB: {
      return capacity_thumb_;
    }
    case AccessType::META:
      return image_pool_.size();
  }
  return 0;
}
/**
 * @brief Access an with-data image from the cache
 *
 * @param id
 * @param type
 * @return std::optional<std::weak_ptr<Image>>
 */
auto ImagePoolManager::AccessElement(const image_id_t& id, const AccessType type)
    -> std::optional<std::weak_ptr<Image>> {
  switch (type) {
    case AccessType::FULL_IMG: {
      auto it = cache_map_full_.find(id);
      if (it == cache_map_full_.end()) {
        return std::nullopt;
      }
      // LRU, place the most recent record to the front
      cache_list_full_.splice(cache_list_full_.begin(), cache_list_full_, it->second);
      return image_pool_[it->first];
    }
    case AccessType::THUMB: {
      auto it = cache_map_thumb_.find(id);
      if (it == cache_map_thumb_.end()) {
        return std::nullopt;
      }
      // LRU, place the most recent record to the front
      cache_list_thumb_.splice(cache_list_thumb_.begin(), cache_list_thumb_, it->second);
      return image_pool_[it->first];
    }
    case AccessType::META:
      // For empty image, return it from the pool directly
      return image_pool_.at(id);
  }
  return std::nullopt;
}

/**
 * @brief Add a with-data image into the cache
 *
 * @param id
 * @param type
 */
void ImagePoolManager::RecordAccess(const image_id_t& id, const AccessType type) {
  switch (type) {
    case AccessType::FULL_IMG: {
      auto it = cache_map_full_.find(id);
      if (it == cache_map_full_.end()) {
        if (cache_list_full_.size() >= capacity_full_) {
          Evict(type);
        }
        // Place the new-added record to the front
        cache_list_full_.push_front(id);
        cache_map_full_[id] = cache_list_full_.begin();
      } else {
        cache_list_full_.splice(cache_list_full_.begin(), cache_list_full_, it->second);
        if (cache_list_full_.front() != id) {
          cache_list_full_.front() = id;
        }
        with_full_.insert(id);
      }
      break;
    }
    case AccessType::THUMB: {
      auto it = cache_map_thumb_.find(id);
      if (it == cache_map_thumb_.end()) {
        if (cache_list_thumb_.size() >= capacity_thumb_) {
          Evict(type);
        }
        cache_list_thumb_.push_front(id);
        cache_map_thumb_[id] = cache_list_thumb_.begin();
      } else {
        cache_list_thumb_.splice(cache_list_thumb_.begin(), cache_list_thumb_, it->second);
        if (cache_list_thumb_.front() != id) {
          cache_list_thumb_.front() = id;
        }
        with_thumb_.insert(id);
      }
      break;
    }
    case AccessType::META: {
    }
  }
}

/**
 * @brief Remove a record according to its id
 *
 * @param id
 * @param type
 */
void ImagePoolManager::RemoveRecord(const image_id_t& id, const AccessType type) {
  switch (type) {
    case AccessType::FULL_IMG: {
      auto it = cache_map_full_.find(id);
      if (it != cache_map_full_.end()) {
        // capture values before erasing from the map (erasing invalidates `it`)
        auto key     = it->first;   // image id
        auto list_it = it->second;  // iterator into cache_list_full_

        cache_list_full_.erase(list_it);
        cache_map_full_.erase(it);

        // with_full_ likely holds image ids, so erase by key
        with_full_.erase(key);
        auto img = image_pool_[key];
        img->ClearThumbnail();
      }
      break;
    }
    case AccessType::THUMB: {
      auto it = cache_map_thumb_.find(id);
      if (it != cache_map_thumb_.end()) {
        auto key     = it->first;
        auto list_it = it->second;

        cache_list_thumb_.erase(list_it);
        cache_map_thumb_.erase(it);

        with_thumb_.erase(key);

        auto img = image_pool_[key];
        img->ClearThumbnail();
      }
      break;
    }
    default:
      break;
  }
}

/**
 * @brief Evict an image from the cache
 *
 * @param type
 * @return std::optional<std::weak_ptr<Image>>
 */
auto ImagePoolManager::Evict(const AccessType type) -> std::optional<std::weak_ptr<Image>> {
  switch (type) {
    case AccessType::FULL_IMG: {
      if (cache_list_full_.empty()) {
        return std::nullopt;
      }
      auto last = cache_list_full_.end();
      do {
        --last;
      } while (image_pool_[*last]->full_pinned_ && last != cache_list_full_.begin());

      if (image_pool_[*last]->full_pinned_) {
        return std::nullopt;
      }
      cache_map_full_.erase(*last);
      with_full_.erase(*last);
      auto evicted_img = image_pool_[*last];
      cache_list_full_.pop_back();
      evicted_img->ClearData();
      return evicted_img;
    }
    case AccessType::THUMB: {
      if (cache_list_thumb_.empty()) {
        return std::nullopt;
      }
      auto last = cache_list_thumb_.end();
      do {
        --last;
      } while (image_pool_[*last]->thumb_pinned_ && last != cache_list_thumb_.begin());

      if (image_pool_[*last]->thumb_pinned_) {
        return std::nullopt;
      }
      cache_map_thumb_.erase(*last);
      with_thumb_.erase(*last);
      auto evicted_img = image_pool_[*last];
      cache_list_thumb_.pop_back();
      evicted_img->ClearThumbnail();
      return evicted_img;
    }
    case AccessType::META: {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

/**
 * @brief Check whether an image resides in the cache
 *
 * @param id
 * @param type
 * @return true
 * @return false
 */
auto ImagePoolManager::CacheContains(const image_id_t& id, const AccessType type) -> bool {
  switch (type) {
    case AccessType::FULL_IMG: {
      return cache_map_full_.contains(id);
    }
    case AccessType::THUMB: {
      return cache_map_thumb_.contains(id);
    }
    case AccessType::META: {
      return image_pool_.contains(id);
    }
  }
  return false;
}

void ImagePoolManager::ResizeCache(const uint32_t new_capacity, const AccessType type) {
  switch (type) {
    case AccessType::FULL_IMG: {
      while (cache_list_full_.size() > new_capacity) {
        Evict(type);
      }
      capacity_full_ = new_capacity;
    }
    case AccessType::THUMB: {
      while (cache_list_thumb_.size() > new_capacity) {
        Evict(type);
      }
      capacity_thumb_ = new_capacity;
    }
    case AccessType::META: {
    }
  }
}

/**
 * @brief Flush the cache and clear all the corresponding image data
 *
 */
void ImagePoolManager::Flush() {
  cache_list_full_.clear();
  cache_list_thumb_.clear();
  cache_map_full_.clear();
  cache_map_thumb_.clear();

  for (auto& id : with_thumb_) {
    image_pool_.at(id)->ClearThumbnail();
  }

  for (auto& id : with_full_) {
    image_pool_.at(id)->ClearData();
  }
}

/**
 * @brief Clear the whole image pool as well as the cache
 *
 */
void ImagePoolManager::Clear() {
  Flush();
  image_pool_.clear();
}

};  // namespace puerhlab