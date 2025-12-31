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
    : _capacity_thumb(_default_capacity_thumb), _capacity_full(_default_capacity_full) {}

ImagePoolManager::ImagePoolManager(uint32_t capacity_thumb, uint32_t capacity_full)
    : _capacity_thumb(capacity_thumb), _capacity_full(capacity_full) {}

/**
 * @brief
 *
 * @return std::unordered_map<image_id_t, std::shared_ptr<Image>>
 */
auto ImagePoolManager::GetPool() -> std::unordered_map<image_id_t, std::shared_ptr<Image>>& {
  return _image_pool;
}

/**
 * @brief Insert an image into the image pool
 *
 * @param img
 */
void ImagePoolManager::Insert(const std::shared_ptr<Image> img) {
  _image_pool[img->_image_id] = img;
}

/**
 * @brief Check whether an image with the given id exists in the image pool
 *
 * @param id
 * @return true image exists
 * @return false image not exists
 */
auto ImagePoolManager::PoolContains(const image_id_t& id) -> bool {
  return _image_pool.contains(id);
}

auto ImagePoolManager::Capacity(AccessType type) -> uint32_t {
  switch (type) {
    case AccessType::FULL_IMG: {
      return _capacity_full;
    }
    case AccessType::THUMB: {
      return _capacity_thumb;
    }
    case AccessType::META:
      return _image_pool.size();
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
      auto it = _cache_map_full.find(id);
      if (it == _cache_map_full.end()) {
        return std::nullopt;
      }
      // LRU, place the most recent record to the front
      _cache_list_full.splice(_cache_list_full.begin(), _cache_list_full, it->second);
      return _image_pool[it->first];
    }
    case AccessType::THUMB: {
      auto it = _cache_map_thumb.find(id);
      if (it == _cache_map_thumb.end()) {
        return std::nullopt;
      }
      // LRU, place the most recent record to the front
      _cache_list_thumb.splice(_cache_list_thumb.begin(), _cache_list_thumb, it->second);
      return _image_pool[it->first];
    }
    case AccessType::META:
      // For empty image, return it from the pool directly
      return _image_pool.at(id);
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
      auto it = _cache_map_full.find(id);
      if (it == _cache_map_full.end()) {
        if (_cache_list_full.size() >= _capacity_full) {
          Evict(type);
        }
        // Place the new-added record to the front
        _cache_list_full.push_front(id);
        _cache_map_full[id] = _cache_list_full.begin();
      } else {
        _cache_list_full.splice(_cache_list_full.begin(), _cache_list_full, it->second);
        if (_cache_list_full.front() != id) {
          _cache_list_full.front() = id;
        }
        _with_full.insert(id);
      }
      break;
    }
    case AccessType::THUMB: {
      auto it = _cache_map_thumb.find(id);
      if (it == _cache_map_thumb.end()) {
        if (_cache_list_thumb.size() >= _capacity_thumb) {
          Evict(type);
        }
        _cache_list_thumb.push_front(id);
        _cache_map_thumb[id] = _cache_list_thumb.begin();
      } else {
        _cache_list_thumb.splice(_cache_list_thumb.begin(), _cache_list_thumb, it->second);
        if (_cache_list_thumb.front() != id) {
          _cache_list_thumb.front() = id;
        }
        _with_thumb.insert(id);
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
      auto it = _cache_map_full.find(id);
      if (it != _cache_map_full.end()) {
        // capture values before erasing from the map (erasing invalidates `it`)
        auto key     = it->first;   // image id
        auto list_it = it->second;  // iterator into _cache_list_full

        _cache_list_full.erase(list_it);
        _cache_map_full.erase(it);

        // _with_full likely holds image ids, so erase by key
        _with_full.erase(key);

        auto img = _image_pool[key];
        img->ClearThumbnail();
      }
      break;
    }
    case AccessType::THUMB: {
      auto it = _cache_map_thumb.find(id);
      if (it != _cache_map_thumb.end()) {
        auto key     = it->first;
        auto list_it = it->second;

        _cache_list_thumb.erase(list_it);
        _cache_map_thumb.erase(it);

        _with_thumb.erase(key);

        auto img = _image_pool[key];
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
      if (_cache_list_full.empty()) {
        return std::nullopt;
      }
      auto last = _cache_list_full.end();
      do {
        --last;
      } while (_image_pool[*last]->_full_pinned && last != _cache_list_full.begin());

      if (_image_pool[*last]->_full_pinned) {
        return std::nullopt;
      }
      _cache_map_full.erase(*last);
      _with_full.erase(*last);
      auto evicted_img = _image_pool[*last];
      _cache_list_full.pop_back();
      evicted_img->ClearData();
      return evicted_img;
    }
    case AccessType::THUMB: {
      if (_cache_list_thumb.empty()) {
        return std::nullopt;
      }
      auto last = _cache_list_thumb.end();
      do {
        --last;
      } while (_image_pool[*last]->_thumb_pinned && last != _cache_list_thumb.begin());

      if (_image_pool[*last]->_thumb_pinned) {
        return std::nullopt;
      }
      _cache_map_thumb.erase(*last);
      _with_thumb.erase(*last);
      auto evicted_img = _image_pool[*last];
      _cache_list_thumb.pop_back();
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
      return _cache_map_full.contains(id);
    }
    case AccessType::THUMB: {
      return _cache_map_thumb.contains(id);
    }
    case AccessType::META: {
      return _image_pool.contains(id);
    }
  }
  return false;
}

void ImagePoolManager::ResizeCache(const uint32_t new_capacity, const AccessType type) {
  switch (type) {
    case AccessType::FULL_IMG: {
      while (_cache_list_full.size() > new_capacity) {
        Evict(type);
      }
      _capacity_full = new_capacity;
    }
    case AccessType::THUMB: {
      while (_cache_list_thumb.size() > new_capacity) {
        Evict(type);
      }
      _capacity_thumb = new_capacity;
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
  _cache_list_full.clear();
  _cache_list_thumb.clear();
  _cache_map_full.clear();
  _cache_map_thumb.clear();

  for (auto& id : _with_thumb) {
    _image_pool.at(id)->ClearThumbnail();
  }

  for (auto& id : _with_full) {
    _image_pool.at(id)->ClearData();
  }
}

/**
 * @brief Clear the whole image pool as well as the cache
 *
 */
void ImagePoolManager::Clear() {
  Flush();
  _image_pool.clear();
}

};  // namespace puerhlab