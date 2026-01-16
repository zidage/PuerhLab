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
    : thumb_cache_(default_capacity_thumb_) {}

ImagePoolManager::ImagePoolManager(uint32_t capacity_thumb)
    : thumb_cache_(capacity_thumb) {}

ImagePoolManager::ImagePoolManager(uint32_t capacity_thumb, uint32_t start_id)
    : id_generator_(start_id), thumb_cache_(capacity_thumb) {}

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

auto ImagePoolManager::InsertEmpty() -> std::shared_ptr<Image> {
  auto new_id         = id_generator_.GenerateID();
  auto img            = std::make_shared<Image>(new_id);
  img->sync_state_    = ImageSyncState::UNSYNCED;
  image_pool_[new_id] = img;
  return img;
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

auto ImagePoolManager::CacheContains(const image_id_t& id) -> bool {
  return thumb_cache_.Contains(id);
}

void ImagePoolManager::RecordAccess(const image_id_t& id) { thumb_cache_.RecordAccess(id, id); }

void ImagePoolManager::RemoveRecord(const image_id_t& id) { thumb_cache_.RemoveRecord(id); }



/**
 * @brief Clear the whole image pool as well as the cache
 *
 */
void ImagePoolManager::Clear() {
  Flush();
  image_pool_.clear();
}

};  // namespace puerhlab