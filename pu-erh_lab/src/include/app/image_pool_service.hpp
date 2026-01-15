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

#pragma once

#include <cstdint>
#include <memory>

#include "image/image.hpp"
#include "image/image_buffer.hpp"
#include "sleeve/storage_service.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"

namespace puerhlab {

struct ThumbStatus {
  ThumbState  state_       = ThumbState::NOT_PRESENT;
  std::string fail_reason_ = "";
};

class ImagePoolService {
 public:
  virtual ~ImagePoolService()                                                 = default;

  // image objects
  virtual auto GetOrCreate(image_id_t id) -> std::shared_ptr<Image>           = 0;
  virtual auto TryGet(image_id_t id) -> std::optional<std::shared_ptr<Image>> = 0;

  // thumbnail state machine
  virtual auto GetThumbStatus(image_id_t id) const -> ThumbStatus             = 0;
  virtual auto MarkThumbPending(image_id_t id) -> bool                        = 0;
  // epoch unmatch, skip and return
  virtual void MarkThumbReady(image_id_t id)                                  = 0;
  virtual void MarkThumbFailed(image_id_t id, std::string message)            = 0;

  // epoch unmatch, skip and return
  virtual void PutThumbnail(image_id_t id, const ImageBuffer& thumb)          = 0;

  // cache control (thumb only)
  virtual void SetThumbCacheCapacity(uint32_t capacity)                       = 0;
  virtual void FlushThumbCache()                                              = 0;
};

class ImagePoolServiceImpl final : public ImagePoolService {
 private:
  std::unique_ptr<ImagePoolManager> pool_manager_;
  // This should be injected from sleeve service in future
  std::shared_ptr<StorageService>   storage_service_;

  std::mutex                        pool_lock_;

 public:
  ImagePoolServiceImpl() = delete;
  ImagePoolServiceImpl(std::shared_ptr<StorageService> storage_service);

  ~ImagePoolServiceImpl() = default;

  auto GetOrCreate(image_id_t id) -> std::shared_ptr<Image> override;
  auto TryGet(image_id_t id) -> std::optional<std::shared_ptr<Image>> override;

  void SyncWithStorage();

  auto GetThumbStatus(image_id_t id) const -> ThumbStatus override;
  auto MarkThumbPending(image_id_t id) -> bool override;

  void MarkThumbReady(image_id_t id) override;
  void MarkThumbFailed(image_id_t id, std::string message) override;

  void PutThumbnail(image_id_t id, const ImageBuffer& thumb) override;

  void SetThumbCacheCapacity(uint32_t capacity) override;
  void FlushThumbCache() override;
};
};  // namespace puerhlab