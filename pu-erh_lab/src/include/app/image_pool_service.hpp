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
struct ImagePoolSyncErrorResult {
  image_id_t  image_id_;
  std::string message_{""};
};

struct ImagePoolSyncStatus {
  std::vector<image_id_t>               synced_images_{};
  std::vector<ImagePoolSyncErrorResult> failed_images_{};
};

class ImagePoolService {
 public:
  virtual ~ImagePoolService()                                                     = default;

  virtual void CreateEmpty(std::function<void(std::shared_ptr<Image>)> operation) = 0;

  template <typename TResult>
  auto Read(image_id_t image_id, std::function<TResult(std::shared_ptr<Image>)> operation)
      -> TResult;

  template <typename TResult>
  auto Write(image_id_t image_id, std::function<TResult(std::shared_ptr<Image>)> operation)
      -> std::pair<TResult, ImagePoolSyncStatus>;

  template <typename TResult>
  auto Write_NoSync(image_id_t image_id, std::function<TResult(std::shared_ptr<Image>)> operation)
      -> TResult;

  virtual void         Remove(image_id_t image_id) = 0;

  virtual auto SyncWithStorage() -> ImagePoolSyncStatus = 0;
};

class ImagePoolServiceImpl final : public ImagePoolService {
 private:
  std::unique_ptr<ImagePoolManager> pool_manager_;
  // This should be injected from sleeve service in future
  std::shared_ptr<StorageService>   storage_service_;

  std::mutex                        pool_lock_;

 public:
  ImagePoolServiceImpl() = delete;
  ImagePoolServiceImpl(std::shared_ptr<StorageService> storage_service, image_id_t start_id);

  ~ImagePoolServiceImpl() = default;

  void CreateEmpty(std::function<void(std::shared_ptr<Image>)> operation) override {
    std::unique_lock lock(pool_lock_);
    if (!pool_manager_) {
      throw std::runtime_error("[ERROR] ImagePoolService: Pool manager is not initialized.");
    }
    auto img = pool_manager_->InsertEmpty();
    img->MarkSyncState(ImageSyncState::UNSYNCED);
    operation(img);
  }

  template <typename TResult>
  auto Read(image_id_t image_id, std::function<TResult(std::shared_ptr<Image>)> operation)
      -> TResult {
    std::unique_lock lock(pool_lock_);
    if (!pool_manager_) {
      throw std::runtime_error("[ERROR] ImagePoolService: Pool manager is not initialized.");
    }

    // Check if the image exists in the pool
    std::shared_ptr<Image> img = nullptr;
    if (pool_manager_->PoolContains(image_id)) {
      img = pool_manager_->GetPool()[image_id];
    } else {
      // Check in the storage
      auto& img_ctrl = storage_service_->GetImageController();
      img            = img_ctrl.GetImageById(image_id);
      if (img) {
        pool_manager_->Insert(img);
      }
    }
    if (!img) {
      throw std::runtime_error(std::format(
          "[ERROR] ImagePoolService: Image with ID {} not found in storage.", image_id));
    }

    if constexpr (std::is_void_v<TResult>) {
      operation(img);
      return;
    } else {
      return operation(img);
    }
  }

  template <typename TResult>
  auto Write(image_id_t image_id, std::function<TResult(std::shared_ptr<Image>)> operation)
      -> std::pair<TResult, ImagePoolSyncStatus> {
    std::unique_lock       lock(pool_lock_);
    ImagePoolSyncStatus    status;
    std::shared_ptr<Image> img;

    // Check if the image exists in the pool
    if (pool_manager_->PoolContains(image_id)) {
      img = pool_manager_->GetPool()[image_id];
    } else {
      // Check in the storage
      auto result = storage_service_->GetImageController().GetImageById(image_id);
      if (!result) {
        status.failed_images_.push_back(
            {image_id, std::format("Image with ID {} not found in storage.", image_id)});
        return status;
      }
      img = result;
      pool_manager_->Insert(img);
    }

    // Perform the operation
    img->MarkSyncState(ImageSyncState::MODIFIED);
    if constexpr (std::is_void_v<TResult>) {
      operation(img);
      status = SyncWithStorage();
      return status;
    } else {
      TResult result = operation(img);
      status         = SyncWithStorage();
      return {result, status};
    }
  }

  template <typename TResult>
  auto Write_NoSync(image_id_t image_id, std::function<TResult(std::shared_ptr<Image>)> operation)
      -> TResult {
    std::unique_lock lock(pool_lock_);
    std::shared_ptr<Image> img;

    // Check if the image exists in the pool
    if (pool_manager_->PoolContains(image_id)) {
      img = pool_manager_->GetPool()[image_id];
    } else {
      // Check in the storage
      auto result = storage_service_->GetImageController().GetImageById(image_id);
      if (!result) {
        throw std::runtime_error(
            std::format("[ERROR] ImagePoolService: Image with ID {} not found in storage.",
                        image_id));
      }
      img = result;
      pool_manager_->Insert(img);
    }

    // Perform the operation
    img->MarkSyncState(ImageSyncState::MODIFIED);

    if constexpr (std::is_void_v<TResult>) {
      operation(img);
      return;
    } else {
      return operation(img);
    }
  }

  void Remove(image_id_t image_id) override;
  auto SyncWithStorage() -> ImagePoolSyncStatus override;
};
};  // namespace puerhlab