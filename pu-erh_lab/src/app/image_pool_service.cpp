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

#include "app/image_pool_service.hpp"

#include <cstdint>
#include <memory>

#include "image/image.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
ImagePoolServiceImpl::ImagePoolServiceImpl(std::shared_ptr<StorageService> storage_service,
                                           image_id_t                      start_id)
    : storage_service_(storage_service) {
  pool_manager_ = std::make_unique<ImagePoolManager>(start_id);
}

auto ImagePoolServiceImpl::CreateAndReturnPinnedEmpty() -> ImagePoolManager::PinnedImageHandle {
  std::unique_lock       lock(pool_lock_);
  return pool_manager_->CreateAndReturnPinnedEmpty();
}

void ImagePoolServiceImpl::Remove(image_id_t image_id) {
  std::unique_lock lock(pool_lock_);
  if (!pool_manager_) {
    throw std::runtime_error("[ERROR] ImagePoolService: Pool manager is not initialized.");
  }

  // Check if the image exists in the pool
  auto img = pool_manager_->GetImage(image_id);
  if (img) {
    img->MarkSyncState(ImageSyncState::DELETED);
  } else {
    // Check in the storage
    try {
      storage_service_->GetImageController().RemoveImageById(image_id);
    } catch (std::exception& e) {
      throw std::runtime_error(std::format(
          "[ERROR] ImagePoolService: Failed to remove image with ID {} from storage: {}", image_id,
          e.what()));
    }
  }

}

auto ImagePoolServiceImpl::SyncWithStorage() -> ImagePoolSyncStatus {
  std::unique_lock        lock(pool_lock_);
  auto&                   img_ctrl = storage_service_->GetImageController();

  std::vector<image_id_t> to_remove;
  ImagePoolSyncStatus     status;
  for (auto& [id, img] : pool_manager_->GetPool()) {
    if (img->GetSyncState() == ImageSyncState::UNSYNCED) {
      try {
        img_ctrl.AddImage(img);
        img->MarkSyncState(ImageSyncState::SYNCED);
        status.synced_images_.push_back(id);
      } catch (std::exception& e) {
        // TODO: Log the error
        status.failed_images_.push_back({id, e.what()});
        continue;
      }
    } else if (img->GetSyncState() == ImageSyncState::MODIFIED) {
      try {
        img_ctrl.UpdateImage(img);
        img->MarkSyncState(ImageSyncState::SYNCED);
        status.synced_images_.push_back(id);
      } catch (std::exception& e) {
        // TODO: Log the error
        status.failed_images_.push_back({id, e.what()});
        continue;
      }
    } else if (img->GetSyncState() == ImageSyncState::DELETED) {
      try {
        img_ctrl.RemoveImageById(id);
        to_remove.push_back(id);
        status.synced_images_.push_back(id);
      } catch (std::exception& e) {
        // TODO: Log the error
        status.failed_images_.push_back({id, e.what()});
        continue;
      }
    }
  }

  for (auto id : to_remove) {
    pool_manager_->GetPool().erase(id);
  }
  return status;
}

auto ImagePoolServiceImpl::GetCurrentID() -> image_id_t {
  std::unique_lock lock(pool_lock_);
  if (!pool_manager_) {
    throw std::runtime_error("[ERROR] ImagePoolService: Pool manager is not initialized.");
  }
  return pool_manager_->GetCurrentID();
}

};  // namespace puerhlab