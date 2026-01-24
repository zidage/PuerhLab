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

#include "app/thumbnail_service.hpp"

#include <cstdint>

#include "renderer/pipeline_task.hpp"


namespace puerhlab {
void ThumbnailService::GetThumbnail(sl_element_id_t id, ThumbnailCallback callback,
                                    bool pin_if_found, CallbackDispatcher dispatcher) {
  if (!image_pool_service_ || !pipeline_service_ || !pipeline_scheduler_) {
    throw std::runtime_error("[ERROR] ThumbnailService: Services not initialized.");
  }

  std::shared_ptr<ThumbnailGuard> guard = nullptr;

  {
    std::unique_lock lock(cache_lock_);
    if (thumbnail_cache_.Contains(id)) {
      // Found in cache
      guard = thumbnail_cache_data_[id];
      if (pin_if_found) {
        guard->pin_count_++;
      }

      if (dispatcher) {
        dispatcher([callback, guard]() { callback(guard); });
      } else {
        callback(guard);
      }
      return;
    }
  }

  // Not found in cache, check if already pending
  std::unique_lock lock(cache_lock_);
  if (pending_.find(id) != pending_.end()) {
    // Already pending, add to callback list
    pending_[id].push_back(callback);
    return;
  } else {
    // Not pending, add to pending list
    pending_[id]  = {callback};
    auto pipeline = pipeline_service_->LoadPipeline(id);
    if (!pipeline) {
      throw std::runtime_error(
          std::format("[ERROR] ThumbnailService: Pipeline for file ID {} not available.", id));
    }
    pipeline->pipeline_->SetForceCPUOutput(true);
    // Set thumbnail task
    PipelineTask thumb_task;
    thumb_task.pipeline_executor_ = pipeline->pipeline_;

    // Read Image descriptor from pool service
    auto img_result               = image_pool_service_->Read<std::shared_ptr<Image>>(
        id, [](std::shared_ptr<Image> img) { return img; });
    if (!img_result) {
      throw std::runtime_error(
          std::format("[ERROR] ThumbnailService: Image with ID {} not found in pool.", id));
    }
    thumb_task.input_desc_                        = std::move(img_result);
    thumb_task.options_.render_desc_.render_type_ = RenderType::THUMBNAIL;
    thumb_task.options_.is_blocking_              = false;
    thumb_task.options_.is_callback_              = true;

    thumb_task.callback_ = [this, id, dispatcher, pipeline](ImageBuffer& result_buffer) {
      std::shared_ptr<ThumbnailGuard> guard = std::make_shared<ThumbnailGuard>();
      guard->thumbnail_buffer_ = std::make_unique<ImageBuffer>(std::move(result_buffer));
      guard->pin_count_        = 1;

      {
        std::unique_lock lock(cache_lock_);
        auto             evicted = thumbnail_cache_.RecordAccess_WithEvict(id, id);
        HandleEvict(evicted);
        thumbnail_cache_data_[id] = guard;

        // Call all pending callbacks
        auto callbacks            = pending_[id];
        pending_.erase(id);
        for (const auto& cb : callbacks) {
          if (dispatcher) {
            dispatcher([cb, guard]() { cb(guard); });
            continue;
          }
          cb(guard);
        }
        // Finally, save pipeline back to storage
        pipeline_service_->SavePipeline(pipeline);
      }
    };

    pipeline_scheduler_->ScheduleTask(std::move(thumb_task));
  }
}

void ThumbnailService::ReleaseThumbnail(sl_element_id_t sleeve_element_id) {
  std::unique_lock lock(cache_lock_);
  if (thumbnail_cache_data_.find(sleeve_element_id) != thumbnail_cache_data_.end()) {
    auto guard = thumbnail_cache_data_[sleeve_element_id];
    if (guard->pin_count_ > 0) {
      guard->pin_count_--;
    }
  }
}

void ThumbnailService::HandleEvict(std::optional<sl_element_id_t> evicted_id) {
  if (evicted_id.has_value()) {
    auto id = evicted_id.value();
    if (thumbnail_cache_data_.find(id) != thumbnail_cache_data_.end()) {
      auto guard = thumbnail_cache_data_[id];
      if (guard->pin_count_ == 0) {
        thumbnail_cache_data_.erase(id);
      } else {
        // Re-insert into cache since it's still pinned
        // "Boost" the cache size to avoid immediate eviction
        thumbnail_cache_.Resize(static_cast<uint32_t>(thumbnail_cache_data_.size() + 5));
        thumbnail_cache_.RecordAccess(id, id);
      }
    }
  } else {
    // No eviction happened, check cache size
    if (thumbnail_cache_data_.size() > default_cache_size_) {
      // Try to reduce size
      thumbnail_cache_.Resize(static_cast<uint32_t>(thumbnail_cache_data_.size() - 1));
    }
  }
}
};  // namespace puerhlab