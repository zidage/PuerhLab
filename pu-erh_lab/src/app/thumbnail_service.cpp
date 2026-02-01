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
#include <format>
#include <memory>

#include "app/pipeline_service.hpp"
#include "renderer/pipeline_task.hpp"
#include "app/render_service.hpp"

namespace puerhlab {

struct ThumbnailService::State {
  static constexpr size_t                    default_cache_size_ = 64;

  std::shared_ptr<SleeveServiceImpl>         sleeve_service_     = nullptr;
  std::shared_ptr<ImagePoolService>          image_pool_service_ = nullptr;
  std::shared_ptr<PipelineMgmtService>       pipeline_service_   = nullptr;

  std::mutex                                 cache_lock_;

  LRUCache<sl_element_id_t, sl_element_id_t> thumbnail_cache_;
  std::unordered_map<sl_element_id_t, std::shared_ptr<ThumbnailGuard>> thumbnail_cache_data_{};
  std::unordered_map<sl_element_id_t, std::vector<ThumbnailCallback>>  pending_{};

  // Pipeline scheduler (global/shared), must outlive tasks.
  std::shared_ptr<PipelineScheduler> pipeline_scheduler_ = nullptr;

  State(std::shared_ptr<SleeveServiceImpl> sleeve_service,
        std::shared_ptr<ImagePoolService> image_pool_service,
        std::shared_ptr<PipelineMgmtService> pipeline_service)
      : sleeve_service_(std::move(sleeve_service)),
        image_pool_service_(std::move(image_pool_service)),
        pipeline_service_(std::move(pipeline_service)),
        thumbnail_cache_(default_cache_size_) {
    pipeline_scheduler_ = RenderService::GetThumbnailOrExportScheduler();
  }
};

ThumbnailService::ThumbnailService(std::shared_ptr<SleeveServiceImpl>   sleeve_service,
                                 std::shared_ptr<ImagePoolService>    image_pool_service,
                                 std::shared_ptr<PipelineMgmtService> pipeline_service)
    : state_(std::make_shared<State>(std::move(sleeve_service),
                                    std::move(image_pool_service),
                                    std::move(pipeline_service))) {}

void ThumbnailService::GetThumbnail(sl_element_id_t id, image_id_t image_id,
                                    ThumbnailCallback callback, bool pin_if_found,
                                    CallbackDispatcher dispatcher) {
  auto st = state_;
  if (!st || !st->image_pool_service_ || !st->pipeline_service_ || !st->pipeline_scheduler_) {
    throw std::runtime_error("[ERROR] ThumbnailService: Services not initialized.");
  }

  std::shared_ptr<ThumbnailGuard> guard = nullptr;

  // Fast path: cache hit. Never invoke callbacks while holding cache_lock_.
  bool cache_hit = false;

  {
    std::unique_lock lock(st->cache_lock_);
    if (st->thumbnail_cache_.Contains(id)) {
      // Found in cache
      guard = st->thumbnail_cache_data_[id];
      if (pin_if_found) {
        guard->pin_count_++;
      }

      cache_hit = true;
    }
  }

  if (cache_hit) {
    if (dispatcher) {
      dispatcher([callback, guard]() { callback(guard); });
    } else {
      callback(guard);
    }
    return;
  }

  // Not found in cache, check if already pending
  std::unique_lock lock(st->cache_lock_);
  if (st->pending_.find(id) != st->pending_.end()) {
    // Already pending, add to callback list
    st->pending_[id].push_back(callback);
    return;
  } else {
    // Not pending, add to pending list

    st->pending_[id] = {callback};
    std::shared_ptr<PipelineGuard> pipeline;
    try {
      pipeline = st->pipeline_service_->LoadPipeline(id);
    } catch (std::exception& e) {
      st->pending_.erase(id);
      std::cout << "[ERROR] ThumbnailService: Failed to load pipeline for file ID " << id
                << ": " << e.what() << std::endl;
    }
    if (!pipeline) {
      throw std::runtime_error(
          std::format("[ERROR] ThumbnailService: Pipeline for file ID {} not available.", id));
    }
    pipeline->pipeline_->SetForceCPUOutput(true);
    // Set thumbnail task
    PipelineTask thumb_task;
    thumb_task.pipeline_executor_ = pipeline->pipeline_;

    // Read Image descriptor from pool service
  auto img_result               = st->image_pool_service_->Read<std::shared_ptr<Image>>(
        image_id, [](std::shared_ptr<Image> img) { return img; });
    if (!img_result) {
      throw std::runtime_error(
          std::format("[ERROR] ThumbnailService: Image with ID {} not found in pool.", id));
    }
    thumb_task.input_desc_                        = std::move(img_result);
    thumb_task.options_.render_desc_.render_type_ = RenderType::THUMBNAIL;
    thumb_task.options_.is_blocking_              = false;
    thumb_task.options_.is_callback_              = true;

    thumb_task.callback_ = [st, id, dispatcher, pipeline](ImageBuffer& result_buffer) {
      std::shared_ptr<ThumbnailGuard> guard = std::make_shared<ThumbnailGuard>();
      guard->thumbnail_buffer_ = std::make_unique<ImageBuffer>(std::move(result_buffer));
      guard->pin_count_        = 1;

      std::vector<ThumbnailCallback> callbacks;

      {
        std::unique_lock lock(st->cache_lock_);
        auto             evicted = st->thumbnail_cache_.RecordAccess_WithEvict(id, id);
        HandleEvict(*st, evicted);
        st->thumbnail_cache_data_[id] = guard;

        // Move callbacks out; invoke outside lock.
        auto it = st->pending_.find(id);
        if (it != st->pending_.end()) {
          callbacks = std::move(it->second);
          st->pending_.erase(it);
        }
      }

      for (const auto& cb : callbacks) {
        if (dispatcher) {
          dispatcher([cb, guard]() { cb(guard); });
        } else {
          cb(guard);
        }
      }

      // Save pipeline back to storage outside lock.
      st->pipeline_service_->SavePipeline(pipeline);
    };

    st->pipeline_scheduler_->ScheduleTask(std::move(thumb_task));
  }
}

void ThumbnailService::ReleaseThumbnail(sl_element_id_t sleeve_element_id) {
  auto st = state_;
  if (!st) {
    return;
  }

  std::unique_lock lock(st->cache_lock_);
  if (st->thumbnail_cache_data_.find(sleeve_element_id) != st->thumbnail_cache_data_.end()) {
    auto guard = st->thumbnail_cache_data_[sleeve_element_id];
    if (guard->pin_count_ > 0) {
      guard->pin_count_--;
    }
  }
}

void ThumbnailService::InvalidateThumbnail(sl_element_id_t sleeve_element_id) {
  auto st = state_;
  if (!st) {
    return;
  }

  std::unique_lock lock(st->cache_lock_);
  // TODO: This is abstraction leak; we should provide an interface in ThumbnailService
  st->thumbnail_cache_.RemoveRecord(sleeve_element_id);
  st->thumbnail_cache_data_.erase(sleeve_element_id);
  // Note: we intentionally do not cancel in-flight renders in pending_.
  // A subsequent GetThumbnail() after invalidation will schedule a fresh render
  // once the cache miss is observed.
}

void ThumbnailService::HandleEvict(State& st, std::optional<sl_element_id_t> evicted_id) {
  if (evicted_id.has_value()) {
    auto id = evicted_id.value();
    if (st.thumbnail_cache_data_.find(id) != st.thumbnail_cache_data_.end()) {
      auto guard = st.thumbnail_cache_data_[id];
      if (guard->pin_count_ == 0) {
        // RAII: erasing from map drops the shared_ptr reference.
        // When the last reference is dropped, ThumbnailGuard destructor runs,
        // which destroys thumbnail_buffer_, which in turn releases all image data.
        st.thumbnail_cache_data_.erase(id);
      } else {
        // Re-insert into cache since it's still pinned
        // "Boost" the cache size to avoid immediate eviction
        st.thumbnail_cache_.Resize(static_cast<uint32_t>(st.thumbnail_cache_data_.size() + 5));
        st.thumbnail_cache_.RecordAccess(id, id);
      }
    }
  } else {
    // No eviction happened, check cache size
    if (st.thumbnail_cache_data_.size() > State::default_cache_size_) {
      // Try to reduce size
      st.thumbnail_cache_.Resize(static_cast<uint32_t>(st.thumbnail_cache_data_.size() - 1));
    }
  }
}
};  // namespace puerhlab