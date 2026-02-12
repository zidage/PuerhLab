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
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "app/pipeline_service.hpp"
#include "app/render_service.hpp"
#include "renderer/pipeline_task.hpp"

namespace puerhlab {
namespace {
void DispatchThumbnailCallback(const ThumbnailCallback&           callback,
                               const CallbackDispatcher&          dispatcher,
                               const std::shared_ptr<ThumbnailGuard>& guard) {
  if (!callback) {
    return;
  }

  try {
    if (dispatcher) {
      dispatcher([callback, guard]() { callback(guard); });
    } else {
      callback(guard);
    }
  } catch (...) {
  }
}

auto IsRenderableThumbnailResult(const ImageBuffer& result_buffer) -> bool {
  return result_buffer.buffer_valid_ || result_buffer.cpu_data_valid_ || result_buffer.gpu_data_valid_;
}
}  // namespace

struct ThumbnailService::State {
  static constexpr size_t                    default_cache_size_ = 64;

  struct PendingCallback {
    ThumbnailCallback  callback_{};
    CallbackDispatcher dispatcher_{};
  };

  std::shared_ptr<SleeveServiceImpl>         sleeve_service_     = nullptr;
  std::shared_ptr<ImagePoolService>          image_pool_service_ = nullptr;
  std::shared_ptr<PipelineMgmtService>       pipeline_service_   = nullptr;

  std::mutex                                 cache_lock_;

  LRUCache<sl_element_id_t, sl_element_id_t> thumbnail_cache_;
  std::unordered_map<sl_element_id_t, std::shared_ptr<ThumbnailGuard>> thumbnail_cache_data_{};
  std::unordered_map<sl_element_id_t, std::vector<PendingCallback>>    pending_{};

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

  std::shared_ptr<ThumbnailGuard> guard;
  {
    std::unique_lock lock(st->cache_lock_);
    if (st->thumbnail_cache_.Contains(id)) {
      auto guard_it = st->thumbnail_cache_data_.find(id);
      if (guard_it != st->thumbnail_cache_data_.end() && guard_it->second) {
        guard = guard_it->second;
        if (pin_if_found) {
          guard->pin_count_++;
        }
      } else {
        // Keep LRU and payload map consistent.
        st->thumbnail_cache_.RemoveRecord(id);
      }
    }
  }

  if (guard) {
    DispatchThumbnailCallback(callback, dispatcher, guard);
    return;
  }

  {
    std::unique_lock lock(st->cache_lock_);
    auto it = st->pending_.find(id);
    if (it != st->pending_.end()) {
      State::PendingCallback pending_cb{};
      pending_cb.callback_   = std::move(callback);
      pending_cb.dispatcher_ = std::move(dispatcher);
      it->second.push_back(std::move(pending_cb));
      return;
    }
    State::PendingCallback pending_cb{};
    pending_cb.callback_   = std::move(callback);
    pending_cb.dispatcher_ = std::move(dispatcher);
    std::vector<State::PendingCallback> pending_callbacks;
    pending_callbacks.push_back(std::move(pending_cb));
    st->pending_.emplace(id, std::move(pending_callbacks));
  }

  auto fail_pending_request = [&](const std::string&               message,
                                  const std::shared_ptr<PipelineGuard>& pipeline) -> void {
    std::vector<State::PendingCallback> callbacks;
    {
      std::unique_lock lock(st->cache_lock_);
      auto             it = st->pending_.find(id);
      if (it != st->pending_.end()) {
        callbacks = std::move(it->second);
        st->pending_.erase(it);
      }
      st->thumbnail_cache_.RemoveRecord(id);
      st->thumbnail_cache_data_.erase(id);
    }

    if (pipeline) {
      try {
        st->pipeline_service_->SavePipeline(pipeline);
      } catch (...) {
      }
    }

    for (const auto& pending_cb : callbacks) {
      DispatchThumbnailCallback(pending_cb.callback_, pending_cb.dispatcher_, nullptr);
    }

    throw std::runtime_error(message);
  };

  std::shared_ptr<PipelineGuard> pipeline;
  try {
    pipeline = st->pipeline_service_->LoadPipeline(id);
  } catch (const std::exception& e) {
    fail_pending_request(
        std::format("[ERROR] ThumbnailService: Failed to load pipeline for file ID {}: {}", id,
                    e.what()),
        nullptr);
  } catch (...) {
    fail_pending_request(
        std::format(
            "[ERROR] ThumbnailService: Failed to load pipeline for file ID {}: unknown error.", id),
        nullptr);
  }

  if (!pipeline || !pipeline->pipeline_) {
    fail_pending_request(
        std::format("[ERROR] ThumbnailService: Pipeline for file ID {} not available.", id),
        nullptr);
  }

  pipeline->pipeline_->SetForceCPUOutput(true);

  std::shared_ptr<Image> img_result;
  try {
    img_result = st->image_pool_service_->Read<std::shared_ptr<Image>>(
        image_id, [](const std::shared_ptr<Image>& img) { return img; });
  } catch (const std::exception& e) {
    fail_pending_request(
        std::format("[ERROR] ThumbnailService: Failed to load image ID {} for element {}: {}",
                    image_id, id, e.what()),
        pipeline);
  } catch (...) {
    fail_pending_request(
        std::format(
            "[ERROR] ThumbnailService: Failed to load image ID {} for element {}: unknown error.",
            image_id, id),
        pipeline);
  }

  if (!img_result) {
    fail_pending_request(
        std::format("[ERROR] ThumbnailService: Image with ID {} not found in pool.", image_id),
        pipeline);
  }

  PipelineTask thumb_task;
  thumb_task.pipeline_executor_                 = pipeline->pipeline_;
  thumb_task.input_desc_                        = std::move(img_result);
  thumb_task.options_.render_desc_.render_type_ = RenderType::THUMBNAIL;
  thumb_task.options_.is_blocking_              = false;
  thumb_task.options_.is_callback_              = true;
  thumb_task.options_.is_seq_callback_          = false;

  thumb_task.callback_ = [st, id, pipeline](ImageBuffer& result_buffer) {
    std::shared_ptr<ThumbnailGuard>      guard;
    std::vector<State::PendingCallback> callbacks;

    {
      std::unique_lock lock(st->cache_lock_);

      auto             pending_it = st->pending_.find(id);
      const bool       request_active = (pending_it != st->pending_.end());
      if (request_active) {
        callbacks = std::move(pending_it->second);
        st->pending_.erase(pending_it);
      }

      const bool valid_result = IsRenderableThumbnailResult(result_buffer);
      if (request_active && valid_result) {
        guard                   = std::make_shared<ThumbnailGuard>();
        guard->thumbnail_buffer_ = std::make_unique<ImageBuffer>(std::move(result_buffer));
        guard->pin_count_        = 1;

        auto evicted = st->thumbnail_cache_.RecordAccess_WithEvict(id, id);
        HandleEvict(*st, evicted);
        st->thumbnail_cache_data_[id] = guard;
      } else {
        st->thumbnail_cache_.RemoveRecord(id);
        st->thumbnail_cache_data_.erase(id);
      }
    }

    try {
      st->pipeline_service_->SavePipeline(pipeline);
    } catch (...) {
    }

    for (const auto& pending_cb : callbacks) {
      DispatchThumbnailCallback(pending_cb.callback_, pending_cb.dispatcher_, guard);
    }
  };

  try {
    st->pipeline_scheduler_->ScheduleTask(std::move(thumb_task));
  } catch (const std::exception& e) {
    fail_pending_request(
        std::format("[ERROR] ThumbnailService: Failed to schedule thumbnail for element {}: {}", id,
                    e.what()),
        pipeline);
  } catch (...) {
    fail_pending_request(
        std::format(
            "[ERROR] ThumbnailService: Failed to schedule thumbnail for element {}: unknown error.",
            id),
        pipeline);
  }
}

void ThumbnailService::ReleaseThumbnail(sl_element_id_t sleeve_element_id) {
  auto st = state_;
  if (!st) {
    return;
  }

  std::unique_lock lock(st->cache_lock_);
  auto             it = st->thumbnail_cache_data_.find(sleeve_element_id);
  if (it == st->thumbnail_cache_data_.end() || !it->second) {
    st->thumbnail_cache_.RemoveRecord(sleeve_element_id);
    return;
  }

  auto guard = it->second;
  if (guard->pin_count_ > 0) {
    guard->pin_count_--;
  }

  if (guard->pin_count_ == 0) {
    // Out-of-range thumbnails should be released immediately.
    st->thumbnail_cache_.RemoveRecord(sleeve_element_id);
    st->thumbnail_cache_data_.erase(it);
  }
}

void ThumbnailService::InvalidateThumbnail(sl_element_id_t sleeve_element_id) {
  auto st = state_;
  if (!st) {
    return;
  }

  std::unique_lock lock(st->cache_lock_);
  // Cancel in-flight joiners for this id and drop any cached payload.
  st->pending_.erase(sleeve_element_id);
  st->thumbnail_cache_.RemoveRecord(sleeve_element_id);
  st->thumbnail_cache_data_.erase(sleeve_element_id);
}

void ThumbnailService::HandleEvict(State& st, std::optional<sl_element_id_t> evicted_id) {
  if (evicted_id.has_value()) {
    const auto id = evicted_id.value();
    auto       it = st.thumbnail_cache_data_.find(id);
    if (it != st.thumbnail_cache_data_.end() && it->second) {
      auto guard = it->second;
      if (guard->pin_count_ <= 0) {
        // RAII: erasing from map drops the shared_ptr reference.
        // When the last reference is dropped, ThumbnailGuard destructor runs,
        // which destroys thumbnail_buffer_, which in turn releases all image data.
        st.thumbnail_cache_data_.erase(it);
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
