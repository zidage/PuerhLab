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
#include <unordered_map>

#include "app/image_pool_service.hpp"
#include "app/pipeline_service.hpp"
#include "concurrency/thread_pool.hpp"
#include "image/image_buffer.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "type/type.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {
struct ThumbnailGuard {
  std::unique_ptr<ImageBuffer> thumbnail_buffer_ = nullptr;
  int                          pin_count_        = 0;
};

using ThumbnailCallback  = std::function<void(std::shared_ptr<ThumbnailGuard>)>;
using CallbackDispatcher = std::function<void(std::function<void()>)>;

class ThumbnailService {
 private:
  std::shared_ptr<ImagePoolService>          image_pool_service_ = nullptr;
  std::shared_ptr<PipelineMgmtService>       pipeline_service_   = nullptr;

  std::mutex                                 cache_lock_;

  static constexpr size_t                    default_cache_size_ = 64;

  LRUCache<sl_element_id_t, sl_element_id_t> thumbnail_cache_;  // Cache up to 64 thumbnails
  std::unordered_map<sl_element_id_t, std::shared_ptr<ThumbnailGuard>> thumbnail_cache_data_{};

  std::unordered_map<sl_element_id_t, std::vector<ThumbnailCallback>>  pending_{};

  // Pipeline scheduler, should be destoryed before any other services and data structures
  std::shared_ptr<PipelineScheduler> pipeline_scheduler_ = nullptr;

  void                               HandleEvict(std::optional<sl_element_id_t> evicted_id);

 public:
  ThumbnailService() = delete;
  ThumbnailService(std::shared_ptr<ImagePoolService>    image_pool_service,
                   std::shared_ptr<PipelineMgmtService> pipeline_service,
                   std::shared_ptr<PipelineScheduler>   pipeline_scheduler)
      : image_pool_service_(image_pool_service),
        pipeline_service_(pipeline_service),
        thumbnail_cache_(default_cache_size_),
        pipeline_scheduler_(pipeline_scheduler) {}
  ~ThumbnailService() = default;

  void GetThumbnail(sl_element_id_t id, ThumbnailCallback callback, bool pin_if_found = true,
                    CallbackDispatcher dispatcher = nullptr);

  void ReleaseThumbnail(sl_element_id_t sleeve_element_id);
};
};  // namespace puerhlab