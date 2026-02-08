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

#include <functional>
#include <memory>
#include <unordered_map>

#include "app/image_pool_service.hpp"
#include "edit/pipeline/pipeline.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "sleeve/storage_service.hpp"
#include "type/type.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {

struct PipelineGuard {
  std::shared_ptr<CPUPipelineExecutor> pipeline_;
  sl_element_id_t                      id_;
  bool                                 dirty_  = false;
  bool                                 pinned_ = false;
  size_t                               pin_count_ = 0;
};

class PipelineMgmtService final {
 private:
  std::shared_ptr<StorageService>                                     storage_service_;

  LRUCache<sl_element_id_t, sl_element_id_t>                          pipeline_cache_;

  std::unordered_map<sl_element_id_t, std::shared_ptr<PipelineGuard>> loaded_pipelines_;

  std::mutex                                                          lock_;

  static constexpr size_t                                             default_cache_capacity_ = 16;

  void HandleEviction(sl_element_id_t evicted_id);

 public:
  PipelineMgmtService() = delete;
  explicit PipelineMgmtService(std::shared_ptr<StorageService> storage_service)
      : storage_service_(storage_service),
        pipeline_cache_(default_cache_capacity_),
        loaded_pipelines_() {}

  void SavePipeline(std::shared_ptr<PipelineGuard> pipeline);

  auto LoadPipeline(sl_element_id_t id) -> std::shared_ptr<PipelineGuard>;

  void Sync();
};
}  // namespace puerhlab
