//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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

  void DeletePipeline(sl_element_id_t id);

  void Sync();
};
}  // namespace puerhlab
