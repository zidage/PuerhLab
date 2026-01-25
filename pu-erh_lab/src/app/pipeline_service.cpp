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

#include "app/pipeline_service.hpp"

#include <cstdint>
#include <memory>
#include <mutex>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "type/type.hpp"

namespace puerhlab {
void PipelineMgmtService::HandleEviction(sl_element_id_t evicted_id) {
  auto it = loaded_pipelines_.find(evicted_id);
  if (it != loaded_pipelines_.end()) {
    auto pipeline_guard = it->second;
    if (!pipeline_guard->pinned_) {
      if (pipeline_guard->dirty_) {
        storage_service_->GetElementController().UpdatePipelineByElementId(
            evicted_id, pipeline_guard->pipeline_);
      }
      loaded_pipelines_.erase(it);
    } else {
      // This pipeline is still pinned, resize the cache and keep it in the cache
      auto keys = pipeline_cache_.GetLRUKeys();
      pipeline_cache_.Resize(keys.size() + 5);
      pipeline_cache_.RecordAccess(evicted_id, evicted_id);
    }
  }
}

auto PipelineMgmtService::LoadPipeline(sl_element_id_t id) -> std::shared_ptr<PipelineGuard> {
  std::unique_lock<std::mutex> guard(lock_);

  if (pipeline_cache_.Contains(id)) {
    auto cached_id = pipeline_cache_.AccessElement(id);
    if (cached_id.has_value()) {
      auto it = loaded_pipelines_.find(cached_id.value());
      if (it != loaded_pipelines_.end()) {
        it->second->pinned_ = true;
        it->second->id_     = id;
        return it->second;
      }
    }
  } else {
    std::shared_ptr<CPUPipelineExecutor> pipeline;
    std::shared_ptr<PipelineGuard>       pipeline_guard;
    try {
      pipeline               = storage_service_->GetElementController().GetPipelineByElementId(id);
      pipeline_guard         = std::make_shared<PipelineGuard>();
      pipeline_guard->dirty_ = false;
    } catch (std::exception& e) {
      throw std::runtime_error(
          "[ERROR] PipelineMgmtService: Failed to load pipeline from storage for element ID " +
          std::to_string(id) + ": " + e.what());
    }
    if (pipeline == nullptr) {
      pipeline = std::make_shared<CPUPipelineExecutor>();
      pipeline->SetBoundFile(id);
      pipeline_guard->dirty_ = true;
    }

    pipeline_guard->pipeline_              = std::move(pipeline);
    pipeline_guard->id_                    = id;
    pipeline_guard->pinned_                = true;
    std::optional<sl_element_id_t> evicted = pipeline_cache_.RecordAccess_WithEvict(id, id);
    if (evicted.has_value()) {
      HandleEviction(evicted.value());
    }
    loaded_pipelines_[id] = pipeline_guard;
    // If no eviction happened, and the cache size is still in "boost" range, resize it
    if (!evicted.has_value() && loaded_pipelines_.size() + 1 > default_cache_capacity_) {
      pipeline_cache_.Resize(loaded_pipelines_.size() - 1);
    }
    return pipeline_guard;
  }
  throw std::runtime_error("[ERROR] PipelineMgmtService: Failed to load pipeline.");
}

void PipelineMgmtService::SavePipeline(std::shared_ptr<PipelineGuard> pipeline) {
  if (!pipeline || !pipeline->dirty_) {
    return;
  }

  std::unique_lock<std::mutex>   guard(lock_);
  // Save the pipeline back to the cache
  sl_element_id_t                id      = pipeline->id_;
  // Store it back to the pipeline cache
  std::optional<sl_element_id_t> evicted = pipeline_cache_.RecordAccess_WithEvict(id, id);
  if (evicted.has_value()) {
    HandleEviction(evicted.value());
  }
  // Unpin the pipeline after saving
  pipeline->pinned_     = false;
  loaded_pipelines_[id] = pipeline;

  // If eviction did not happen, but the cache size is still in "boost" range, resize it
  if (!evicted.has_value() && loaded_pipelines_.size() + 1 > default_cache_capacity_) {
    pipeline_cache_.Resize(static_cast<uint32_t>(loaded_pipelines_.size() - 1));
  }
}

void PipelineMgmtService::Sync() {
  std::unique_lock<std::mutex> guard(lock_);
  for (auto& pair : loaded_pipelines_) {
    auto pipeline_guard = pair.second;
    if (pipeline_guard->dirty_) {
      storage_service_->GetElementController().UpdatePipelineByElementId(pipeline_guard->id_,
                                                                         pipeline_guard->pipeline_);
      pipeline_guard->dirty_ = false;
    }
  }
}
}  // namespace puerhlab