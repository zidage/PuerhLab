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
  // If the would-be evicted pipeline is pinned, keep it and evict another entry instead.
  // This avoids unbounded cache growth during batch export when a pipeline is temporarily pinned.
  sl_element_id_t candidate = evicted_id;
  const size_t    max_attempts =
      loaded_pipelines_.empty() ? 1 : (loaded_pipelines_.size() + 1);

  for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
    auto it = loaded_pipelines_.find(candidate);
    if (it == loaded_pipelines_.end()) {
      return;
    }

    auto pipeline_guard = it->second;
    if (!pipeline_guard->pinned_) {
      if (pipeline_guard->dirty_) {
        storage_service_->GetElementController().UpdatePipelineByElementId(
            candidate, pipeline_guard->pipeline_);
      }
      // Clear intermediate buffers before removing from cache to ensure timely memory release
      pipeline_guard->pipeline_->ClearAllIntermediateBuffers();
      // Release persistent GPU allocations to avoid holding VRAM for evicted pipelines.
      pipeline_guard->pipeline_->ReleaseAllGPUResources();
      loaded_pipelines_.erase(it);
      return;
    }

    // Pinned: put it back into the LRU and evict a different entry.
    auto next = pipeline_cache_.RecordAccess_WithEvict(candidate, candidate);
    if (!next.has_value()) {
      return;
    }
    candidate = next.value();
  }

  // Fallback: if everything is pinned, allow temporary growth to avoid evicting in-use pipelines.
  auto keys = pipeline_cache_.GetLRUKeys();
  pipeline_cache_.Resize(static_cast<uint32_t>(keys.size() + 5));
  pipeline_cache_.RecordAccess(evicted_id, evicted_id);
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
      pipeline_guard->dirty_ = false;
    }

    pipeline->SetExecutionStages(); // TODO: Use service as the only way to set/reset execution stages
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
  if (!pipeline) {
    return;
  }

  // // Always clear intermediate buffers when returning pipeline to cache
  // // This releases memory from input/output images that are no longer needed
  // pipeline->pipeline_->ClearAllIntermediateBuffers();
  // // Export tends to run many distinct pipelines; avoid retaining large scratch buffers in cache.
  // pipeline->pipeline_->ReleaseAllGPUResources();

  pipeline->pipeline_->ResetExecutionStages();

  if (!pipeline->dirty_) {
    // Even if not dirty, we still need to unpin and return to cache
    std::unique_lock<std::mutex> guard(lock_);
    pipeline->pinned_ = false;
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
