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
#include <algorithm>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "type/type.hpp"

namespace puerhlab {
namespace {
void ResetToDefaults(OperatorParams& params) {
  // OperatorParams has const members, so it is not assignable.
  // Reinitialize in-place to restore default values.
  std::destroy_at(&params);
  std::construct_at(&params);
}

void EnsureDefaultOutputTransform(CPUPipelineExecutor& exec) {
  auto& global_params = exec.GetGlobalParams();
  auto& output_stage  = exec.GetStage(PipelineStageName::Output_Transform);

  // Older stored pipelines (or partially-initialized ones) might miss the ODT descriptor.
  // Without it, the GPU path won't have precomputed ODT tables and can render black.
  if (!output_stage.GetOperator(OperatorType::ODT).has_value()) {
    nlohmann::json output_params;
    output_params["aces_odt"] = {{"encoding_space", "rec709"},
                                 {"encoding_etof", "gamma_2_2"},
                                 {"limiting_space", "rec709"},
                                 {"peak_luminance", 100.0f}};
    output_stage.SetOperator(OperatorType::ODT, output_params, global_params);
  }
}

void EnsureDefaultColorTemp(CPUPipelineExecutor& exec) {
  auto& global_params = exec.GetGlobalParams();
  auto& to_ws_stage   = exec.GetStage(PipelineStageName::To_WorkingSpace);

  if (to_ws_stage.GetOperator(OperatorType::COLOR_TEMP).has_value()) {
    return;
  }

  std::string mode = "as_shot";
  float       cct  = 6500.0f;
  float       tint = 0.0f;

  auto& raw_stage = exec.GetStage(PipelineStageName::Image_Loading);
  auto  raw_entry = raw_stage.GetOperator(OperatorType::RAW_DECODE);
  if (raw_entry.has_value() && raw_entry.value() && raw_entry.value()->op_) {
    const nlohmann::json raw_params = raw_entry.value()->op_->GetParams();
    if (raw_params.contains("raw") && raw_params["raw"].is_object()) {
      const auto& raw = raw_params["raw"];
      if (raw.contains("use_camera_wb") && raw["use_camera_wb"].is_boolean() &&
          !raw["use_camera_wb"].get<bool>()) {
        mode = "custom";
      }
      if (raw.contains("user_wb") && raw["user_wb"].is_number()) {
        cct = std::clamp(raw["user_wb"].get<float>(), 2000.0f, 15000.0f);
      }
    }
  }

  nlohmann::json color_temp_params;
  color_temp_params["color_temp"] = {
      {"mode", mode},
      {"cct", cct},
      {"tint", tint},
      {"resolved_cct", cct},
      {"resolved_tint", tint},
  };
  to_ws_stage.SetOperator(OperatorType::COLOR_TEMP, color_temp_params, global_params);
}

void ResyncGlobalParamsFromOperators(CPUPipelineExecutor& exec) {
  // Global params are consumed/mutated during GPU parameter conversion (dirty flags cleared).
  // Cached pipelines also release GPU resources when returned to the service.
  // Rebuild global params from operator params so ODT/LMT GPU resources are re-uploaded.
  auto& global_params = exec.GetGlobalParams();
  ResetToDefaults(global_params);

  for (int i = 0; i < static_cast<int>(PipelineStageName::Stage_Count); ++i) {
    auto& stage = exec.GetStage(static_cast<PipelineStageName>(i));
    for (auto& [op_type, op_entry] : stage.GetAllOperators()) {
      (void)op_type;
      if (!op_entry.op_) {
        continue;
      }
      op_entry.op_->SetGlobalParams(global_params);
    }
  }
}
}  // namespace

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
    if (pipeline_guard->pin_count_ == 0) {
      pipeline_guard->pinned_ = false;
      std::unique_lock<std::mutex> render_guard(pipeline_guard->pipeline_->GetRenderLock());
      if (pipeline_guard->dirty_) {
        storage_service_->GetElementController().UpdatePipelineByElementId(
            candidate, pipeline_guard->pipeline_);
      }
      // Clear intermediate buffers before removing from cache to ensure timely memory release
      pipeline_guard->pipeline_->ClearAllIntermediateBuffers();
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
        // If the pipeline was previously returned to cache (unpinned), it likely had its
        // execution stages reset (e.g. to detach frame sinks). Re-initialize it here so callers
        // that don't explicitly call SetExecutionStages() won't pay the cost or crash.
        if (!it->second->pinned_) {
          it->second->pipeline_->SetBoundFile(id);
          it->second->pipeline_->SetExecutionStages();
          // Reset transient render/cache state to a consistent FAST_PREVIEW baseline.
          it->second->pipeline_->SetRenderRegion(0, 0, 1.0f);
          it->second->pipeline_->SetRenderRes(false, 4096);
          it->second->pipeline_->SetForceCPUOutput(false);
          it->second->pipeline_->SetEnableCache(true);
          it->second->pipeline_->SetDecodeRes(DecodeRes::FULL);

          EnsureDefaultOutputTransform(*it->second->pipeline_);
          EnsureDefaultColorTemp(*it->second->pipeline_);
          ResyncGlobalParamsFromOperators(*it->second->pipeline_);
        }

        it->second->pin_count_++;
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

    // Ensure the loaded pipeline is bound to the requested element id.
    pipeline->SetBoundFile(id);

    EnsureDefaultOutputTransform(*pipeline);
    EnsureDefaultColorTemp(*pipeline);
    ResyncGlobalParamsFromOperators(*pipeline);

    pipeline->SetExecutionStages(); // TODO: Use service as the only way to set/reset execution stages
    pipeline_guard->pipeline_              = std::move(pipeline);
    pipeline_guard->id_                    = id;
    pipeline_guard->pinned_                = true;
    pipeline_guard->pin_count_             = 1;
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

  std::unique_lock<std::mutex> guard(lock_);

  if (pipeline->pin_count_ > 0) {
    pipeline->pin_count_--;
  }
  const bool last_pin = (pipeline->pin_count_ == 0);
  pipeline->pinned_   = !last_pin;

  // Shared by multiple callers (e.g. thumbnail + export): only the last owner may release/reset.
  if (!last_pin) {
    return;
  }

  // Always clear intermediate buffers and GPU resources when returning a pipeline to cache.
  // This prevents large cached allocations (and any frame-sink related state) from leaking across
  // editor sessions and hurting interactive performance.
  {
    std::unique_lock<std::mutex> render_guard(pipeline->pipeline_->GetRenderLock());
    pipeline->pipeline_->ClearAllIntermediateBuffers();
    pipeline->pipeline_->ResetExecutionStages();
  }

  if (!pipeline->dirty_) {
    return;
  }

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
