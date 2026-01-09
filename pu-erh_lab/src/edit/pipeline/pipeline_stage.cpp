//  Copyright 2025 Yurun Zi
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

#include "edit/pipeline/pipeline_stage.hpp"

#include <memory>
#include <opencv2/highgui.hpp>
#include <stdexcept>

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
PipelineStage::PipelineStage(PipelineStageName stage, bool enable_cache, bool is_streamable)
    : is_streamable_(is_streamable), enable_cache_(enable_cache), stage_(stage) {
  operators_ = std::make_unique<std::map<OperatorType, OperatorEntry>>();
  if (stage_ == PipelineStageName::Image_Loading) {
    input_cache_valid_ = true;  // No input for image loading stage, so input cache is always valid
  }
}

void PipelineStage::SetOperator(OperatorType op_type, nlohmann::json param) {
  auto it = operators_->find(op_type);
  if (it == operators_->end()) {
    auto op = OperatorFactory::Instance().Create(op_type, param);
    operators_->emplace(op_type, OperatorEntry{true, op});
    SetOutputCacheValid(false);
    if (dependents_) dependents_->SetOutputCacheValid(false);
  } else {
    (it->second).op_->SetParams(param);
    SetOutputCacheValid(false);
    if (dependents_) dependents_->SetOutputCacheValid(false);
  }
}

void PipelineStage::SetOperator(OperatorType op_type, nlohmann::json param,
                                OperatorParams& global_params) {
  SetOperator(op_type, param);
  auto it = operators_->find(op_type);
  if (it != operators_->end()) {
    it->second.op_->SetGlobalParams(global_params);
  }
}

auto PipelineStage::GetOperator(OperatorType op_type) const -> std::optional<OperatorEntry*> {
  auto it = operators_->find(op_type);
  if (it == operators_->end()) {
    return std::nullopt;
  }
  return &(it->second);
}

void PipelineStage::EnableOperator(OperatorType op_type, bool enable) {
  auto it = operators_->find(op_type);
  if (it != operators_->end()) {
    if (it->second.enable_ != enable) {
      SetOutputCacheValid(false);
      if (dependents_) dependents_->SetInputCacheValid(false);
    }
    it->second.enable_ = enable;
  }
}

void PipelineStage::SetInputImage(std::shared_ptr<ImageBuffer> input) {
  input_img_ = input;
  input_set_ = true;
}

void PipelineStage::SetInputCacheValid(bool valid) {
  input_cache_valid_ = valid;
  if (!valid) {
    // Invalidate output cache if input cache is invalidated
    SetOutputCacheValid(false);
  }
}

void PipelineStage::SetOutputCacheValid(bool valid) {
  if (output_cache_valid_ != valid && !valid) {
    output_cache_.reset();
    if (next_stage_) {
      next_stage_->SetInputCacheValid(false);
    }
  }
  output_cache_valid_ = valid;
}

auto PipelineStage::ApplyStage(OperatorParams& global_params) -> std::shared_ptr<ImageBuffer> {
  if (!input_set_) {
    throw std::runtime_error("PipelineExecutor: No valid input image set");
  }

  if (enable_cache_) {
    if (CacheValid()) {
      if (next_stage_ && next_stage_->CacheValid()) {
        // Both input and output cache are valid, skip processing and copying
        return nullptr;
      }
      // Output cache is valid, but next stage's input cache is not valid, copy output cache to next
      return output_cache_;
    }

    bool has_enabled_op = operators_->size() > 0;
    if (has_enabled_op && !is_streamable_) {
      // Non-streamable stage with enabled operators, process the entire image at once
      auto current_img = std::make_shared<ImageBuffer>(input_img_->Clone());
      for (const auto& op_entry : *operators_) {
        if (op_entry.second.enable_) {
          op_entry.second.op_->Apply(current_img);
        }
      }
      output_cache_ = current_img;
    } else if (is_streamable_) {
      // Streamable stage with enabled operators, use tile scheduler
      if (!gpu_setup_done_) {
        // If tile scheduler is not set, create one
        // SetStaticTileScheduler();
        SetGPUExecutor();
      }
      output_cache_ = std::make_shared<ImageBuffer>();
      // _static_tile_scheduler->SetInputImage(_input_img);
      gpu_executor_.SetParams(global_params);
      gpu_executor_.Execute(output_cache_);
      // output_cache_->SyncToCPU(); // TODO: remove this for future optimization
      // auto current_img = _static_tile_scheduler->ApplyOps(global_params);
      // auto 
      // output_cache_    = current_img;

    } else {
      output_cache_ = input_img_;
    }

    // input_set_ = false;
    // input_img_.reset();
    SetOutputCacheValid(true);
    if (next_stage_) {
      next_stage_->SetInputCacheValid(true);
    }
    return output_cache_;

  } else {
    // No caching, always process the image without using streaming
    bool has_enabled_op = operators_->size() > 0;
    if (has_enabled_op) {
      std::shared_ptr<ImageBuffer> current_img = input_img_;
      for (const auto& op_entry : *operators_) {
        if (op_entry.second.enable_) {
          op_entry.second.op_->Apply(current_img);
        }
      }
      output_cache_ = current_img;
    } else {
      output_cache_ = input_img_;
    }

    // Set to false to avoid using cache
    SetOutputCacheValid(false);
    if (next_stage_) {
      next_stage_->SetInputCacheValid(false);
    }
    return output_cache_;
  }
}

auto PipelineStage::HasInput() -> bool { return input_set_; }
void PipelineStage::ResetAll() {
  operators_->clear();
  input_img_.reset();
  output_cache_.reset();
  input_set_          = false;
  input_cache_valid_  = false;
  output_cache_valid_ = false;
  dependents_ = nullptr;
  prev_stage_   = nullptr;
  next_stage_   = nullptr;
  input_img_    = nullptr;
  output_cache_ = nullptr;
  input_set_    = false;
}

void PipelineStage::ResetCache() {
  output_cache_.reset();
  output_cache_valid_ = false;
  if (next_stage_) {
    next_stage_->SetInputCacheValid(false);
  }
}

auto PipelineStage::GetStageNameString() const -> std::string {
  switch (stage_) {
    case PipelineStageName::Image_Loading:
      return "Image Loading";
    case PipelineStageName::To_WorkingSpace:
      return "To Working Space";
    case PipelineStageName::Basic_Adjustment:
      return "Basic Adjustment";
    case PipelineStageName::Color_Adjustment:
      return "Color Adjustment";
    case PipelineStageName::Detail_Adjustment:
      return "Detail Adjustment";
    case PipelineStageName::Output_Transform:
      return "Output Transform";
    case PipelineStageName::Geometry_Adjustment:
      return "Geometry Adjustment";
    case PipelineStageName::Merged_Stage:
      return "Merged Stage";
    default:
      return "Unknown Stage";
  }

  return "Unknown Stage";
}

auto PipelineStage::ExportStageParams() const -> nlohmann::json {
  nlohmann::json inner;
  for (const auto& [op_type, op_entry] : *operators_) {
    inner[op_entry.op_->GetScriptName()] = op_entry.ExportOperatorParams();
  }
  nlohmann::json j;
  j.emplace(GetStageNameString(), std::move(inner));
  return j;
}

void PipelineStage::ImportStageParams(const nlohmann::json& j) {
  ResetAll();

  std::string stage_name = GetStageNameString();
  if (!j.contains(stage_name)) {
    return;
  }
  nlohmann::json stage_json = j[stage_name];
  for (auto& [op_name, op_json] : stage_json.items()) {
    if (!op_json.contains("params")) {
      continue;
    }
    nlohmann::json params  = op_json.value("params", nlohmann::json::object());
    OperatorType   op_type = op_json.value("type", OperatorType::UNKNOWN);
    SetOperator(op_type, params);
  }
}
};  // namespace puerhlab