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

#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/highgui.hpp>
#include <stdexcept>

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "image/image.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace {
auto IsResizeEffectivelyNoOp(const IOperatorBase& op, int width, int height) -> bool {
  const nlohmann::json params = op.GetParams();
  if (!params.contains("resize") || !params["resize"].is_object()) {
    return false;
  }

  const auto& resize = params["resize"];
  const bool  enable_scale = resize.value("enable_scale", false);
  const bool  enable_roi   = resize.value("enable_roi", false);
  const int   maximum_edge = resize.value("maximum_edge", 2048);

  if (!enable_scale && !enable_roi) {
    return true;
  }

  if (width <= 0 || height <= 0) {
    return false;
  }

  return std::max(width, height) <= maximum_edge;
}

auto IsCropRotateEffectivelyNoOp(const IOperatorBase& op) -> bool {
  const nlohmann::json params = op.GetParams();
  if (!params.contains("crop_rotate") || !params["crop_rotate"].is_object()) {
    return false;
  }

  const auto& crop_rotate = params["crop_rotate"];
  if (!crop_rotate.value("enabled", false)) {
    return true;
  }

  const bool  enable_crop = crop_rotate.value("enable_crop", false);
  if (!enable_crop) {
    return true;
  }

  const float angle = crop_rotate.value("angle_degrees", 0.0f);
  if (std::abs(angle) > 1e-4f) {
    return false;
  }

  if (!crop_rotate.contains("crop_rect") || !crop_rotate["crop_rect"].is_object()) {
    return true;
  }
  const auto& rect = crop_rotate["crop_rect"];
  const float x    = rect.value("x", 0.0f);
  const float y    = rect.value("y", 0.0f);
  const float w    = rect.value("w", 1.0f);
  const float h    = rect.value("h", 1.0f);

  return std::abs(x) <= 1e-4f && std::abs(y) <= 1e-4f && std::abs(w - 1.0f) <= 1e-4f &&
         std::abs(h - 1.0f) <= 1e-4f;
}
}  // namespace

PipelineStage::PipelineStage(PipelineStageName stage, bool enable_cache, bool is_streamable)
    : is_streamable_(is_streamable), enable_cache_(enable_cache), stage_(stage) {
  stage_role_ = DetermineStageRole(stage_, is_streamable_);
  operators_  = std::make_unique<std::map<OperatorType, OperatorEntry>>();
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
    if (it->second.op_) {
      const auto current_param = it->second.op_->GetParams();
      if (current_param == param) {
        return;
      }
    }
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

void PipelineStage::EnableOperator(OperatorType op_type, bool enable,
                                   OperatorParams& global_params) {
  EnableOperator(op_type, enable);
  auto it = operators_->find(op_type);
  if (it != operators_->end()) {
    // Keep value parameters synced, then apply explicit enable/disable state.
    it->second.op_->SetGlobalParams(global_params);
    it->second.op_->EnableGlobalParams(global_params, enable);
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
    input_img_.reset();
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

  switch (stage_role_) {
    case StageRole::DescriptorOnly:
      return ApplyDescriptorOnly();
    case StageRole::CpuOperators:
      return ApplyCpuOperators();
    case StageRole::GpuStreamable:
      return ApplyGpuStream(global_params);
    case StageRole::GpuOperators:
      return ApplyGpuOperators();
  }

  // Fallback, should never hit.
  return input_img_;
}

auto PipelineStage::HasInput() -> bool { return input_set_; }
void PipelineStage::ResetAll() {
  operators_->clear();
  input_img_.reset();
  output_cache_.reset();
  input_set_          = false;
  input_cache_valid_  = false;
  output_cache_valid_ = false;
  dependents_         = nullptr;
  prev_stage_         = nullptr;
  next_stage_         = nullptr;
  input_img_          = nullptr;
  output_cache_       = nullptr;
  input_set_          = false;
}

void PipelineStage::ResetRuntimeResources(RuntimeResetMode mode) {
  const auto clear_intermediate_buffers = [this]() {
    input_img_.reset();
    output_cache_.reset();
    input_set_          = false;
    input_cache_valid_  = false;
    output_cache_valid_ = false;
  };

  const auto release_gpu_resources = [this]() {
    if (stage_role_ != StageRole::GpuStreamable) {
      return;
    }
    gpu_executor_.ReleaseResources();
    gpu_setup_done_ = false;
  };

  switch (mode) {
    case RuntimeResetMode::InvalidateCache:
      output_cache_.reset();
      output_cache_valid_ = false;
      if (next_stage_) {
        next_stage_->SetInputCacheValid(false);
      }
      return;
    case RuntimeResetMode::ClearIntermediateBuffers:
      clear_intermediate_buffers();
      return;
    case RuntimeResetMode::ReleaseGpuResources:
      release_gpu_resources();
      return;
    case RuntimeResetMode::ClearIntermediateBuffersAndGpu:
      clear_intermediate_buffers();
      release_gpu_resources();
      return;
  }
}

auto PipelineStage::DetermineStageRole(PipelineStageName stage, bool is_streamable) -> StageRole {
  switch (stage) {
    case PipelineStageName::Image_Loading:
    case PipelineStageName::Geometry_Adjustment:
      return StageRole::GpuOperators;
    case PipelineStageName::Merged_Stage:
      return StageRole::GpuStreamable;
    default:
      (void)is_streamable;
      return StageRole::DescriptorOnly;
  }
}

bool PipelineStage::HasEnabledOperator() const {
  return std::any_of(operators_->begin(), operators_->end(),
                     [](const auto& entry) { return entry.second.enable_; });
}

std::shared_ptr<ImageBuffer> PipelineStage::ApplyDescriptorOnly() {
  if (!enable_cache_) {
    SetOutputCacheValid(false);
    if (next_stage_) {
      next_stage_->SetInputCacheValid(false);
    }
    return input_img_;
  }

  if (CacheValid()) {
    std::cout << "RAW Cache VALID" << std::endl;
    if (next_stage_ && next_stage_->CacheValid()) {
      return nullptr;
    }
    return output_cache_;
  }

  output_cache_ = input_img_;
  SetOutputCacheValid(true);
  if (next_stage_) {
    next_stage_->SetInputCacheValid(true);
  }

  return output_cache_;
}

std::shared_ptr<ImageBuffer> PipelineStage::ApplyGpuOperators() {
  auto execute_ops = [&]() {
    if (!HasEnabledOperator()) return input_img_;

    if (stage_ == PipelineStageName::Geometry_Adjustment) {
      int width = 0;
      int height = 0;
      if (input_img_->gpu_data_valid_) {
        const auto& gpu_data = input_img_->GetGPUData();
        width                = gpu_data.cols;
        height               = gpu_data.rows;
      } else if (input_img_->cpu_data_valid_) {
        const auto& cpu_data = input_img_->GetCPUData();
        width                = cpu_data.cols;
        height               = cpu_data.rows;
      }

      bool has_enabled = false;
      bool all_noop    = true;
      for (const auto& [op_type, op_entry] : *operators_) {
        if (!op_entry.enable_ || !op_entry.op_) {
          continue;
        }
        has_enabled = true;
        if ((op_type == OperatorType::RESIZE &&
             !IsResizeEffectivelyNoOp(*op_entry.op_, width, height)) ||
            (op_type == OperatorType::CROP_ROTATE && !IsCropRotateEffectivelyNoOp(*op_entry.op_)) ||
            (op_type != OperatorType::RESIZE && op_type != OperatorType::CROP_ROTATE)) {
          all_noop = false;
          break;
        }
      }

      // Geometry stage is frequently configured as "full-res passthrough" during editing.
      // Reuse upstream cache directly to avoid holding another full-resolution GpuMat.
      if (has_enabled && all_noop) {
        return input_img_;
      }
    }

    auto current_img = std::make_shared<ImageBuffer>();
    if (input_img_->gpu_data_valid_ && !input_img_->buffer_valid_) {
      auto& input_gpu_mat = input_img_->GetGPUData();
      current_img->InitGPUData(input_gpu_mat.cols, input_gpu_mat.rows, input_gpu_mat.type());
      auto& output_gpu_mat = current_img->GetGPUData();
      input_gpu_mat.copyTo(output_gpu_mat);
      current_img->gpu_data_valid_ = true;
    } else if (input_img_->buffer_valid_) {
      auto buffer = input_img_->GetBuffer();
      current_img = std::make_shared<ImageBuffer>(std::move(buffer));
    }

    for (const auto& op_entry : *operators_) {
      if (op_entry.second.enable_) {
        op_entry.second.op_->ApplyGPU(current_img);
      }
    }
    current_img->gpu_data_valid_ = true;
    return current_img;
  };

  if (!input_img_->gpu_data_valid_ && !input_img_->buffer_valid_) {
    input_img_->SyncToGPU();
  }
  // This is different from ApplyGpuStream, as this function applies individual GPU operators
  if (!enable_cache_) {
    SetOutputCacheValid(false);
    if (next_stage_) {
      next_stage_->SetInputCacheValid(false);
    }
    output_cache_ = execute_ops();
    SetOutputCacheValid(false);
    if (next_stage_) {
      next_stage_->SetInputCacheValid(false);
    }
    return output_cache_;
  }

  if (CacheValid()) {
    if (next_stage_ && next_stage_->CacheValid()) {
      return nullptr;
    }
    return output_cache_;
  }

  output_cache_ = execute_ops();
  SetOutputCacheValid(true);
  if (next_stage_) {
    next_stage_->SetInputCacheValid(true);
  }
  return output_cache_;
}

std::shared_ptr<ImageBuffer> PipelineStage::ApplyCpuOperators() {
  auto execute_ops = [&]() {
    if (!HasEnabledOperator()) return input_img_;
    auto current_img = std::make_shared<ImageBuffer>(input_img_->Clone());
    for (const auto& op_entry : *operators_) {
      if (op_entry.second.enable_) {
        op_entry.second.op_->Apply(current_img);
      }
    }
    return current_img;
  };

  if (!enable_cache_) {
    output_cache_ = execute_ops();
    SetOutputCacheValid(false);
    if (next_stage_) {
      next_stage_->SetInputCacheValid(false);
    }
    return output_cache_;
  }

  if (CacheValid()) {
    if (next_stage_ && next_stage_->CacheValid()) {
      return nullptr;
    }
    return output_cache_;
  }

  output_cache_ = execute_ops();
  SetOutputCacheValid(true);
  if (next_stage_) {
    next_stage_->SetInputCacheValid(true);
    if (next_stage_->stage_role_ == StageRole::GpuStreamable) {
      output_cache_->SyncToGPU();
    }
  }
  return output_cache_;
}

std::shared_ptr<ImageBuffer> PipelineStage::ApplyGpuStream(OperatorParams& global_params) {
  if (!gpu_setup_done_) {
    SetGPUExecutor();
  }

  output_cache_ = std::make_shared<ImageBuffer>();
  gpu_executor_.SetParams(global_params);
  gpu_executor_.SetInputImage(input_img_);
  if (force_cpu_output_) {
    gpu_executor_.Execute(output_cache_);
  } else {
    gpu_executor_.Execute(nullptr);
  }

  if (force_cpu_output_) {
    try {
      output_cache_->SyncToCPU();
      output_cache_->ReleaseGPUData();  // Free GPU memory after sync
    } catch (const std::exception&) {
      // Keep GPU result if CPU sync fails; caller will validate.
    }
  }

  // GPU stage output is not cached; downstream stages receive fresh data.
  SetOutputCacheValid(false);
  if (next_stage_) {
    next_stage_->SetInputCacheValid(false);
  }
  return output_cache_;
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
    if (!op_entry.op_ || op_entry.op_->GetOperatorType() == OperatorType::RESIZE ||
        op_entry.op_->GetOperatorType() == OperatorType::UNKNOWN) {
      continue;
    }
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
    if (op_type != OperatorType::UNKNOWN && op_type != OperatorType::RESIZE) {
      SetOperator(op_type, params);
    }
  }
}

void PipelineStage::ImportStageParams(const nlohmann::json& j, OperatorParams& global_params) {
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
    if (op_type != OperatorType::UNKNOWN && op_type != OperatorType::RESIZE) {
      SetOperator(op_type, params, global_params);
    }
  }
}
};  // namespace puerhlab
