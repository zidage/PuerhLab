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

#include "edit/pipeline/pipeline_cpu.hpp"

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include "edit/operators/cst/lmt_op.hpp"
#include "edit/pipeline/pipeline.hpp"

#ifdef HAVE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#include <opencv2/opencv.hpp>

#include "edit/operators/CPU_kernels/cpu_kernels.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/operators/utils/color_conversion.hpp"
#include "edit/pipeline/pipeline_stage.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
CPUPipelineExecutor::CPUPipelineExecutor()
    : enable_cache_(false),
      stages_({{PipelineStageName::Image_Loading, enable_cache_, false},
               {PipelineStageName::Geometry_Adjustment, enable_cache_, false},
               {PipelineStageName::To_WorkingSpace, enable_cache_, true},
               {PipelineStageName::Basic_Adjustment, enable_cache_, true},
               {PipelineStageName::Color_Adjustment, enable_cache_, true},
               {PipelineStageName::Detail_Adjustment, enable_cache_, true},
               {PipelineStageName::Output_Transform, enable_cache_, true}}) {}

void CPUPipelineExecutor::ResetStages() {
  for (size_t i = 0; i < stages_.size(); i++) {
    stages_[i].ResetAll();
  }
}

void CPUPipelineExecutor::SetEnableCache(bool enable_cache) {
  if (enable_cache_ == enable_cache) return;
  enable_cache_ = enable_cache;
  // Reinitialize stages with the new cache setting
  ResetStages();
}

CPUPipelineExecutor::CPUPipelineExecutor(bool enable_cache)
    : enable_cache_(enable_cache),
      stages_({{PipelineStageName::Image_Loading, enable_cache_, false},
               {PipelineStageName::Geometry_Adjustment, enable_cache_, false},
               {PipelineStageName::To_WorkingSpace, enable_cache_, true},
               {PipelineStageName::Basic_Adjustment, enable_cache_, true},
               {PipelineStageName::Color_Adjustment, enable_cache_, true},
               {PipelineStageName::Detail_Adjustment, enable_cache_, true},
               {PipelineStageName::Output_Transform, enable_cache_, true}}) {}

auto CPUPipelineExecutor::GetBackend() -> PipelineBackend { return backend_; }

auto CPUPipelineExecutor::GetStage(PipelineStageName stage) -> PipelineStage& {
  return stages_[static_cast<int>(stage)];
}

auto CPUPipelineExecutor::Apply(std::shared_ptr<ImageBuffer> input)
    -> std::shared_ptr<ImageBuffer> {
  auto* first_stage = exec_stages_.front();
  if (!first_stage) {
    return input;
  }
  std::shared_ptr<ImageBuffer> output;
  if (!first_stage->CacheValid()) {
    output = std::make_shared<ImageBuffer>(input->Clone());
    for (auto* stage : exec_stages_) {
      stage->SetInputImage(output);
      output = stage->ApplyStage(global_params_);
    }
  } else {
    // If cache is valid, use cached output
    output = first_stage->GetOutputCache();
    for (auto* stage : exec_stages_) {
      if (stage != first_stage) {
        stage->SetInputImage(output);
        output = stage->ApplyStage(global_params_);
      }
    }
  }

  return output;
}

void CPUPipelineExecutor::SetPreviewMode(bool is_thumbnail) {
  is_thumbnail_     = is_thumbnail;

  thumbnail_params_ = {};  // TODO: Use default params for now
  if (!is_thumbnail_) {
    // Disable resizing in image loading stage
    stages_[static_cast<int>(PipelineStageName::Image_Loading)].EnableOperator(
        OperatorType::RESIZE,
        false);  // If RESIZE operator not exist, this function will do nothing
    return;
  }
  stages_[static_cast<int>(PipelineStageName::Geometry_Adjustment)].SetOperator(
      OperatorType::RESIZE, thumbnail_params_);
}

void CPUPipelineExecutor::SetExecutionStages() {
  exec_stages_.clear();
  std::vector<PipelineStage*> streamable_stages;

  auto merged = std::make_unique<PipelineStage>(PipelineStageName::Merged_Stage, true, true);

  exec_stages_.push_back(&stages_[static_cast<int>(PipelineStageName::Image_Loading)]);
  exec_stages_.push_back(&stages_[static_cast<int>(PipelineStageName::Geometry_Adjustment)]);
  exec_stages_.push_back(merged.get());

  merged_stages_ = std::move(merged);

  for (size_t i = 2; i < stages_.size(); i++) {
    PipelineStage& stage = stages_[i];
    stage.AddDependent(merged_stages_.get());
  }

  // Chain the execution stages
  for (size_t i = 0; i < exec_stages_.size(); i++) {
    PipelineStage* prev_stage = (i > 0) ? exec_stages_[i - 1] : nullptr;
    PipelineStage* next_stage = (i < exec_stages_.size() - 1) ? exec_stages_[i + 1] : nullptr;
    exec_stages_[i]->SetNeighbors(prev_stage, next_stage);
  }
}

void CPUPipelineExecutor::ResetExecutionStages() {
  for (auto& stage : stages_) {
    stage.ResetDependents();
    stage.ResetNeighbors();
    stage.ResetCache();
  }
  exec_stages_.clear();
  merged_stages_.reset();
}

auto CPUPipelineExecutor::ExportPipelineParams() const -> nlohmann::json {
  nlohmann::json j;
  for (const auto& stage : stages_) {
    nlohmann::json stage_json     = stage.ExportStageParams();
    j[stage.GetStageNameString()] = std::move(stage_json);
  }
  return j;
}

void CPUPipelineExecutor::ImportPipelineParams(const nlohmann::json& j) {
  ResetExecutionStages();
  for (auto& stage : stages_) {
    std::string stage_name = stage.GetStageNameString();
    if (j.contains(stage_name)) {
      nlohmann::json stage_json = j[stage_name];
      // When importing, stage's import function will do reset internally
      stage.ImportStageParams(stage_json);
    }
  }
}

};  // namespace puerhlab