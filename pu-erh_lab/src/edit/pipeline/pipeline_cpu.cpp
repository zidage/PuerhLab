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

#include "edit/pipeline/pipeline.hpp"

#ifdef HAVE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#include <opencv2/opencv.hpp>

#include "edit/operators/op_base.hpp"
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
               {PipelineStageName::Output_Transform, enable_cache_, true}}) {
  render_params_["resize"] = {};
}

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
               {PipelineStageName::Output_Transform, enable_cache_, true}}) {
  render_params_["resize"] = {};
}

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
      stage->SetForceCPUOutput(force_cpu_output_);
      output = stage->ApplyStage(global_params_);
    }
  } else {
    // If cache is valid, use cached output
    output = first_stage->GetOutputCache();
    for (auto* stage : exec_stages_) {
      if (stage != first_stage) {
        stage->SetInputImage(output);
        stage->SetForceCPUOutput(force_cpu_output_);
        output = stage->ApplyStage(global_params_);
      }
    }
  }

  return output;
}

[[deprecated("SetPreviewMode is deprecated, set from pipeline scheduler instead")]] void
CPUPipelineExecutor::SetPreviewMode(bool is_thumbnail) {
  // is_thumbnail_  = is_thumbnail;

  // render_params_ = {};  // TODO: Use default params for now
  // if (!is_thumbnail_) {
  //   // Disable resizing in image loading stage
  //   stages_[static_cast<int>(PipelineStageName::Image_Loading)].EnableOperator(
  //       OperatorType::RESIZE,
  //       false);  // If RESIZE operator not exist, this function will do nothing
  //   return;
  // }
  // stages_[static_cast<int>(PipelineStageName::Geometry_Adjustment)].SetOperator(
  //     OperatorType::RESIZE, render_params_);
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

  // Chain the execution stages for caching
  for (size_t i = 0; i < exec_stages_.size(); i++) {
    PipelineStage* prev_stage = (i > 0) ? exec_stages_[i - 1] : nullptr;
    PipelineStage* next_stage = (i < exec_stages_.size() - 1) ? exec_stages_[i + 1] : nullptr;
    exec_stages_[i]->SetNeighbors(prev_stage, next_stage);
  }
}

void CPUPipelineExecutor::SetExecutionStages(IFrameSink* frame_sink) {
  SetExecutionStages();

  // Set frame sink for the last stage
  if (!exec_stages_.empty()) {
    exec_stages_.back()->SetFrameSink(frame_sink);
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

void CPUPipelineExecutor::SetRenderRegion(int x, int y, float scale_factor) {
  auto& resize_params         = render_params_["resize"];
  // render_params_["resize"] = {
  //   {"enable_roi", true},
  //   {"roi", {{"x", x}, {"y", y}, {"resize_factor", scale_factor}}},
  // };
  resize_params["enable_roi"] = false;
  resize_params["roi"]        = {{"x", x}, {"y", y}, {"resize_factor", scale_factor}};

  stages_[static_cast<int>(PipelineStageName::Geometry_Adjustment)].SetOperator(
      OperatorType::RESIZE, render_params_);
}

void CPUPipelineExecutor::SetRenderRes(bool full_res, int max_side_length) {
  auto& resize_params           = render_params_["resize"];
  // render_params_["resize"] = {
  //   {"enable_scale", true},
  //   {"maximum_edge", max_side_length},
  // };
  resize_params["enable_scale"] = !full_res;
  resize_params["maximum_edge"] = max_side_length;

  stages_[static_cast<int>(PipelineStageName::Geometry_Adjustment)].SetOperator(
      OperatorType::RESIZE, render_params_);
}

void CPUPipelineExecutor::RegisterAllOperators() {
  // It is really silly to hardcode the operators here.
  // I should keep things more flexible in the future.
  auto& image_loading_stage = stages_[static_cast<int>(PipelineStageName::Image_Loading)];
  image_loading_stage.SetOperator(OperatorType::RAW_DECODE, {});

  auto& geometry_adjustment_stage =
      stages_[static_cast<int>(PipelineStageName::Geometry_Adjustment)];
  geometry_adjustment_stage.SetOperator(OperatorType::RESIZE, render_params_);

  auto& to_working_space_stage = stages_[static_cast<int>(PipelineStageName::To_WorkingSpace)];
  to_working_space_stage.SetOperator(OperatorType::TO_WS, {});

  auto& basic_adjustment_stage = stages_[static_cast<int>(PipelineStageName::Basic_Adjustment)];
  basic_adjustment_stage.SetOperator(OperatorType::EXPOSURE, {});
  basic_adjustment_stage.SetOperator(OperatorType::CONTRAST, {});
  basic_adjustment_stage.SetOperator(OperatorType::WHITE, {});
  basic_adjustment_stage.SetOperator(OperatorType::BLACK, {});
  basic_adjustment_stage.SetOperator(OperatorType::HIGHLIGHTS, {});
  basic_adjustment_stage.SetOperator(OperatorType::SHADOWS, {});
  basic_adjustment_stage.SetOperator(OperatorType::CURVE, {});

  auto& color_adjustment_stage = stages_[static_cast<int>(PipelineStageName::Color_Adjustment)];
  color_adjustment_stage.SetOperator(
      OperatorType::TINT,
      {});  // TODO: This thing should be in the raw decoding part. Future fix needed.
  color_adjustment_stage.SetOperator(OperatorType::SATURATION, {});
  color_adjustment_stage.SetOperator(OperatorType::VIBRANCE, {});
  color_adjustment_stage.SetOperator(OperatorType::HLS, {});
  color_adjustment_stage.SetOperator(OperatorType::COLOR_WHEEL, {});

  auto& detail_adjustment_stage = stages_[static_cast<int>(PipelineStageName::Detail_Adjustment)];
  detail_adjustment_stage.SetOperator(OperatorType::SHARPEN, {});
  detail_adjustment_stage.SetOperator(OperatorType::CLARITY, {});
}

};  // namespace puerhlab