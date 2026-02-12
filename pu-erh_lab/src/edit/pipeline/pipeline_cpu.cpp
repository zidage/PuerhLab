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

#include <algorithm>
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

  // Initialize default pipeline
  InitDefaultPipeline();
}

void CPUPipelineExecutor::ResetExecutionStagesCache() {
  for (auto& stage : exec_stages_) {
    stage->ResetCache();
  }
}

void CPUPipelineExecutor::ResetStages() {
  for (size_t i = 0; i < stages_.size(); i++) {
    stages_[i].ResetAll();
  }
}

void CPUPipelineExecutor::SetEnableCache(bool enable_cache) {
  if (enable_cache_ == enable_cache && enable_cache) return;
  enable_cache_ = enable_cache;
  // Reinitialize stages with the new cache setting
  ResetExecutionStagesCache();
  for (auto* stage : exec_stages_) {
    const bool stage_cache_enabled = (stage->stage_ == PipelineStageName::Merged_Stage)
                                         ? false
                                         : enable_cache_;
    stage->SetEnableCache(stage_cache_enabled);
  }
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
  // Initialize default pipeline
  InitDefaultPipeline();
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
  if (enable_cache_) {
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
  } else {
    // Cache is disabled, just process the stages sequentially
    output = std::make_shared<ImageBuffer>(input->Clone());
    for (auto* stage : exec_stages_) {
      stage->SetInputImage(output);
      stage->SetForceCPUOutput(force_cpu_output_);
      output = stage->ApplyStage(global_params_);
    }
  }

  return output;
}

[[deprecated("SetPreviewMode is deprecated, set from pipeline scheduler instead")]] void
CPUPipelineExecutor::SetPreviewMode(bool) {
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
  frame_sink_ = nullptr;
  std::vector<PipelineStage*> streamable_stages;

  // Merged GPU stream stage is always re-executed; keeping its cache enabled only retains
  // transient objects without reuse value.
  auto merged = std::make_unique<PipelineStage>(PipelineStageName::Merged_Stage, false, true);

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
  frame_sink_ = frame_sink;

  // Set frame sink for the last stage
  if (!exec_stages_.empty()) {
    exec_stages_.back()->SetFrameSink(frame_sink);
  }
}

void CPUPipelineExecutor::ResetExecutionStages() {
  frame_sink_ = nullptr;
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
      stage.ImportStageParams(stage_json, global_params_);
    }
  }
}

void CPUPipelineExecutor::SetRenderRegion(int x, int y, float scale_factor_x, float scale_factor_y) {
  auto& resize_params = render_params_["resize"];

  const float clamped_scale_x = std::clamp(scale_factor_x, 1e-4f, 1.0f);
  const float clamped_scale_y =
      std::clamp((scale_factor_y > 0.0f) ? scale_factor_y : scale_factor_x, 1e-4f, 1.0f);
  resize_params["enable_roi"] = (clamped_scale_x < (1.0f - 1e-4f)) ||
                                (clamped_scale_y < (1.0f - 1e-4f));
  resize_params["roi"]        = {{"x", std::max(0, x)},
                                 {"y", std::max(0, y)},
                                 {"resize_factor_x", clamped_scale_x},
                                 {"resize_factor_y", clamped_scale_y},
                                 {"resize_factor", std::max(clamped_scale_x, clamped_scale_y)}};

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

void CPUPipelineExecutor::SetDecodeRes(DecodeRes res) {
  if (decode_res_ == res) {
    return;
  }
  decode_res_     = res;

  // TODO: Abstraction leak, need better design later
  auto& raw_stage = GetStage(PipelineStageName::Image_Loading);
  auto  raw_param = raw_stage.GetOperator(OperatorType::RAW_DECODE).value()->ExportOperatorParams();
  raw_param["params"]["raw"]["decode_res"] = static_cast<int>(res);
  raw_stage.SetOperator(OperatorType::RAW_DECODE, raw_param["params"]);
}

auto CPUPipelineExecutor::GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> {
  if (!frame_sink_) {
    return std::nullopt;
  }
  return frame_sink_->GetViewportRenderRegion();
}

void CPUPipelineExecutor::SetNextFramePresentationMode(FramePresentationMode mode) const {
  if (!frame_sink_) {
    return;
  }
  frame_sink_->SetNextFramePresentationMode(mode);
}

void CPUPipelineExecutor::RegisterAllOperators() {
  // It is really silly to hardcode the operators here.
  // I should keep things more flexible in the future.

  auto& basic_adjustment_stage = stages_[static_cast<int>(PipelineStageName::Basic_Adjustment)];
  basic_adjustment_stage.SetOperator(OperatorType::EXPOSURE, {{"exposure", 2.0f}}, global_params_);
  basic_adjustment_stage.SetOperator(OperatorType::CONTRAST, {}, global_params_);
  basic_adjustment_stage.SetOperator(OperatorType::WHITE, {}, global_params_);
  basic_adjustment_stage.SetOperator(OperatorType::BLACK, {}, global_params_);
  basic_adjustment_stage.SetOperator(OperatorType::HIGHLIGHTS, {}, global_params_);
  basic_adjustment_stage.SetOperator(OperatorType::SHADOWS, {}, global_params_);
  basic_adjustment_stage.SetOperator(OperatorType::CURVE, {}, global_params_);

  auto& color_adjustment_stage = stages_[static_cast<int>(PipelineStageName::Color_Adjustment)];
  color_adjustment_stage.SetOperator(
      OperatorType::TINT,
      {}, global_params_);  // TODO: This thing should be in the raw decoding part. Future fix needed.
  color_adjustment_stage.SetOperator(OperatorType::SATURATION, {{"saturation", 30.0f}}, global_params_);
  color_adjustment_stage.SetOperator(OperatorType::VIBRANCE, {}, global_params_);
  color_adjustment_stage.SetOperator(OperatorType::HLS, {}, global_params_);
  color_adjustment_stage.SetOperator(OperatorType::COLOR_WHEEL, {}, global_params_);

  auto& detail_adjustment_stage = stages_[static_cast<int>(PipelineStageName::Detail_Adjustment)];
  detail_adjustment_stage.SetOperator(OperatorType::SHARPEN, {}, global_params_);
  detail_adjustment_stage.SetOperator(OperatorType::CLARITY, {}, global_params_);
}

void CPUPipelineExecutor::SetTemplateParams() {
  // Set some common parameters for template pipelines
  auto&          raw_stage     = GetStage(PipelineStageName::Image_Loading);
  auto&          global_params = GetGlobalParams();
  nlohmann::json decode_params;
#ifdef HAVE_CUDA
  decode_params["raw"]["cuda"] = true;
#else
  decode_params["raw"]["cuda"] = false;
#endif
  decode_params["raw"]["highlights_reconstruct"] = false;
  decode_params["raw"]["use_camera_wb"]          = true;
  decode_params["raw"]["user_wb"]                = 7600.f;
  decode_params["raw"]["backend"]                = "puerh";
  raw_stage.SetOperator(OperatorType::RAW_DECODE, decode_params);

  nlohmann::json output_params;
  auto&          output_stage = GetStage(PipelineStageName::Output_Transform);
  output_params["aces_odt"]   = {{"encoding_space", "rec709"},
                                 {"encoding_etof", "gamma_2_2"},
                                 {"limiting_space", "rec709"},
                                 {"peak_luminance", 100.0f}};
  output_stage.SetOperator(OperatorType::ODT, output_params, global_params);
}

void CPUPipelineExecutor::InitDefaultPipeline() {
  SetTemplateParams();
  RegisterAllOperators();
  SetExecutionStages();
}

void CPUPipelineExecutor::ClearAllIntermediateBuffers() {
  for (auto& stage : exec_stages_) {
    stage->ClearIntermediateBuffers();
  }
  ReleaseAllGPUResources();

  if (merged_stages_) {
    merged_stages_->ClearIntermediateBuffers();
  }
}

void CPUPipelineExecutor::ReleaseAllGPUResources() {
  for (auto& stage : exec_stages_) {
    stage->ReleaseGPUResources();
  }

  if (merged_stages_) {
    merged_stages_->ReleaseGPUResources();
  }
}

};  // namespace puerhlab
