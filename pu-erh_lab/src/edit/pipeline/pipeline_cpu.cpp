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
    : _enable_cache(false),
      _stages({{PipelineStageName::Image_Loading, _enable_cache, false},
               {PipelineStageName::Geometry_Adjustment, _enable_cache, false},
               {PipelineStageName::To_WorkingSpace, _enable_cache, true},
               {PipelineStageName::Basic_Adjustment, _enable_cache, true},
               {PipelineStageName::Color_Adjustment, _enable_cache, true},
               {PipelineStageName::Detail_Adjustment, _enable_cache, true},
               {PipelineStageName::Output_Transform, _enable_cache, true}}) {}

void CPUPipelineExecutor::ResetStages() {
  for (size_t i = 0; i < _stages.size(); i++) {
    _stages[i].ResetAll();
  }
}

void CPUPipelineExecutor::SetEnableCache(bool enable_cache) {
  if (_enable_cache == enable_cache) return;
  _enable_cache = enable_cache;
  // Reinitialize stages with the new cache setting
  ResetStages();
}

CPUPipelineExecutor::CPUPipelineExecutor(bool enable_cache)
    : _enable_cache(enable_cache),
      _stages({{PipelineStageName::Image_Loading, _enable_cache, false},
               {PipelineStageName::Geometry_Adjustment, _enable_cache, false},
               {PipelineStageName::To_WorkingSpace, _enable_cache, true},
               {PipelineStageName::Basic_Adjustment, _enable_cache, true},
               {PipelineStageName::Color_Adjustment, _enable_cache, true},
               {PipelineStageName::Detail_Adjustment, _enable_cache, true},
               {PipelineStageName::Output_Transform, _enable_cache, true}}) {}

auto CPUPipelineExecutor::GetBackend() -> PipelineBackend { return _backend; }

auto CPUPipelineExecutor::GetStage(PipelineStageName stage) -> PipelineStage& {
  return _stages[static_cast<int>(stage)];
}

auto CPUPipelineExecutor::Apply(std::shared_ptr<ImageBuffer> input)
    -> std::shared_ptr<ImageBuffer> {
  auto* first_stage = _exec_stages.front();
  if (!first_stage) {
    return input;
  }
  std::shared_ptr<ImageBuffer> output;
  if (!first_stage->CacheValid()) {
    output = std::make_shared<ImageBuffer>(input->Clone());
    for (auto* stage : _exec_stages) {
      stage->SetInputImage(output);
      output = stage->ApplyStage(_global_params);
    }
  } else {
    // If cache is valid, use cached output
    output = first_stage->GetOutputCache();
    for (auto* stage : _exec_stages) {
      if (stage != first_stage) {
        stage->SetInputImage(output);
        output = stage->ApplyStage(_global_params);
      }
    }
  }

  return output;
}

void CPUPipelineExecutor::SetPreviewMode(bool is_thumbnail) {
  _is_thumbnail     = is_thumbnail;

  _thumbnail_params = {};  // TODO: Use default params for now
  if (!_is_thumbnail) {
    // Disable resizing in image loading stage
    _stages[static_cast<int>(PipelineStageName::Image_Loading)].EnableOperator(
        OperatorType::RESIZE,
        false);  // If RESIZE operator not exist, this function will do nothing
    return;
  }
  _stages[static_cast<int>(PipelineStageName::Geometry_Adjustment)].SetOperator(
      OperatorType::RESIZE, _thumbnail_params);
}

void CPUPipelineExecutor::SetExecutionStages() {
  _exec_stages.clear();
  std::vector<PipelineStage*> streamable_stages;

  auto merged = std::make_unique<PipelineStage>(PipelineStageName::Merged_Stage, true, true);

  _exec_stages.push_back(&_stages[static_cast<int>(PipelineStageName::Image_Loading)]);
  _exec_stages.push_back(&_stages[static_cast<int>(PipelineStageName::Geometry_Adjustment)]);
  _exec_stages.push_back(merged.get());

  _merged_stages = std::move(merged);

  for (size_t i = 2; i < _stages.size(); i++) {
    PipelineStage& stage = _stages[i];
    stage.AddDependent(_merged_stages.get());
  }

  // Chain the execution stages
  for (size_t i = 0; i < _exec_stages.size(); i++) {
    PipelineStage* prev_stage = (i > 0) ? _exec_stages[i - 1] : nullptr;
    PipelineStage* next_stage = (i < _exec_stages.size() - 1) ? _exec_stages[i + 1] : nullptr;
    _exec_stages[i]->SetNeighbors(prev_stage, next_stage);
  }
}

void CPUPipelineExecutor::ResetExecutionStages() {
  for (auto& stage : _stages) {
    stage.ResetDependents();
    stage.ResetNeighbors();
    stage.ResetCache();
  }
  _exec_stages.clear();
  _merged_stages.reset();
}

auto CPUPipelineExecutor::ExportPipelineParams() const -> nlohmann::json {
  nlohmann::json j;
  for (const auto& stage : _stages) {
    nlohmann::json stage_json     = stage.ExportStageParams();
    j[stage.GetStageNameString()] = std::move(stage_json);
  }
  return j;
}

void CPUPipelineExecutor::ImportPipelineParams(const nlohmann::json& j) {
  ResetExecutionStages();
  for (auto& stage : _stages) {
    std::string stage_name = stage.GetStageNameString();
    if (j.contains(stage_name)) {
      nlohmann::json stage_json = j[stage_name];
      // When importing, stage's import function will do reset internally
      stage.ImportStageParams(stage_json);
    }
  }
}

};  // namespace puerhlab