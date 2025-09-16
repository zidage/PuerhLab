#include "edit/pipeline/pipeline_cpu.hpp"

#include <easy/profiler.h>

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/op_base.hpp"
#include "edit/operators/utils/color_conversion.hpp"
#include "edit/pipeline/pipeline_utils.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
CPUPipelineExecutor::CPUPipelineExecutor()
    : _stages({{PipelineStageName::Image_Loading, false},
               {PipelineStageName::To_WorkingSpace, false},
               {PipelineStageName::Basic_Adjustment, false},
               {PipelineStageName::Color_Adjustment, false},
               {PipelineStageName::Detail_Adjustment, false},
               {PipelineStageName::Output_Transform, false},
               {PipelineStageName::Geometry_Adjustment, false}}) {
  // Link stages
  for (size_t i = 0; i < _stages.size(); i++) {
    PipelineStage* prev = (i == 0) ? nullptr : &_stages[i - 1];
    PipelineStage* next = (i == _stages.size() - 1) ? nullptr : &_stages[i + 1];
    _stages[i].SetNeighbors(prev, next);
  }
}

void CPUPipelineExecutor::ResetStages() {
  for (size_t i = 0; i < _stages.size(); i++) {
    _stages[i]          = PipelineStage(_stages[i]._stage, _enable_cache);
    PipelineStage* prev = (i == 0) ? nullptr : &_stages[i - 1];
    PipelineStage* next = (i == _stages.size() - 1) ? nullptr : &_stages[i + 1];
    _stages[i].SetNeighbors(prev, next);
  }
}

void CPUPipelineExecutor::SetEnableCache(bool enable_cache) {
  if (_enable_cache == enable_cache) return;
  _enable_cache = enable_cache;
  // Reinitialize stages with the new cache setting
  ResetStages();
}

CPUPipelineExecutor::CPUPipelineExecutor(bool enable_cache)
    : _stages({{PipelineStageName::Image_Loading, enable_cache},
               {PipelineStageName::To_WorkingSpace, enable_cache},
               {PipelineStageName::Basic_Adjustment, enable_cache},
               {PipelineStageName::Color_Adjustment, enable_cache},
               {PipelineStageName::Detail_Adjustment, enable_cache},
               {PipelineStageName::Output_Transform, enable_cache},
               {PipelineStageName::Geometry_Adjustment, enable_cache}}) {
  // Link stages
  for (size_t i = 0; i < _stages.size(); i++) {
    PipelineStage* prev = (i == 0) ? nullptr : &_stages[i - 1];
    PipelineStage* next = (i == _stages.size() - 1) ? nullptr : &_stages[i + 1];
    _stages[i].SetNeighbors(prev, next);
  }
}

auto CPUPipelineExecutor::GetBackend() -> PipelineBackend { return _backend; }

auto CPUPipelineExecutor::GetStage(PipelineStageName stage) -> PipelineStage& {
  return _stages[static_cast<int>(stage)];
}

auto CPUPipelineExecutor::Apply(ImageBuffer& input) -> ImageBuffer {
  auto output = std::make_shared<ImageBuffer>(input.Clone());

  for (auto& stage : _stages) {
    EASY_NONSCOPED_BLOCK(std::format("Apply stage: {}", stage.GetStageNameString()).c_str(),
                         profiler::colors::Red);
    stage.SetInputImage(output);
    output = stage.ApplyStage();
    EASY_END_BLOCK
  }
  return output->Clone();
}

void CPUPipelineExecutor::SetThumbnailMode(bool is_thumbnail) {
  _is_thumbnail     = is_thumbnail;
  _thumbnail_params = {};  // TODO: Use default params for now
  if (!_is_thumbnail) {
    // Disable resizing in image loading stage
    _stages[static_cast<int>(PipelineStageName::Image_Loading)].EnableOperator(
        OperatorType::RESIZE,
        false);  // If RESIZE operator not exist, this function will do nothing
    return;
  }
  _stages[static_cast<int>(PipelineStageName::Image_Loading)].SetOperator(OperatorType::RESIZE,
                                                                          _thumbnail_params);
}
};  // namespace puerhlab