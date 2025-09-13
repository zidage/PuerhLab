#include "edit/pipeline/pipeline_cpu.hpp"

#include <easy/profiler.h>

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
               {PipelineStageName::Geometry_Adjustment, false}}) {}

auto CPUPipelineExecutor::GetStage(PipelineStageName stage) -> PipelineStage& {
  return _stages[static_cast<int>(stage)];
}

auto CPUPipelineExecutor::Apply(ImageBuffer& input) -> ImageBuffer {
  ImageBuffer output = input.Clone();

  for (auto& stage : _stages) {
    EASY_NONSCOPED_BLOCK(std::format("Apply stage: {}", stage.GetStageNameString()).c_str(),
                         profiler::colors::Red);
    stage.SetInputImage(std::move(output));
    output = stage.ApplyStage();
    EASY_END_BLOCK
  }
  return output;
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