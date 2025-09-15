#include "edit/pipeline/preview_pipeline_cpu.hpp"

#include <future>

#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/preview_pipeline_utils.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
CPU_PreviewPipelineExecutor::CPU_PreviewPipelineExecutor()
    : _stages({{PipelineStageName::Image_Loading},
               {PipelineStageName::To_WorkingSpace},
               {PipelineStageName::Basic_Adjustment},
               {PipelineStageName::Color_Adjustment},
               {PipelineStageName::Detail_Adjustment},
               {PipelineStageName::Output_Transform},
               {PipelineStageName::Geometry_Adjustment}}) {
  for (size_t i = 0; i < _stages.size(); i++) {
    PreviewPipelineStage* next = (i == _stages.size() - 1) ? nullptr : &_stages[i + 1];
    _stages[i].SetNextStage(next);
  }
  // Set default params for thumbnail mode, use default params for now
  _stages[static_cast<int>(PipelineStageName::Image_Loading)].SetOperator(OperatorType::RESIZE,
                                                                          _thumbnail_params);
}

auto CPU_PreviewPipelineExecutor::GetBackend() -> PipelineBackend { return _backend; }

auto CPU_PreviewPipelineExecutor::GetStage(PipelineStageName) -> PipelineStage& {
  throw std::runtime_error(
      "Use GetPreviewStage instead of GetStage for CPU_PreviewPipelineExecutor");
}

auto CPU_PreviewPipelineExecutor::GetPreviewStage(PipelineStageName stage)
    -> PreviewPipelineStage& {
  return _stages[static_cast<int>(stage)];
}

auto CPU_PreviewPipelineExecutor::Apply(ImageBuffer&) -> ImageBuffer {
  throw std::runtime_error(
      "Use Apply with FrameId instead of Apply without FrameId for CPU_PreviewPipelineExecutor");
}

auto CPU_PreviewPipelineExecutor::ExecuteStage(size_t idx, uint64_t frame_id,
                                               std::shared_ptr<ImageBuffer> input)
    -> std::shared_future<std::shared_ptr<ImageBuffer>> {
  if (idx >= _stages.size()) {
    std::promise<std::shared_ptr<ImageBuffer>> p;
    p.set_value(input);
    return p.get_future().share();
  }

  _stages[idx].SetInputImage(frame_id, input);
  auto output = _stages[idx].ApplyStage(frame_id);
  return ExecuteStage(idx + 1, frame_id, output);
}

auto CPU_PreviewPipelineExecutor::Apply(FrameId id, ImageBuffer& input)
    -> std::shared_future<ImageBuffer> {}

};  // namespace puerhlab