#include "edit/pipeline/pipeline_cpu.hpp"

#include "image/image_buffer.hpp"

namespace puerhlab {
CPUPipeline::CPUPipeline()
    : _stages({{PipelineStageName::To_WorkingSpace, false},
               {PipelineStageName::Basic_Adjustment, false},
               {PipelineStageName::Color_Adjustment, false},
               {PipelineStageName::Detail_Adjustment, false},
               {PipelineStageName::Output_Transform, false},
               {PipelineStageName::Geometry_Adjustment, false}}) {}

auto CPUPipeline::Apply(ImageBuffer& input) -> ImageBuffer {
  auto output = ImageBuffer{input.GetCPUData().clone()};
  for (auto& stage : _stages) {
    stage.SetInputImage(std::move(output));
    output = stage.ApplyStage();
  }
  return output;
}
};  // namespace puerhlab