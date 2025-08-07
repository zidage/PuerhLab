#include "edit/pipeline/pipeline_cpu.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/op_base.hpp"
#include "edit/pipeline/pipeline_utils.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
CPUPipeline::CPUPipeline()
    : _stages({{PipelineStageName::To_WorkingSpace, false},
               {PipelineStageName::Basic_Adjustment, false},
               {PipelineStageName::Color_Adjustment, false},
               {PipelineStageName::Detail_Adjustment, false},
               {PipelineStageName::Output_Transform, false},
               {PipelineStageName::Geometry_Adjustment, false}}) {}

auto CPUPipeline::GetStage(PipelineStageName stage) -> PipelineStage& {
  return _stages[static_cast<int>(stage)];
}

auto CPUPipeline::Apply(ImageBuffer& input) -> ImageBuffer {
  auto output = ImageBuffer{input.GetCPUData().clone()};
  for (auto& stage : _stages) {
    if (stage._stage == PipelineStageName::Basic_Adjustment) {
      cv::cvtColor(output.GetCPUData(), output.GetCPUData(), cv::COLOR_RGB2Lab);
      std::vector<cv::Mat> channels;
      cv::split(output.GetCPUData(), channels);
      cv::Mat L_channel = channels[0];
      stage.SetInputImage(std::move(L_channel));
      auto modified_L = stage.ApplyStage();

      channels[0]     = modified_L.GetCPUData();
      cv::merge(channels, output.GetCPUData());
      cv::cvtColor(output.GetCPUData(), output.GetCPUData(), cv::COLOR_Lab2RGB);
    } else {
      stage.SetInputImage(std::move(output));
      output = stage.ApplyStage();
    }
  }
  return output;
}
};  // namespace puerhlab