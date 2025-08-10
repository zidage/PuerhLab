#include "edit/pipeline/pipeline_cpu.hpp"

#include <easy/profiler.h>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/op_base.hpp"
#include "edit/operators/utils/color_conversion.hpp"
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
      EASY_BLOCK("To Lab Color Space");
      cv::UMat uSrc, uDst;
      output.GetCPUData().copyTo(uSrc);
      GPUCvtColor(uSrc, uDst, cv::COLOR_RGB2Lab);

      std::vector<cv::UMat> channels;
      cv::split(uDst, channels);
      cv::Mat L_channel;
      channels[0].copyTo(L_channel);
      EASY_END_BLOCK;
      stage.SetInputImage(std::move(L_channel));
      auto modified_L = stage.ApplyStage();

      EASY_BLOCK("To RGB Color Space");
      modified_L.GetCPUData().copyTo(channels[0]);
      cv::merge(channels, uSrc);
      cv::cvtColor(uSrc, uDst, cv::COLOR_Lab2RGB);
      uDst.copyTo(output.GetCPUData());
      EASY_END_BLOCK;
    } else {
      stage.SetInputImage(std::move(output));
      output = stage.ApplyStage();
    }
  }
  return output;
}
};  // namespace puerhlab