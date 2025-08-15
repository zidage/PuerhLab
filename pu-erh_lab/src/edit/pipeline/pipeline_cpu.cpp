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
      output.SyncToGPU();
      auto& gpu_data = output.GetGPUData();
      cv::cuda::cvtColor(gpu_data, gpu_data, cv::COLOR_RGB2Lab);
      std::vector<cv::cuda::GpuMat> channels;
      cv::cuda::split(gpu_data, channels);
      ImageBuffer L_channel{std::move(channels[0])};
      L_channel.SyncToCPU();
      EASY_END_BLOCK;

      stage.SetInputImage(std::move(L_channel));
      auto modified_L = stage.ApplyStage();

      EASY_BLOCK("To RGB Color Space");
      modified_L.SyncToGPU();
      channels[0] = modified_L.GetGPUData();
      cv::cuda::merge(channels, gpu_data);
      cv::cuda::cvtColor(gpu_data, gpu_data, cv::COLOR_Lab2RGB);
      output.SyncToCPU();
      EASY_END_BLOCK;
    } else {
      stage.SetInputImage(std::move(output));
      output = stage.ApplyStage();
    }
  }
  return output;
}
};  // namespace puerhlab