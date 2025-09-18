#ifdef _WIN32
#include "edit/scheduler/pipeline_scheduler.hpp"

#include <memory>
#include <opencv2/cudaimgproc.hpp>
#include <string>

#include "decoders/raw_decoder.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "image/image_buffer.hpp"
#include "pipeline_test_fixation.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "type/supported_file_type.hpp"
#include "utils/clock/time_provider.hpp"
#include "utils/string/convert.hpp"

using namespace puerhlab;

#include <lcms2.h>

#include <algorithm>
#include <fstream>

using namespace puerhlab;
void SetPipelineStages(std::shared_ptr<PipelineExecutor> executor) {
  auto&          raw_stage = executor->GetStage(PipelineStageName::Image_Loading);
  nlohmann::json decode_params;
  decode_params["raw"]["cuda"]                   = true;
  decode_params["raw"]["highlights_reconstruct"] = false;
  decode_params["raw"]["use_camera_wb"]          = true;
  decode_params["raw"]["backend"]                = "puerh";
  raw_stage.SetOperator(OperatorType::RAW_DECODE, decode_params);

  nlohmann::json to_ws_params;
  to_ws_params["ocio"] = {
      {"src", "Linear Rec.709 (sRGB)"}, {"dst", "ACEScct"}, {"normalize", true}};

  auto&          to_ws = executor->GetStage(PipelineStageName::To_WorkingSpace);
  to_ws.SetOperator(OperatorType::CST, to_ws_params);

  nlohmann::json output_params;
  auto& output_stage = executor->GetStage(PipelineStageName::Output_Transform);
  output_params["ocio"]       = {{"src", "ACEScct"}, {"dst", "Camera Rec.709"}, {"limit", true}};
  output_stage.SetOperator(OperatorType::CST, output_params);
}

TEST_F(PipelineTests, SchedulerBasic) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path = std::string(TEST_IMG_PATH) + std::string("raw/building");
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory() && is_supported_file(img.path())) imgs.push_back(img.path());
    }

    manager.LoadToPath(imgs, L"");

    // Read image data
    auto              img_pool = manager.GetPool()->GetPool();

    PipelineScheduler scheduler{};
    for (const auto& img : img_pool) {
      auto         img_ptr = img.second;
      PipelineTask task;

      auto         buffer        = ByteBufferLoader::LoadFromImage(img_ptr);
      task._input                = std::make_shared<ImageBuffer>(std::move(buffer));
      task._pipeline_executor    = std::make_shared<CPUPipelineExecutor>();
      SetPipelineStages(task._pipeline_executor);

      task._options._is_blocking = false;
      task._options._is_callback = true;

      task._callback             = [img_ptr](ImageBuffer& output) {
        output.SyncToGPU();

        auto&            gpu_data = output.GetGPUData();
        gpu_data.convertTo(gpu_data, CV_16UC3, 65535.0f);
        cv::cuda::cvtColor(gpu_data, gpu_data, cv::COLOR_RGB2BGR);

        output.SyncToCPU();

        std::string file_name = conv::ToBytes(img_ptr->_image_path.filename().wstring());
        std::string time      = TimeProvider::TimePointToString(TimeProvider::Now());

        std::string save_name = file_name + "_" + time;
        static constexpr auto save_path = TEST_IMG_PATH "/my_pipeline/batch_results/{}.tif";
        cv::imwrite(std::format(save_path, save_name), output.GetCPUData());
      };

      scheduler.ScheduleTask(std::move(task));
    }
  }
}
#endif