#include "decoders/raw_decoder.hpp"
#include "edit/operators/detail/clarity_op.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "pipeline_test_fixation.hpp"
#include "sleeve/sleeve_manager.hpp"
#include "utils/string/convert.hpp"

using namespace puerhlab;
TEST_F(PipelineTests, SimpleTest1) {
  {
    SleeveManager manager{db_path_};
    ImageLoader   image_loader(128, 8, 0);
    image_path_t  path =
        L"D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_images\\real_test\\light";
    std::vector<image_path_t> imgs;
    for (const auto& img : std::filesystem::directory_iterator(path)) {
      if (!img.is_directory()) imgs.push_back(img.path());
    }

    manager.LoadToPath(imgs, L"");

    // Read image data
    ThreadPool     thread_pool{8};
    auto           img_pool = manager.GetPool()->GetPool();

    nlohmann::json to_ws_params;
    to_ws_params["ocio"] = {{"src", ""}, {"dst", "ACEScct"}};

    nlohmann::json exposure_params;
    exposure_params["exposure"] = 0.15;

    nlohmann::json white_params;
    white_params["tone_region"] = {{"region", "white"}, {"offset", -50}};

    nlohmann::json output_params;
    output_params["ocio"] = {{"src", "ACEScct"}, {"dst", "Camera Rec.709"}};

    for (auto& pair : img_pool) {
      auto task = [pair, img_pool, to_ws_params, white_params, exposure_params,
                   output_params]() mutable {
        CPUPipeline pipeline{};
        auto&       to_ws = pipeline.GetStage(PipelineStageName::To_WorkingSpace);
        to_ws.SetOperator(OperatorType::CST, to_ws_params);

        auto& adj = pipeline.GetStage(PipelineStageName::Basic_Adjustment);
        adj.SetOperator(OperatorType::TONE_REGION, white_params);
        adj.SetOperator(OperatorType::EXPOSURE, exposure_params);

        auto& output_stage = pipeline.GetStage(PipelineStageName::Output_Transform);
        output_stage.SetOperator(OperatorType::CST, output_params);
        auto            img = pair.second;
        RawDecoder      decoder;
        std::ifstream   file(img->_image_path, std::ios::binary | std::ios::ate);
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(fileSize);
        if (!file.read(buffer.data(), fileSize)) {
          FAIL();
        }
        file.close();

        decoder.Decode(std::move(buffer), img);

        ImageBuffer source{img->GetImageData()};
        auto        output = pipeline.Apply(source);
        cv::Mat     to_save_rec709;
        output.GetCPUData().convertTo(to_save_rec709, CV_16UC3, 65535.0f);
        cv::cvtColor(to_save_rec709, to_save_rec709, cv::COLOR_RGB2BGR);
        cv::imwrite(std::format("D:\\Projects\\pu-erh_lab\\pu-erh_lab\\tests\\resources\\sample_"
                                "images\\my_pipeline\\batch_results\\{}.tiff",
                                conv::ToBytes(img->_image_name)),
                    to_save_rec709);
        img->ClearData();
        to_save_rec709.release();
        output.ReleaseCPUData();
      };
      thread_pool.Submit(task);
    }
  }
}